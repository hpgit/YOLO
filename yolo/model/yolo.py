from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from omegaconf import ListConfig, OmegaConf
from torch import nn

from yolo.config.config import ModelConfig, YOLOLayer
from yolo.model.module import Conv, MultiheadDetection, RepConv
from yolo.tools.dataset_preparation import prepare_weight
from yolo.utils.logger import logger
from yolo.utils.module_utils import create_activation_function, get_layer_map


class YOLO(nn.Module):
    """
    A preliminary YOLO (You Only Look Once) model class still under development.

    Parameters:
        model_cfg: Configuration for the YOLO model. Expected to define the layers,
                   parameters, and any other relevant configuration details.
    """

    def __init__(self, model_cfg: ModelConfig, class_num: int = 80):
        super(YOLO, self).__init__()
        self.num_classes = class_num
        self.model_name = model_cfg.name
        self.layer_map = get_layer_map()  # Get the map Dict[str: Module]
        self.model: List[YOLOLayer] = nn.ModuleList()
        self.reg_max = getattr(model_cfg.anchor, "reg_max", 16)
        self.nms_free = getattr(model_cfg, "nms_free", False)
        if not isinstance(self.nms_free, bool):
            raise ValueError("model.nms_free must be a boolean.")
        self.build_model(model_cfg.model)
        if self.nms_free:
            main_index = self.layer_index.get("Main")
            main_head = self.model[main_index - 1] if main_index is not None else None
            if not isinstance(main_head, MultiheadDetection) or not main_head.output:
                raise ValueError("NMS-free detection requires a Main MultiheadDetection output.")
            main_head.enable_nms_free()
        activation = getattr(model_cfg, "activation", None)
        if activation is not None:
            # Include nested backbone/neck and Main/AUX head convolutions, while
            # preserving the linear branches inside RepConv for deploy fusion.
            for module in self.modules():
                if isinstance(module, (Conv, RepConv)) and not isinstance(module.act, nn.Identity):
                    module.act = create_activation_function(activation)

    def build_model(self, model_arch: Dict[str, List[Dict[str, Dict[str, Dict]]]]):
        self.layer_index = {}
        output_dim, layer_idx = [3], 1
        logger.info(f":tractor: Building YOLO")
        for arch_name in model_arch:
            if model_arch[arch_name]:
                logger.info(f"  :building_construction:  Building {arch_name}")
            for layer_idx, layer_spec in enumerate(model_arch[arch_name], start=layer_idx):
                layer_type, layer_info = next(iter(layer_spec.items()))
                layer_args = layer_info.get("args", {})

                # Get input source
                source = self.get_source_idx(layer_info.get("source", -1), layer_idx)

                # Find in channels
                if any(module in layer_type for module in ["Conv", "ELAN", "ADown", "AConv", "CBLinear"]):
                    layer_args["in_channels"] = output_dim[source]
                if any(module in layer_type for module in ["Detection", "Segmentation", "Classification"]):
                    if isinstance(source, list):
                        layer_args["in_channels"] = [output_dim[idx] for idx in source]
                    else:
                        layer_args["in_channel"] = output_dim[source]
                    layer_args["num_classes"] = self.num_classes
                    layer_args["reg_max"] = self.reg_max

                # create layers
                layer = self.create_layer(layer_type, source, layer_info, **layer_args)
                self.model.append(layer)

                if layer.tags:
                    if layer.tags in self.layer_index:
                        raise ValueError(f"Duplicate tag '{layer_info['tags']}' found.")
                    self.layer_index[layer.tags] = layer_idx

                out_channels = self.get_out_channels(layer_type, layer_args, output_dim, source)
                output_dim.append(out_channels)
                setattr(layer, "out_c", out_channels)
            layer_idx += 1

    def forward(self, x, external: Optional[Dict] = None, shortcut: Optional[str] = None):
        if self.nms_free and not self.training and shortcut is None:
            shortcut = "Main"
        y = {0: x, **(external or {})}
        output = dict()
        for index, layer in enumerate(self.model, start=1):
            if isinstance(layer.source, list):
                model_input = [y[idx] for idx in layer.source]
            else:
                model_input = y[layer.source]

            external_input = {source_name: y[source_name] for source_name in layer.external}

            x = layer(model_input, **external_input)
            y[-1] = x
            if layer.usable:
                y[index] = x
            if layer.output:
                if self.nms_free and layer.tags == "Main" and isinstance(x, dict):
                    output.update(x)
                else:
                    output[layer.tags] = x
                if layer.tags == shortcut:
                    return output
        return output

    def get_out_channels(self, layer_type: str, layer_args: dict, output_dim: list, source: Union[int, list]):
        if hasattr(layer_args, "out_channels"):
            return layer_args["out_channels"]
        if layer_type == "CBFuse":
            return output_dim[source[-1]]
        if isinstance(source, int):
            return output_dim[source]
        if isinstance(source, list):
            return sum(output_dim[idx] for idx in source)

    def get_source_idx(self, source: Union[ListConfig, str, int], layer_idx: int):
        if isinstance(source, ListConfig):
            return [self.get_source_idx(index, layer_idx) for index in source]
        if isinstance(source, str):
            source = self.layer_index[source]
        if source < -1:
            source += layer_idx
        if source > 0:  # Using Previous Layer's Output
            self.model[source - 1].usable = True
        return source

    def create_layer(self, layer_type: str, source: Union[int, list], layer_info: Dict, **kwargs) -> YOLOLayer:
        if layer_type in self.layer_map:
            layer = self.layer_map[layer_type](**kwargs)
            setattr(layer, "layer_type", layer_type)
            setattr(layer, "source", source)
            setattr(layer, "in_c", kwargs.get("in_channels", None))
            setattr(layer, "output", layer_info.get("output", False))
            setattr(layer, "tags", layer_info.get("tags", None))
            setattr(layer, "external", layer_info.get("external", []))
            setattr(layer, "usable", 0)
            return layer
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def save_load_weights(self, weights: Union[Path, OrderedDict]):
        """
        Update the model's weights with the provided weights.

        args:
            weights: A OrderedDict containing the new weights.
        """
        nms_free = getattr(self, "nms_free", False)
        if isinstance(weights, Path):
            weights = torch.load(weights, map_location=torch.device("cpu"), weights_only=False)
        checkpoint_state = weights.get("model_state_dict", weights.get("state_dict", weights))
        has_one2one = any(".one2one_heads." in name for name in checkpoint_state)
        if has_one2one and not nms_free:
            raise ValueError(
                "NMS-free checkpoint requires model.nms_free=true; refusing to discard one-to-one weights."
            )
        if "qat" in weights:
            if nms_free:
                raise ValueError("NMS-free detection currently supports floating-point checkpoints only, not QAT.")
            from yolo.tools.qat import load_qat_state

            load_qat_state(self, weights)
            return
        if "state_dict" in weights:
            weights = weights["state_dict"]
        # Accept released inner-module weights, YOLO state_dicts, and Lightning
        # checkpoints. Preserve both prediction branches for trained dual heads.
        weights = {name.removeprefix("model.model.").removeprefix("model."): tensor for name, tensor in weights.items()}
        model_state_dict = self.model.state_dict()
        if has_one2one:
            invalid_one2one = [
                name
                for name, tensor in model_state_dict.items()
                if ".one2one_heads." in name and (name not in weights or tensor.shape != weights[name].shape)
            ]
            if invalid_one2one:
                raise ValueError("NMS-free checkpoint has missing or incompatible one-to-one weights.")

        # TODO1: autoload old version weight
        # TODO2: weight transform if num_class difference

        error_dict = {"Mismatch": set(), "Not Found": set()}
        for model_key, model_weight in model_state_dict.items():
            if nms_free and not has_one2one and ".one2one_heads." in model_key:
                continue  # Initialized from the loaded dense branch below.
            if model_key not in weights:
                error_dict["Not Found"].add(tuple(model_key.split(".")[:-2]))
                continue
            if model_weight.shape != weights[model_key].shape:
                error_dict["Mismatch"].add(tuple(model_key.split(".")[:-2]))
                continue
            model_state_dict[model_key] = weights[model_key]

        for error_name, error_set in error_dict.items():
            error_dict = dict()
            for layer_idx, *layer_name in error_set:
                if layer_idx not in error_dict:
                    error_dict[layer_idx] = [".".join(layer_name)]
                else:
                    error_dict[layer_idx].append(".".join(layer_name))
            for layer_idx, layer_name in error_dict.items():
                layer_name.sort()
                logger.warning(f":warning: Weight {error_name} for Layer {layer_idx}: {', '.join(layer_name)}")

        self.model.load_state_dict(model_state_dict)
        if nms_free and not has_one2one:
            main_head = self.model[self.layer_index["Main"] - 1]
            main_head.one2one_heads.load_state_dict(main_head.heads.state_dict())
            logger.info("Initialized NMS-free one-to-one heads from loaded Main detection weights.")


def create_model(
    model_cfg: ModelConfig, weight_path: Union[bool, Path] = True, class_num: int = 80, qat_cfg=None
) -> YOLO:
    """Constructs and returns a model from a Dictionary configuration file.

    Args:
        config_file (dict): The configuration file of the model.

    Returns:
        YOLO: An instance of the model defined by the given configuration.
    """
    OmegaConf.set_struct(model_cfg, False)
    model = YOLO(model_cfg, class_num)
    if model.nms_free and qat_cfg is not None and qat_cfg.enabled:
        raise ValueError("NMS-free detection currently supports floating-point training only, not QAT.")
    if weight_path:
        if weight_path == True:
            weight_path = Path("weights") / f"{model_cfg.name}.pt"
        elif isinstance(weight_path, str):
            weight_path = Path(weight_path)

        if not weight_path.exists():
            logger.info(f"🌐 Weight {weight_path} not found, try downloading")
            prepare_weight(weight_path=weight_path)
        if weight_path.exists():
            model.save_load_weights(weight_path)
            logger.info(":white_check_mark: Success load model & weight")
    else:
        logger.info(":white_check_mark: Success load model")
    if qat_cfg is not None and qat_cfg.enabled:
        from yolo.tools.qat import prepare_qat

        prepare_qat(model, qat_cfg)
    return model
