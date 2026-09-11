"""Create a reproducible COCO 1/20 subset without copying image contents."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def prepare(source, output, seed=10, denominator=20):
    source, output = source.resolve(), output.resolve()
    if denominator < 1 or source == output:
        raise ValueError("Use a separate output directory and positive denominator")
    manifest = {"source": str(source), "seed": seed, "denominator": denominator, "splits": {}}
    for split in ("train2017", "val2017"):
        annotation_path = source / "annotations" / f"instances_{split}.json"
        raw = annotation_path.read_bytes()
        data = json.loads(raw)
        images = sorted(data["images"], key=lambda item: item["id"])
        chosen = sorted(random.Random(f"{seed}:{split}").sample(images, max(1, len(images) // denominator)), key=lambda item: item["id"])
        ids = {item["id"] for item in chosen}
        annotations = [item for item in data["annotations"] if item["image_id"] in ids]
        image_dir = output / "images" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        expected = {item["file_name"] for item in chosen}
        if any(p.name not in expected for p in image_dir.iterdir()):
            raise ValueError(f"Existing subset differs: {image_dir}; choose a new output")
        for item in chosen:
            original = source / "images" / split / item["file_name"]
            if not original.is_file():
                raise FileNotFoundError(original)
            target = image_dir / item["file_name"]
            if target.is_symlink():
                if target.resolve() != original.resolve():
                    raise ValueError(f"Unexpected image link: {target}")
            elif target.exists():
                raise ValueError(f"Expected image symlink: {target}")
            else:
                target.symlink_to(original)
        filtered = {**data, "images": chosen, "annotations": annotations}
        dest = output / "annotations" / annotation_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(filtered, separators=(",", ":")) + "\n"
        if dest.exists() and dest.read_text() != content:
            raise ValueError(f"Existing annotations differ: {dest}; choose a new output")
        dest.write_text(content)
        manifest["splits"][split] = {
            "source_images": len(images), "images": len(chosen), "annotations": len(annotations),
            "image_ids": sorted(ids), "source_annotations_sha256": hashlib.sha256(raw).hexdigest(),
            "subset_annotations_sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
    train_ids = set(manifest["splits"]["train2017"]["image_ids"])
    assert train_ids.isdisjoint(manifest["splits"]["val2017"]["image_ids"])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "image_ids"} for key, value in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/coco"))
    parser.add_argument("--output", type=Path, default=Path("data/coco-1over20"))
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--denominator", type=int, default=20)
    args = parser.parse_args()
    prepare(args.source, args.output, args.seed, args.denominator)
