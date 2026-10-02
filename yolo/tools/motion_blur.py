"""Mild, centered motion blur shared by the training augmentation pipelines."""

import random

import cv2
import numpy as np
from PIL import Image


class MotionBlur:
    """Blend a short directional blur with the original, preserving all labels.

    Defaults apply a 3-pixel horizontal, vertical or diagonal blur to 10% of
    images, retaining 50% of the original image. Uses the seeded Python RNG.
    """

    def __init__(self, prob=0.1, kernel_size=3, strength=0.5):
        self.prob = float(prob)
        self.strength = float(strength)
        if not 0 <= self.prob <= 1:
            raise ValueError("motion blur probability must be between 0 and 1")
        if not 0 <= self.strength <= 1:
            raise ValueError("motion blur strength must be between 0 and 1")
        if isinstance(kernel_size, bool) or not isinstance(kernel_size, int) or kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("motion blur kernel_size must be an odd integer >= 3")
        self.kernel_size = kernel_size

    def __call__(self, image, boxes):
        # A disabled transform must not disturb the upstream recipe's RNG.
        if self.prob == 0 or self.strength == 0 or random.random() >= self.prob:
            return image, boxes

        size = self.kernel_size
        kernel = np.zeros((size, size), dtype=np.float32)
        direction = random.randrange(4)
        if direction == 0:
            kernel[size // 2, :] = 1
        elif direction == 1:
            kernel[:, size // 2] = 1
        elif direction == 2:
            np.fill_diagonal(kernel, 1)
        else:
            np.fill_diagonal(kernel[:, ::-1], 1)
        kernel /= size

        pixels = np.asarray(image)
        blurred = cv2.filter2D(pixels, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        output = cv2.addWeighted(pixels, 1 - self.strength, blurred, self.strength, 0)
        if isinstance(image, Image.Image):
            output = Image.fromarray(output)
        return output, boxes
