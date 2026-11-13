"""
core/preprocessor.py — Frame Enhancement Pipeline
===================================================
Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
on the luminance channel to improve detection accuracy in
poorly-lit construction site footage, then resizes to model
input dimensions.

Why CLAHE on L-channel only?
  Applying histogram equalization to RGB channels independently
  introduces color distortion. LAB separates luminance from
  chrominance, so we boost contrast without shifting colors.
"""

import cv2
import numpy as np


class FramePreprocessor:
    """Enhances raw camera frames for optimal detector performance."""

    def __init__(self, config: dict):
        preproc = config["preprocessing"]
        # CLAHE parameters — clip_limit controls contrast amplification ceiling,
        # tile_grid sets the number of contextual histogram regions
        self.clip_limit = preproc["clahe_clip_limit"]
        self.tile_grid = tuple(preproc["clahe_tile_grid"])
        self.target_size = tuple(preproc["target_size"])

        # Create the CLAHE operator once and reuse per frame
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid,
        )

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Full enhancement pipeline: CLAHE on luminance + resize.

        Steps:
          1. BGR → LAB  (isolate luminance from color)
          2. CLAHE on L channel  (boost contrast without color shift)
          3. LAB → BGR  (reconstruct the enhanced color frame)
          4. Resize to target_size  (match model input dimensions)
        """
        # Convert to LAB color space — L is lightness, A/B are chrominance
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE only to the lightness channel
        l_enhanced = self.clahe.apply(l_channel)

        # Reconstruct the LAB image and convert back to BGR
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        # Resize to model input size (width, height for cv2.resize)
        enhanced = cv2.resize(enhanced, self.target_size)

        return enhanced

    def normalize(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1] float32.
        Useful for models that expect normalized input rather than uint8.
        """
        return frame.astype(np.float32) / 255.0
