from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from hccr.preprocessing import EvalPreprocessor, TrainPreprocessor
from hccr.preprocessing.gallery import save_gallery


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("L", (20, 10), 255)
        ImageDraw.Draw(self.image).rectangle((7, 2, 12, 8), fill=0)

    def test_eval_output_has_target_size(self) -> None:
        self.assertEqual(EvalPreprocessor(image_size=32)(self.image).size, (32, 32))

    def test_train_blur_is_optional(self) -> None:
        self.assertEqual(
            TrainPreprocessor(image_size=32, blur_radius=1)(self.image).size, (32, 32)
        )

    def test_optional_normalization_steps_preserve_target_size(self) -> None:
        transform = EvalPreprocessor(
            image_size=32,
            center_by_centroid=True,
            otsu_binarize=True,
            median_filter_size=3,
        )
        self.assertEqual(transform(self.image).size, (32, 32))

    def test_train_augmentation_is_not_fixed_rotation(self) -> None:
        transform = TrainPreprocessor(image_size=32, blur_radius=None)
        self.assertNotEqual(
            transform(self.image).tobytes(), transform(self.image).tobytes()
        )

    def test_gallery_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gallery.png"
            save_gallery([self.image], EvalPreprocessor(image_size=16), output)
            self.assertTrue(output.is_file())
