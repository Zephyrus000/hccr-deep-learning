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
        options = {
            "image_size": 32,
            "rotation_degrees": 0,
            "translate_ratio": 0,
            "scale_min": 1.0,
            "scale_max": 1.0,
        }
        sharp = TrainPreprocessor(**options, blur_probability=0.0)(self.image)
        blurred = TrainPreprocessor(**options, blur_probability=1.0)(self.image)
        self.assertEqual(sharp.size, (32, 32))
        self.assertNotEqual(sharp.tobytes(), blurred.tobytes())

    def test_eval_preprocessing_is_deterministic(self) -> None:
        first = EvalPreprocessor(image_size=32)(self.image)
        second = EvalPreprocessor(image_size=32)(self.image)
        self.assertEqual(first.size, (32, 32))
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_eval_preprocessing_keeps_white_background(self) -> None:
        transformed = EvalPreprocessor(image_size=32)(self.image)
        self.assertEqual(transformed.getpixel((0, 0)), 255)
        self.assertLess(min(transformed.get_flattened_data()), 255)

    def test_train_transform_records_augmentation_contract(self) -> None:
        transform = TrainPreprocessor(
            image_size=32,
            rotation_degrees=0,
            translate_ratio=0,
            scale_min=1.0,
            scale_max=1.0,
            blur_probability=0.0,
        )
        self.assertEqual(transform(self.image).info["applied_augmentations"], ())

    def test_train_augmentation_is_not_fixed_rotation(self) -> None:
        transform = TrainPreprocessor(image_size=32, blur_probability=0.0)
        self.assertNotEqual(
            transform(self.image).tobytes(), transform(self.image).tobytes()
        )

    def test_train_scale_augmentation_preserves_size(self) -> None:
        transform = TrainPreprocessor(
            image_size=32,
            rotation_degrees=0,
            translate_ratio=0,
            scale_min=0.9,
            scale_max=0.9,
            blur_probability=0.0,
        )
        transformed = transform(self.image)
        self.assertEqual(transformed.size, (32, 32))
        self.assertEqual(transformed.info["applied_augmentations"], ())

    def test_gallery_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gallery.png"
            save_gallery(
                [self.image], EvalPreprocessor(image_size=16), output, ["U+4E00"]
            )
            self.assertTrue(output.is_file())
            with Image.open(output) as gallery:
                self.assertEqual(gallery.size, (64, 64))
