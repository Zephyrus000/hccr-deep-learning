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

    def test_white_on_black_is_exact_inverse_of_default_polarity(self) -> None:
        black_on_white = EvalPreprocessor(image_size=32)(self.image)
        white_on_black = EvalPreprocessor(
            image_size=32, input_polarity="white_on_black"
        )(self.image)
        self.assertEqual(
            white_on_black.tobytes(),
            bytes(255 - value for value in black_on_white.get_flattened_data()),
        )

    def test_unknown_input_polarity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "input_polarity"):
            EvalPreprocessor(input_polarity="unknown")(self.image)

    def test_train_augmentation_is_not_fixed_rotation(self) -> None:
        transform = TrainPreprocessor(image_size=32, blur_radius=None)
        self.assertNotEqual(
            transform(self.image).tobytes(), transform(self.image).tobytes()
        )

    def test_elastic_transform_records_application_and_preserves_size(self) -> None:
        transform = TrainPreprocessor(
            image_size=32,
            rotation_degrees=0,
            translate_pixels=0,
            scale_range=(1.0, 1.0),
            elastic_probability=1.0,
        )
        transformed = transform(self.image)
        self.assertEqual(transformed.size, (32, 32))
        self.assertEqual(transformed.info["applied_augmentations"], ("elastic",))

    def test_morphology_is_mutually_exclusive_with_black_foreground(self) -> None:
        common = {
            "image_size": 32,
            "rotation_degrees": 0,
            "translate_pixels": 0,
            "scale_range": (1.0, 1.0),
        }
        eroded = TrainPreprocessor(**common, erosion_probability=1.0)(self.image)
        dilated = TrainPreprocessor(**common, dilation_probability=1.0)(self.image)
        eroded_foreground = sum(value < 128 for value in eroded.get_flattened_data())
        dilated_foreground = sum(value < 128 for value in dilated.get_flattened_data())
        self.assertEqual(eroded.info["applied_augmentations"], ("erosion",))
        self.assertEqual(dilated.info["applied_augmentations"], ("dilation",))
        self.assertLess(eroded_foreground, dilated_foreground)

    def test_morphology_keeps_semantics_after_output_polarity_change(self) -> None:
        common = {
            "image_size": 32,
            "input_polarity": "white_on_black",
            "rotation_degrees": 0,
            "translate_pixels": 0,
            "scale_range": (1.0, 1.0),
        }
        eroded = TrainPreprocessor(**common, erosion_probability=1.0)(self.image)
        dilated = TrainPreprocessor(**common, dilation_probability=1.0)(self.image)
        eroded_foreground = sum(value > 128 for value in eroded.get_flattened_data())
        dilated_foreground = sum(value > 128 for value in dilated.get_flattened_data())
        self.assertLess(eroded_foreground, dilated_foreground)

    def test_invalid_augmentation_probabilities_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to at most 1"):
            TrainPreprocessor(erosion_probability=0.6, dilation_probability=0.6)(
                self.image
            )

    def test_gallery_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gallery.png"
            save_gallery(
                [self.image], EvalPreprocessor(image_size=16), output, ["U+4E00"]
            )
            self.assertTrue(output.is_file())
            with Image.open(output) as gallery:
                self.assertEqual(gallery.size, (64, 64))
