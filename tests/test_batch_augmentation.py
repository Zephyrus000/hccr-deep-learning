from __future__ import annotations

import unittest

import torch

from hccr.preprocessing.batch import TensorBatchAugmenter


class BatchAugmentationTests(unittest.TestCase):
    def glyphs(self, count=4):
        images = torch.ones(count, 1, 32, 32)
        images[:, :, 4:28, 12:20] = 0
        images[:, :, 12:20, 4:28] = 0
        return images

    def test_seed_and_epoch_replay_without_global_rng_changes(self):
        images = self.glyphs()
        first = TensorBatchAugmenter(32, seed=17)
        second = TensorBatchAugmenter(32, seed=17)
        state = torch.get_rng_state().clone()
        first.set_epoch(3)
        expected, tags = first(images)
        second.set_epoch(3)
        actual, other_tags = second(images)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(tags, other_tags)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        first.set_epoch(3)
        replay, replay_tags = first(images)
        torch.testing.assert_close(replay, expected, rtol=0, atol=0)
        self.assertEqual(replay_tags, tags)
        first.set_epoch(4)
        changed, _ = first(images)
        self.assertFalse(torch.equal(changed, expected))

    def test_blank_images_remain_exactly_white_with_blur(self):
        transformed, tags = TensorBatchAugmenter(32, blur_probability=1)(
            torch.ones(3, 1, 32, 32)
        )
        self.assertTrue(torch.equal(transformed, torch.ones_like(transformed)))
        self.assertTrue(all("gaussian_blur" in tag for tag in tags))

    def test_guard_preserves_ink_inside_frame_for_large_warps(self):
        images = torch.ones(24, 1, 32, 32)
        images[:, :, 1:31, 1:31] = 0
        transformed, tags = TensorBatchAugmenter(
            32,
            rotation_degrees=75,
            translate_ratio=0.15,
            scale_min=1.1,
            scale_max=1.6,
        )(images)
        self.assertTrue((transformed.amin(dim=(1, 2, 3)) < 0.1).all())
        for border in (
            transformed[:, :, :2],
            transformed[:, :, -2:],
            transformed[:, :, :, :2],
            transformed[:, :, :, -2:],
        ):
            self.assertTrue(torch.equal(border, torch.ones_like(border)))
        self.assertTrue(all(tag == "scale,translation,rotation" for tag in tags))

    def test_zero_augmentation_is_exact_identity_without_gradient(self):
        images = self.glyphs().requires_grad_(True)
        augment = TensorBatchAugmenter(
            32,
            rotation_degrees=0,
            translate_ratio=0,
            scale_min=1,
            scale_max=1,
        )
        transformed, tags = augment(images)
        self.assertTrue(torch.equal(transformed, images))
        self.assertFalse(transformed.requires_grad)
        self.assertEqual(tags, [""] * 4)
        self.assertEqual(transformed.shape, images.shape)
        self.assertEqual(transformed.dtype, torch.float32)
        self.assertEqual(transformed.device, images.device)

    def test_optional_blur_changes_only_selected_samples_and_white_padding(self):
        images = self.glyphs(20)
        augment = TensorBatchAugmenter(
            32,
            rotation_degrees=0,
            translate_ratio=0,
            scale_min=1,
            scale_max=1,
            blur_probability=0.5,
        )
        transformed, tags = augment(images)
        self.assertIn("", tags)
        self.assertIn("gaussian_blur", tags)
        for index, tag in enumerate(tags):
            if tag:
                self.assertFalse(torch.equal(transformed[index], images[index]))
            else:
                self.assertTrue(torch.equal(transformed[index], images[index]))
        self.assertTrue(
            torch.equal(transformed[:, :, 0], torch.ones_like(transformed[:, :, 0]))
        )

    def test_metadata_names_the_changed_geometry_and_parameter_policy(self):
        metadata = TensorBatchAugmenter(32).metadata()
        self.assertEqual(metadata["backend"], "tensor_affine_v1")
        self.assertIn("guard=2px", metadata["geometry_policy"])
        self.assertEqual(metadata["blur_sigma"], 0.5)
        self.assertEqual(
            metadata["tag_policy"], "sampled_operations_before_geometry_guard"
        )

    def test_augmentation_ignores_surrounding_autocast_precision(self):
        images = self.glyphs()
        expected, tags = TensorBatchAugmenter(32, blur_probability=1)(images)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual, actual_tags = TensorBatchAugmenter(32, blur_probability=1)(images)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual_tags, tags)
        self.assertEqual(actual.dtype, torch.float32)

    def test_invalid_configuration_and_input_fail_at_boundary(self):
        for options in (
            {"scale_min": 0},
            {"translate_ratio": 2},
            {"blur_probability": -1},
            {"rotation_degrees": float("nan")},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TensorBatchAugmenter(32, **options)
        augment = TensorBatchAugmenter(32)
        with self.assertRaises(ValueError):
            augment(self.glyphs().half())
        with self.assertRaises(ValueError):
            augment(torch.ones(2, 3, 32, 32))
        with self.assertRaises(ValueError):
            augment.set_epoch(-1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_cuda_share_parameters_and_close_float32_results(self):
        images = self.glyphs()
        cpu = TensorBatchAugmenter(32, seed=29, blur_probability=0.5)
        cuda = TensorBatchAugmenter(32, seed=29, blur_probability=0.5)
        expected, tags = cpu(images)
        actual, cuda_tags = cuda(images.cuda())
        self.assertEqual(tags, cuda_tags)
        self.assertEqual(actual.device.type, "cuda")
        torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
