"""Unit tests for pure utility functions in utils.py."""

import math
import numpy as np
import pytest
import torch

# conftest.py stubs out OCC / wandb / diffusers before this import
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import (
    generate_random_string,
    get_bbox_norm,
    scale_bboxes,
    compute_bbox_centers_and_sizes,
    pad_repeat,
    pad_zero,
    pad_minusone,
    randn_tensor,
)


# ---------------------------------------------------------------------------
# generate_random_string
# ---------------------------------------------------------------------------


class TestGenerateRandomString:
    def test_correct_length(self):
        for length in [0, 1, 8, 32, 128]:
            s = generate_random_string(length)
            assert len(s) == length

    def test_alphanumeric_only(self):
        import string

        allowed = set(string.ascii_letters + string.digits)
        for _ in range(20):
            s = generate_random_string(16)
            assert set(s).issubset(allowed), f"Unexpected characters in: {s!r}"

    def test_randomness(self):
        """Two calls should almost never produce the same string."""
        results = {generate_random_string(16) for _ in range(10)}
        assert len(results) > 1


# ---------------------------------------------------------------------------
# get_bbox_norm
# ---------------------------------------------------------------------------


class TestGetBboxNorm:
    def test_unit_cube(self):
        # A unit cube has a diagonal of sqrt(3)
        pts = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 1],
            ],
            dtype=float,
        )
        norm = get_bbox_norm(pts)
        assert math.isclose(norm, math.sqrt(3), rel_tol=1e-6)

    def test_single_axis(self):
        pts = np.array([[0, 0, 0], [5, 0, 0]], dtype=float)
        assert math.isclose(get_bbox_norm(pts), 5.0)

    def test_negative_coordinates(self):
        pts = np.array([[-1, -1, -1], [1, 1, 1]], dtype=float)
        assert math.isclose(get_bbox_norm(pts), math.sqrt(12), rel_tol=1e-6)

    def test_single_point(self):
        pts = np.array([[3, 4, 5]], dtype=float)
        assert get_bbox_norm(pts) == 0.0


# ---------------------------------------------------------------------------
# scale_bboxes
# ---------------------------------------------------------------------------


class TestScaleBboxes:
    def test_scale_to_minus1_plus1(self):
        bboxes = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        scaled = scale_bboxes(bboxes, (-1, 1))
        assert torch.allclose(scaled[0], torch.tensor(-1.0))
        assert torch.allclose(scaled[-1], torch.tensor(1.0))

    def test_scale_to_0_1(self):
        bboxes = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
        scaled = scale_bboxes(bboxes, (0, 1))
        assert torch.allclose(scaled[0], torch.tensor(0.0))
        assert torch.allclose(scaled[-1], torch.tensor(1.0))

    def test_custom_min_max(self):
        bboxes = torch.tensor([3.0, 5.0, 7.0])
        scaled = scale_bboxes(bboxes, (0, 1), min_values=torch.tensor(3.0), max_values=torch.tensor(7.0))
        assert torch.allclose(scaled[1], torch.tensor(0.5))

    def test_output_shape_preserved(self):
        bboxes = torch.rand(4, 6)
        scaled = scale_bboxes(bboxes, (0, 1))
        assert scaled.shape == bboxes.shape


# ---------------------------------------------------------------------------
# compute_bbox_centers_and_sizes
# ---------------------------------------------------------------------------


class TestComputeBboxCentersAndSizes:
    def test_numpy_input_returns_numpy(self):
        min_c = np.array([[0.0, 0.0, 0.0]])
        max_c = np.array([[2.0, 4.0, 6.0]])
        centers, sizes = compute_bbox_centers_and_sizes(min_c, max_c)
        assert isinstance(centers, np.ndarray)
        assert isinstance(sizes, np.ndarray)
        np.testing.assert_allclose(centers[0], [1.0, 2.0, 3.0])
        assert sizes[0] == 6.0

    def test_torch_input_returns_tensor(self):
        min_c = torch.tensor([[0.0, 0.0, 0.0]])
        max_c = torch.tensor([[2.0, 2.0, 2.0]])
        centers, sizes = compute_bbox_centers_and_sizes(min_c, max_c)
        assert isinstance(centers, torch.Tensor)
        assert torch.allclose(centers[0], torch.tensor([1.0, 1.0, 1.0]))
        assert sizes[0] == 2.0

    def test_1d_input_broadcast(self):
        min_c = np.array([0.0, 0.0, 0.0])
        max_c = np.array([4.0, 4.0, 4.0])
        centers, sizes = compute_bbox_centers_and_sizes(min_c, max_c)
        np.testing.assert_allclose(centers[0], [2.0, 2.0, 2.0])

    def test_swapped_corners_handled(self):
        """Function should handle min > max gracefully."""
        min_c = np.array([[5.0, 5.0, 5.0]])
        max_c = np.array([[0.0, 0.0, 0.0]])
        centers, sizes = compute_bbox_centers_and_sizes(min_c, max_c)
        # After torch.minimum / torch.maximum, should give same result as correct order
        np.testing.assert_allclose(centers[0], [2.5, 2.5, 2.5])
        assert sizes[0] == 5.0


# ---------------------------------------------------------------------------
# pad_repeat
# ---------------------------------------------------------------------------


class TestPadRepeat:
    def test_output_length(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        result = pad_repeat(x, 7)
        assert len(result) == 7

    def test_exact_multiple(self):
        x = np.array([[1], [2]])
        result = pad_repeat(x, 6)
        assert len(result) == 6

    def test_values_are_repeats(self):
        x = np.array([[1, 0], [2, 0]])
        result = pad_repeat(x, 5)
        # All values should come from x
        for row in result:
            assert row in x.tolist()


# ---------------------------------------------------------------------------
# pad_zero / pad_minusone
# ---------------------------------------------------------------------------


class TestPadZero:
    def test_output_length(self):
        x = np.ones((3, 4))
        result = pad_zero(x, 6)
        assert result.shape == (6, 4)

    def test_padding_is_zero(self):
        x = np.ones((3, 4))
        result = pad_zero(x, 6)
        np.testing.assert_array_equal(result[3:], 0)

    def test_mask_returned(self):
        x = np.ones((3, 4))
        result, mask = pad_zero(x, 6, return_mask=True)
        assert mask.shape == (6,)
        assert mask[2] == False  # noqa: E712 (True means padding)
        assert mask[3] == True  # noqa: E712


class TestPadMinusone:
    def test_output_length(self):
        x = np.ones((2, 5))
        result = pad_minusone(x, 5)
        assert result.shape == (5, 5)

    def test_padding_is_minus_one(self):
        x = np.zeros((2, 3))
        result = pad_minusone(x, 4)
        np.testing.assert_array_equal(result[2:], -1)

    def test_original_values_preserved(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = pad_minusone(x, 5)
        np.testing.assert_array_equal(result[:2], x)


# ---------------------------------------------------------------------------
# randn_tensor
# ---------------------------------------------------------------------------


class TestRandnTensor:
    def test_output_shape(self):
        t = randn_tensor((4, 8, 16))
        assert t.shape == (4, 8, 16)

    def test_device_cpu(self):
        t = randn_tensor((2, 4), device=torch.device("cpu"))
        assert t.device.type == "cpu"

    def test_dtype(self):
        t = randn_tensor((3, 3), dtype=torch.float32)
        assert t.dtype == torch.float32

    def test_seeded_generator_reproducible(self):
        gen = torch.Generator()
        gen.manual_seed(42)
        t1 = randn_tensor((5, 5), generator=gen)
        gen.manual_seed(42)
        t2 = randn_tensor((5, 5), generator=gen)
        assert torch.allclose(t1, t2)

    def test_batched_generator(self):
        generators = [torch.Generator() for _ in range(4)]
        for i, g in enumerate(generators):
            g.manual_seed(i)
        t = randn_tensor((4, 8), generator=generators)
        assert t.shape == (4, 8)
