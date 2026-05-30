#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal smoke tests for grid_splitter_v4 core functions."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on path for import
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grid_splitter_v4 import (  # noqa: E402
    _imread_unicode,
    auto_detect_grid_dimensions,
    detect_outer_borders,
    split_grid_clean,
)

TEST_IMAGES = ROOT / "test_images"


def _open(name: str):
    """Read a test image, skip test if missing."""
    path = TEST_IMAGES / name
    if not path.exists():
        pytest.skip(f"test image not found: {name}")
    img = _imread_unicode(str(path))
    if img is None:
        pytest.skip(f"cannot decode: {name}")
    return img


class TestAutoDetect:
    """Verify auto_detect_grid_dimensions on known images."""

    def test_5x5_from_01(self):
        """01.png is a 5×5 grid."""
        img = _open("01.png")
        result = auto_detect_grid_dimensions(img)
        assert result is not None, "should detect a grid"
        rows, cols = result
        assert rows == 5, f"expected 5 rows, got {rows}"
        assert cols == 5, f"expected 5 cols, got {cols}"

    def test_3x3_from_jiugongge(self):
        """九宫格.png is a 3×3 grid."""
        img = _open("九宫格.png")
        result = auto_detect_grid_dimensions(img)
        assert result is not None, "should detect a grid"
        rows, cols = result
        assert rows == 3, f"expected 3 rows, got {rows}"
        assert cols == 3, f"expected 3 cols, got {cols}"


class TestOuterBorders:
    """Smoke test for border detection."""

    def test_returns_tuple_of_four(self):
        """detect_outer_borders always returns (top, bot, left, right)."""
        img = _open("01.png")
        borders = detect_outer_borders(img)
        assert len(borders) == 4
        assert all(isinstance(b, int) for b in borders)


class TestSplitGrid:
    """End-to-end split on a known image."""

    def test_split_5x5_produces_25_cells(self):
        """01.png as 5×5 → 25 clips, all same size."""
        img_path = str(TEST_IMAGES / "01.png")
        if not os.path.exists(img_path):
            pytest.skip("test image missing")
        with tempfile.TemporaryDirectory() as tmp:
            n = split_grid_clean(
                image_path=img_path,
                output_folder=tmp,
                rows=5,
                cols=5,
                auto_detect=True,
            )
            assert n == 25, f"expected 25 clips, got {n}"
            clips = sorted(Path(tmp).glob("clip_*.jpg"))
            assert len(clips) == 25

            # All clips should be identical size
            import cv2
            import numpy as np

            sizes = set()
            for c in clips:
                data = np.fromfile(str(c), dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                sizes.add(img.shape[:2])
            assert len(sizes) == 1, f"all clips must be same size, got {sizes}"
