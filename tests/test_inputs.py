import argparse

import pytest

from detect import confidence, parse_source


@pytest.mark.parametrize("value", ["0", "1.2", "-0.1", "nan"])
def test_rejects_invalid_confidence(value):
    with pytest.raises(argparse.ArgumentTypeError):
        confidence(value)


def test_camera_index_is_explicit():
    assert parse_source("0") == 0


def test_missing_media_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        parse_source(str(tmp_path / "missing.mp4"))


def test_existing_local_media(tmp_path):
    image = tmp_path / "input.png"
    image.write_bytes(b"placeholder")
    assert parse_source(str(image)) == str(image)
