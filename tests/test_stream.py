import sys
from types import SimpleNamespace

import numpy as np
import pytest

from detect import run


def stub_model(monkeypatch, frames):
    state = {"closed": False, "calls": 0}

    def stream():
        try:
            for frame in frames:
                yield SimpleNamespace(plot=lambda frame=frame: frame)
        finally:
            state["closed"] = True

    class Detector:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, **kwargs):
            state["calls"] += 1
            return stream()

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Detector))
    return state


def arguments(tmp_path):
    weights = tmp_path / "custom.pt"
    weights.write_bytes(b"test placeholder; not a model")
    source = tmp_path / "video.mp4"
    source.write_bytes(b"test placeholder; model adapter is substituted")
    return SimpleNamespace(
        weights=str(weights), source=str(source), output=None, conf=0.4, imgsz=640, output_fps=25, show=False
    )


def test_stream_closes_after_all_frames(tmp_path, monkeypatch, capsys):
    state = stub_model(monkeypatch, [np.zeros((48, 64, 3), np.uint8)] * 3)
    run(arguments(tmp_path))
    assert state["closed"]
    assert "Processed 3 frame(s)" in capsys.readouterr().out


def test_empty_video_is_an_error_not_a_success(tmp_path, monkeypatch):
    state = stub_model(monkeypatch, [])
    with pytest.raises(ValueError, match="decodable frames"):
        run(arguments(tmp_path))
    assert state["closed"]
