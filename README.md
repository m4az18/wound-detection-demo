# Wound detection demo

An experimental image and video inference workflow for a custom wound detector.

The inference entry point accepts a local weights file, validates inputs, streams
results, and optionally saves annotated output. Model weights and original media
are deliberately excluded from the repository.

## Run locally

Python 3.10 or newer. Install dependencies in a virtual environment:

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -r requirements.txt
python detect.py --source data/example.mp4 --weights models/custom.pt --show
python detect.py --source data/example.jpg --weights models/custom.pt --output outputs/example.png
```

Create `data/` and `models/` locally. Supply media you are authorized to process
and weights whose license permits your use. Only load trusted model files.
A camera is opened only when its index is explicitly supplied, e.g. `--source 0`.
For video export, set `--output outputs/result.mp4 --output-fps 25`; the output
rate is explicit and does not automatically follow source timing. Exports contain
annotated video frames only, with no audio track.

## Engineering choices

- Stream predictions to keep memory usage independent of video length.
- Keep model weights outside Git and make their location explicit.
- Use `--show` for desktop previews; headless processing is the default.
- Refuse accidental output overwrite and release the video writer on exit.

## Evaluation and limitations

This is a research demonstration. It has no clinical validation and must not be used for diagnosis, triage, or treatment decisions. The original image examples are excluded because consent and redistribution rights are unverified.

Before reporting a metric, record dataset provenance, class definitions, split
method, model version, per-class precision/recall, mAP and inference hardware.
See [the evaluation template](docs/evaluation.md).

## Checks

```bash
python -m pip install -r requirements-dev.txt
ruff check .
pytest -q
```

Tests cover argument validation and source handling. They do not establish model
accuracy. Full inference requires separately supplied weights.
