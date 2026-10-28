"""Run a local YOLO detector against an image, video, or an explicit camera index."""

import argparse
from pathlib import Path


def confidence(value):
    result = float(value)
    if not 0 < result <= 1:
        raise argparse.ArgumentTypeError("confidence must be in (0, 1]")
    return result


def parse_source(value):
    if value.isdecimal():
        return int(value)
    path = Path(value)
    if not path.is_file():
        raise ValueError(f"Input file does not exist: {path}")
    return str(path)


def run(args):
    import cv2
    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.is_file():
        raise ValueError("Supply an existing local weights file with --weights")
    source = parse_source(args.source)
    model = YOLO(str(weights), task="detect")
    # Ultralytics' stream generator avoids retaining all video frames in memory.
    results = model.predict(
        source=source, conf=args.conf, imgsz=args.imgsz, stream=True, verbose=False, save=False
    )
    output = Path(args.output) if args.output else None
    writer = None
    if output:
        if output.exists():
            raise ValueError(f"Refusing to overwrite: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    image_input = isinstance(source, str) and Path(source).suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }
    if output and image_input and output.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Image output must use .jpg or .png")
    if output and not image_input and output.suffix.lower() != ".mp4":
        raise ValueError("Video output must use .mp4")
    try:
        count = 0
        for result in results:
            frame = result.plot()
            count += 1
            if output and image_input:
                if not cv2.imwrite(str(output), frame):
                    raise OSError("Could not write output image")
            elif output:
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.output_fps, (width, height)
                    )
                    if not writer.isOpened():
                        raise OSError("Could not initialize MP4 writer")
                writer.write(frame)
            if args.show:
                cv2.imshow("Local detection", frame)
                if cv2.waitKey(0 if image_input else 1) & 0xFF == ord("q"):
                    break
        if not count:
            raise ValueError("Input did not yield any decodable frames")
        print(f"Processed {count} frame(s)")
    finally:
        results.close()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Local image/video path, or camera index")
    parser.add_argument("--weights", required=True, help="Trusted local .pt model; never committed")
    parser.add_argument("--conf", type=confidence, default=0.4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", help="Optional annotated image or MP4")
    parser.add_argument(
        "--output-fps", type=float, default=25.0, help="Explicit video output rate; frames are not resampled"
    )
    parser.add_argument("--show", action="store_true", help="Open a desktop preview; q exits")
    args = parser.parse_args()
    if args.imgsz < 32 or args.output_fps <= 0:
        parser.error("imgsz must be >=32 and output-fps must be positive")
    try:
        run(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
