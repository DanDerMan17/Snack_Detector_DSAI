import argparse
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--image", type=Path, help="Test an existing image without opening the camera")
    parser.add_argument("--output", type=Path, help="Annotated output for --image")
    args = parser.parse_args()

    if not 0 <= args.conf <= 1:
        parser.error("--conf has to be between 0 and 1")
    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")
    if args.image and args.preview:
        parser.error("--image cannot be used with --preview")
    if args.output and not args.image:
        parser.error("--output requires --image")

    model = None

    if not args.preview:
        weights = args.weights

        if not weights.is_file():
            raise SystemExit(f"Snack model is missing: {weights}")

        from ultralytics import YOLO

        model = YOLO(str(weights))
        print(f"Model: {weights.resolve()} | confidence: {args.conf:.0%}")
        print("Classes: " + ", ".join(model.names.values()))

    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise SystemExit(f"Cannot read image: {args.image}")
        output = args.output or ROOT / "runs/predictions" / f"{args.image.stem}_detected.jpg"
        if output.resolve() == args.image.resolve():
            parser.error("--output must differ from the input image")
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), result.plot()):
            raise SystemExit(f"Cannot save output: {output}")
        print(f"Detected {len(result.boxes)} snacks. Saved: {output}")
        return

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    window = "Snack Detection | S: save | Q: exit"

    try:
        if not camera.isOpened():
            raise SystemExit("camera not found!")

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while True:
            success, frame = camera.read()

            if not success:
                raise SystemExit("no camera capture!")

            if model is None:
                shown = frame.copy()
            else:
                result = model.predict(
                    frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    verbose=False,
                )[0]

                shown = result.plot()
                status = f"Snacks: {len(result.boxes)} | confidence >= {args.conf:.0%}"
                cv2.putText(shown, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            .65, (0, 255, 255), 2)

            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

            if key == ord("s"):
                folder = ROOT / "data/raw"
                folder.mkdir(parents=True, exist_ok=True)

                name = datetime.now().strftime("snack_%Y%m%d_%H%M%S_%f.jpg")

                if cv2.imwrite(str(folder / name), frame):
                    print(f"Picture saved: {name}")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
