import argparse
from datetime import datetime
from pathlib import Path
from re import VERBOSE

import cv2

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()

    if not 0 <= args.conf <= 1:
        parser.error("--conf has to be between 0 and 1")

    model = None

    if not args.preview:
        weights = ROOT / "models/best.pt"

        if not weights.is_file():
            raise SystemExit("Snack model is missing")

        from ultralytics import YOLO

        model = YOLO(str(weights))

    camera = cv2.VideoCapture(args.camera)
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
                    imgsz=640,
                    verbose=False,
                )[0]

                shown = result.plot()

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
