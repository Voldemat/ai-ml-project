import sys

import cv2

from src import inference


def main() -> None:
    image = cv2.imread(sys.argv[1])
    assert image is not None
    svc = inference.load_model()
    has_pedestrian = inference.run(svc, image)
    print(f"Has pedestrian: {has_pedestrian}")


if __name__ == "__main__":
    main()
