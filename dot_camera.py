import cv2
import numpy as np
from gpiozero import Button
from PIL import Image
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306


WIDTH, HEIGHT = 128, 64


def floyd_steinberg_dither(gray: np.ndarray) -> np.ndarray:
    img = gray.astype(np.float32)
    h, w = img.shape

    for y in range(h):
        for x in range(w):
            old = img[y, x]
            new = 255.0 if old > 127 else 0.0
            img[y, x] = new
            err = old - new

            if x + 1 < w:
                img[y, x + 1] += err * 7 / 16
            if y + 1 < h and x - 1 >= 0:
                img[y + 1, x - 1] += err * 3 / 16
            if y + 1 < h:
                img[y + 1, x] += err * 5 / 16
            if y + 1 < h and x + 1 < w:
                img[y + 1, x + 1] += err * 1 / 16

    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def main() -> None:
    serial = i2c(port=1, address=0x3C)
    oled = ssd1306(serial, width=WIDTH, height=HEIGHT)

    button = Button(17, pull_up=True, bounce_time=0.3)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera at /dev/video0")

    try:
        while True:
            button.wait_for_press()

            ok, frame = cap.read()
            if not ok:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            dithered = floyd_steinberg_dither(resized)

            binary = (dithered > 127).astype(np.uint8) * 255
            image = Image.fromarray(binary, mode="L").convert("1")
            oled.display(image)

            button.wait_for_release()

    finally:
        cap.release()
        oled.clear()


if __name__ == "__main__":
    main()
