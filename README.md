# Dot Camera

Raspberry Pi camera system that captures a photo and renders it as a dot portrait on a 0.96" SSD1306 OLED screen (`128x64`).

The device is intentionally blind: no live preview, no framing screen. Press the button, capture, process, display dots.

## Product Definition

- Capture from a USB webcam.
- Convert image to grayscale.
- Resize to `128x64`.
- Apply Floyd-Steinberg dithering to produce binary black/white output.
- Display as white dots on black background on SSD1306 OLED.
- Hold image until next button press.

## Hardware

- Raspberry Pi (any model with GPIO)
- USB webcam
- SSD1306 OLED `128x64` (I2C)
- Push button

### Wiring

- OLED `VCC` -> Pi `3.3V`
- OLED `GND` -> Pi `GND`
- OLED `SCL` -> `GPIO3` (I2C SCL)
- OLED `SDA` -> `GPIO2` (I2C SDA)
- Button one leg -> `GPIO17`
- Button other leg -> `GND`

## Software Dependencies

- Python 3
- `opencv-python` (camera capture)
- `Pillow` (image processing)
- `luma.oled` + `luma.core` (OLED driver)
- `gpiozero` (button input)

Install with:

```bash
python3 -m pip install opencv-python Pillow luma.oled luma.core gpiozero
```

## Runtime Flow

1. Initialize camera once with `cv2.VideoCapture(0)` and keep it open.
2. Initialize OLED as SSD1306 on I2C port `1`, address `0x3C`.
3. Wait for button press on GPIO 17.
4. Capture one frame from webcam.
5. Convert BGR to grayscale using `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)`.
6. Resize to exactly `128x64`.
7. Apply Floyd-Steinberg dithering (grayscale -> binary).
8. Convert to PIL mode `'1'` and display on OLED.
9. Keep image shown until the next button press.
10. Repeat.

## Floyd-Steinberg Dithering

Process each pixel left-to-right, top-to-bottom:

- Threshold: if value `> 127`, set to `255` (white), else `0` (black)
- Error: `error = old_value - new_value`
- Distribute to neighbors:
  - right: `7/16 * error`
  - bottom-left: `3/16 * error`
  - bottom: `5/16 * error`
  - bottom-right: `1/16 * error`

## OLED Configuration

- Driver: `luma.oled.device.ssd1306`
- I2C port: `1`
- I2C address: `0x3C`
- Image requirements: PIL `Image` in mode `'1'`, size `(128, 64)`

## Button Configuration

Use `gpiozero.Button`:

```python
Button(17, pull_up=True, bounce_time=0.3)
```

## Reference Implementation (Single Script)

Save as `dot_camera.py`:

```python
import cv2
import numpy as np
from gpiozero import Button
from PIL import Image
from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306


WIDTH, HEIGHT = 128, 64


def floyd_steinberg_dither(gray: np.ndarray) -> np.ndarray:
	# Work in float so propagated error keeps precision.
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
```

## Run

```bash
python3 dot_camera.py
```

## Notes

- Enable I2C on Raspberry Pi (`sudo raspi-config` -> Interface Options -> I2C).
- Confirm OLED is visible with `i2cdetect -y 1` (should show `0x3c`).
- Keep the camera object open between captures for faster response.
