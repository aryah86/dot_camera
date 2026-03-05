import cv2
import numpy as np
import pygame
import sys


WIDTH, HEIGHT = 128, 64
SCALE = 4
WIN_WIDTH, WIN_HEIGHT = WIDTH * SCALE, HEIGHT * SCALE


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


def render_to_surface(dithered: np.ndarray, surface: pygame.Surface) -> None:
    surface.fill((0, 0, 0))
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if dithered[y, x] > 127:
                pygame.draw.rect(
                    surface,
                    (255, 255, 255),
                    (x * SCALE, y * SCALE, SCALE, SCALE),
                )


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Dot Camera Simulator")
    screen.fill((0, 0, 0))
    pygame.display.flip()

    print("Dot Camera Simulator")
    print("Press SPACE to capture, ESC to quit.")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    ok, frame = cap.read()
                    if not ok:
                        print("Warning: Failed to capture frame.")
                        continue

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(
                        gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA
                    )
                    dithered = floyd_steinberg_dither(resized)
                    render_to_surface(dithered, screen)
                    pygame.display.flip()

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
