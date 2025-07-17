#!/usr/bin/env python3
"""
Frequencsphere – Raspberry Pi edition
• Runs full‑screen on any attached frame‑buffer (HDMI or SPI LCD)
• Auto‑detects resolution so you do NOT need to edit WIDTH/HEIGHT
• Clean exit by tapping the screen (touch) or pressing Esc / Ctrl‑C
"""

import os
import sys
import time
import math
import numpy as np
import pygame
import sounddevice as sd

# ---------- Audio configuration ----------
SAMPLE_RATE   = 44_100
BLOCK_SIZE    = 2048
DEVICE        = None          # Use default USB mic – change index if you wish
CHANNELS      = 1             # Mini mic is mono

# ---------- Visual / run‑time configuration ----------
SMOOTHING_FACTOR     = 0.30
BASE_RADIUS, MAX_RADIUS = 60, 150         # Scaled for a 480×320 screen
MIN_FREQUENCY, MAX_FREQUENCY = 80, 1_000  # Hz
ROTATION_BASE_SPEED, ROTATION_MAX_SPEED = 0.005, 0.02
ROTATION_SMOOTHING   = 0.10

# ---------- Globals ----------
audio_data                 = np.zeros(BLOCK_SIZE)
smoothed_magnitudes        = np.zeros(BLOCK_SIZE // 2)
rotation_angle             = 0
rotation_speed             = ROTATION_BASE_SPEED
smoothed_rotation_speed    = ROTATION_BASE_SPEED


# ---------- Audio callback ----------
def audio_callback(indata, frames, time_info, status):
    global audio_data
    if status:
        print("Audio status:", status, file=sys.stderr)
    if indata.size:
        audio_data = indata[:, 0].copy()


# ---------- Pygame helpers ----------
def init_pygame():
    # Tell SDL to use the SPI framebuffer if it exists
    if not os.getenv("DISPLAY"):
        os.environ["SDL_FBDEV"]       = os.getenv("SDL_FBDEV", "/dev/fb1")
        os.environ["SDL_VIDEODRIVER"] = "fbcon"

    pygame.init()
    # Query the *real* screen size
    info              = pygame.display.Info()
    width, height     = info.current_w, info.current_h
    screen            = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)
    clock             = pygame.time.Clock()
    return screen, clock, width, height


def freq_to_color(freq, mag):
    hue        = 360 * (freq - MIN_FREQUENCY) / (MAX_FREQUENCY - MIN_FREQUENCY)
    saturation = max(0, min(mag * 100, 100))
    color      = pygame.Color(0, 0, 0)
    color.hsva = (hue % 360, saturation, 100, 100)
    return color


def visualise(screen, width, height, data):
    global smoothed_magnitudes, rotation_angle, rotation_speed, smoothed_rotation_speed

    # FFT
    fft_data       = np.fft.fft(data)
    magnitudes     = np.abs(fft_data[:len(data)//2])
    frequencies    = np.fft.fftfreq(len(data), 1 / SAMPLE_RATE)[:len(data)//2]

    # Normalise & smooth
    max_mag        = magnitudes.max() or 1
    norm_mag       = magnitudes / max_mag
    smoothed_magnitudes = (SMOOTHING_FACTOR * norm_mag +
                           (1 - SMOOTHING_FACTOR) * smoothed_magnitudes)

    # Bass‑driven rotation speed
    bass_idx       = np.where((frequencies >= 20) & (frequencies <= 200))[0]
    bass_mag       = smoothed_magnitudes[bass_idx].mean() if bass_idx.size else 0
    rotation_speed = ROTATION_BASE_SPEED + bass_mag * (ROTATION_MAX_SPEED - ROTATION_BASE_SPEED)
    smoothed_rotation_speed = (ROTATION_SMOOTHING * rotation_speed +
                               (1 - ROTATION_SMOOTHING) * smoothed_rotation_speed)
    rotation_angle += smoothed_rotation_speed

    # Fade
    fade = pygame.Surface((width, height)); fade.set_alpha(30); fade.fill((0, 0, 0))
    screen.blit(fade, (0, 0))

    # Draw
    centre = (width // 2, height // 2)
    pts, cols = [], []
    for mag, freq in zip(smoothed_magnitudes, frequencies):
        if MIN_FREQUENCY <= freq <= MAX_FREQUENCY:
            angle  = 2 * math.pi * (freq - MIN_FREQUENCY) / (MAX_FREQUENCY - MIN_FREQUENCY) + rotation_angle
            rad    = BASE_RADIUS + int(mag * (MAX_RADIUS - BASE_RADIUS))
            x      = int(centre[0] + rad * math.cos(angle))
            y      = int(centre[1] + rad * math.sin(angle))
            pts.append((x, y)); cols.append(freq_to_color(freq, mag))

    for i in range(len(pts) - 1):
        pygame.draw.line(screen, cols[i], pts[i], pts[i+1], 2)
    if pts:
        pygame.draw.line(screen, cols[-1], pts[-1], pts[0], 2)


def main():
    screen, clock, width, height = init_pygame()

    # Start audio
    with sd.InputStream(callback=audio_callback,
                        channels=CHANNELS,
                        samplerate=SAMPLE_RATE,
                        blocksize=BLOCK_SIZE,
                        device=DEVICE):
        while True:
            for ev in pygame.event.get():
                if (ev.type == pygame.QUIT or
                    (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q)) or
                    (ev.type == pygame.MOUSEBUTTONDOWN)):
                    pygame.quit(); sys.exit(0)

            if audio_data.size:
                visualise(screen, width, height, audio_data)

            pygame.display.flip()
            clock.tick(60)


if __name__ == "__main__":
    main()
