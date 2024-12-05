import numpy as np
import pygame
import sounddevice as sd
import time
import math

# Audio configuration
SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
DEVICE = None  # Use the default input device
CHANNELS = 1   # Mono input

# Pygame configuration
WIDTH = 800
HEIGHT = 800
FPS = 60

# Visualization configuration
SMOOTHING_FACTOR = 0.2
MIN_FREQUENCY = 50
MAX_FREQUENCY = 2000  # Full audible spectrum

# Global variables
audio_data = np.zeros(BLOCK_SIZE)
smoothed_magnitudes = np.zeros(BLOCK_SIZE // 2)
color_wave_phase = 0  # Variable to shift colors around the mandala

def audio_callback(indata, frames, time, status):
    global audio_data
    if status:
        print(f"Audio callback status: {status}")
    if indata is not None and len(indata) > 0:
        audio_data = indata[:, 0].copy()
    else:
        print("No audio data received")

def initialize_pygame():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Mandalalala")
    clock = pygame.time.Clock()
    return screen, clock

def draw_mandala(screen, frequencies, magnitudes, color_wave_phase):
    num_points = 360  # Number of points around the circle
    center_x = WIDTH // 2
    center_y = HEIGHT // 2
    max_radius = min(WIDTH, HEIGHT) // 2 * 0.8  # Leave some padding

    angle_step = 2 * math.pi / num_points

    # Prepare frequencies and magnitudes
    freq_indices = np.linspace(0, len(magnitudes) - 1, num_points, dtype=int)
    magnitudes = magnitudes[freq_indices]
    frequencies = frequencies[freq_indices]

    # Compute points
    points = []
    for i in range(num_points):
        angle = i * angle_step
        mag = magnitudes[i]
        freq = frequencies[i]

        # Calculate radius with pulsing effect
        radius = max_radius * (0.5 + mag)

        # Calculate position
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        points.append((x, y))

    # Draw the mandala pattern
    for i in range(num_points):
        start_point = points[i]
        end_point = points[(i + num_points // 2) % num_points]  # Connect opposite points
        mag = magnitudes[i]
        freq = frequencies[i]
        angle = i * angle_step

        # Color based on angle and audio-induced phase shift
        hue = (angle / (2 * math.pi) * 360 + color_wave_phase) % 360
        saturation = 100
        value = min(max(mag * 500, 50), 100)
        color = pygame.Color(0, 0, 0)
        color.hsva = (hue, saturation, value, 100)

        # Draw line
        pygame.draw.line(screen, color, start_point, end_point, 2)

def main():
    global smoothed_magnitudes, color_wave_phase

    screen, clock = initialize_pygame()

    # Start audio stream
    try:
        audio_stream = sd.InputStream(
            callback=audio_callback,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            device=DEVICE,
        )
        audio_stream.start()
    except Exception as e:
        print(f"Error starting audio stream: {e}")
        return

    running = True
    while running:
        screen.fill((0, 0, 0))  # Clear screen

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Process audio data
        data = audio_data
        if len(data) > 0:
            # Calculate FFT
            fft_data = np.fft.fft(data)
            fft_magnitude = np.abs(fft_data[:len(data)//2])
            fft_frequency = np.fft.fftfreq(len(data), 1/SAMPLE_RATE)[:len(data)//2]

            # Only consider frequencies within our range
            freq_mask = (fft_frequency >= MIN_FREQUENCY) & (fft_frequency <= MAX_FREQUENCY)
            fft_frequency = fft_frequency[freq_mask]
            fft_magnitude = fft_magnitude[freq_mask]

            # Normalize and smooth FFT magnitude
            max_magnitude = np.max(fft_magnitude)
            if max_magnitude > 0:
                normalized_magnitude = fft_magnitude / max_magnitude
            else:
                normalized_magnitude = np.zeros_like(fft_magnitude)

            # Ensure smoothed_magnitudes has the correct length
            if len(smoothed_magnitudes) != len(normalized_magnitude):
                smoothed_magnitudes = np.zeros_like(normalized_magnitude)

            smoothed_magnitudes = SMOOTHING_FACTOR * normalized_magnitude + (1 - SMOOTHING_FACTOR) * smoothed_magnitudes

            # Update color_wave_phase based on bass magnitude
            bass_indices = np.where((fft_frequency >= 20) & (fft_frequency <= 200))[0]
            bass_magnitude = np.mean(smoothed_magnitudes[bass_indices]) if len(bass_indices) > 0 else 0

            # Adjust the scaling factor to get a suitable phase increment
            color_wave_phase += bass_magnitude * 50  # Adjust scaling factor as needed
            color_wave_phase %= 360  # Keep within 0-360 degrees

        else:
            # Default values if no audio input
            fft_frequency = np.linspace(MIN_FREQUENCY, MAX_FREQUENCY, BLOCK_SIZE // 2)
            smoothed_magnitudes = np.zeros(BLOCK_SIZE // 2)

        # Draw the mandala with moving colors
        draw_mandala(screen, fft_frequency, smoothed_magnitudes, color_wave_phase)

        pygame.display.flip()
        clock.tick(FPS)

    # Clean up
    audio_stream.stop()
    pygame.quit()

if __name__ == "__main__":
    main()
