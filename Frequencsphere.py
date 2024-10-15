import numpy as np
import pygame
import sounddevice as sd
import time
import math

# Audio configuration
SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
DEVICE = None  # Use the default input device
CHANNELS = 2   # Set to 2 if your microphone is stereo

# Pygame configuration
WIDTH = 800
HEIGHT = 800
FPS = 60

# Visualization configuration
BASE_RADIUS = 100
MAX_RADIUS = 300
SMOOTHING_FACTOR = 0.3
MIN_FREQUENCY = 80
MAX_FREQUENCY = 1000

# Rotation configuration
ROTATION_BASE_SPEED = 0.005   # Base rotation speed
ROTATION_MAX_SPEED = 0.02     # Maximum rotation speed
ROTATION_SMOOTHING = 0.1      # Smoothing factor for rotation speed

# Global variables
audio_data = np.zeros(BLOCK_SIZE)
smoothed_magnitudes = np.zeros(BLOCK_SIZE // 2)
rotation_angle = 0
rotation_speed = ROTATION_BASE_SPEED
smoothed_rotation_speed = ROTATION_BASE_SPEED

def audio_callback(indata, frames, time, status):
    global audio_data
    if status:
        print(f"Audio callback status: {status}")
    if indata is not None and len(indata) > 0:
        audio_data = indata[:, 0].copy()
    else:
        print("No audio data received")

def initialize_pygame():
    print("Initializing Pygame...")
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    print("Pygame display mode set successfully")
    pygame.display.set_caption("Frequencsphere")
    clock = pygame.time.Clock()
    print("Pygame initialized successfully")
    return screen, clock

def freq_to_color(freq, mag):
    # Map frequency to hue (0-360)
    norm_freq = (freq - MIN_FREQUENCY) / (MAX_FREQUENCY - MIN_FREQUENCY)
    hue = (norm_freq % 1.0) * 360  # Ensure hue is between 0 and 360

    # Map magnitude to saturation
    saturation = min(max(mag * 100, 0), 100)  # Clamp between 0 and 100

    value = 100  # Full brightness

    color = pygame.Color(0, 0, 0)
    color.hsva = (hue, saturation, value, 100)

    return color

def visualize(screen, data):
    global smoothed_magnitudes, rotation_angle, rotation_speed, smoothed_rotation_speed

    # Calculate FFT
    fft_data = np.fft.fft(data)
    fft_magnitude = np.abs(fft_data[:len(data)//2])
    fft_frequency = np.fft.fftfreq(len(data), 1/SAMPLE_RATE)[:len(data)//2]

    # Normalize and smooth FFT magnitude
    max_magnitude = np.max(fft_magnitude)
    if max_magnitude > 0:
        normalized_magnitude = fft_magnitude / max_magnitude
    else:
        normalized_magnitude = np.zeros_like(fft_magnitude)

    smoothed_magnitudes = SMOOTHING_FACTOR * normalized_magnitude + (1 - SMOOTHING_FACTOR) * smoothed_magnitudes

    # Adjust rotation speed based on bass magnitude
    bass_freq_indices = np.where((fft_frequency >= 20) & (fft_frequency <= 200))[0]
    bass_magnitude = np.mean(smoothed_magnitudes[bass_freq_indices]) if len(bass_freq_indices) > 0 else 0

    # Map bass magnitude to rotation speed
    rotation_speed = ROTATION_BASE_SPEED + (bass_magnitude * (ROTATION_MAX_SPEED - ROTATION_BASE_SPEED))
    # Smooth the rotation speed to prevent abrupt changes
    smoothed_rotation_speed = (ROTATION_SMOOTHING * rotation_speed) + ((1 - ROTATION_SMOOTHING) * smoothed_rotation_speed)

    rotation_angle += smoothed_rotation_speed  # Update rotation angle with smoothed speed

    # Create fading effect
    fade_surface = pygame.Surface((WIDTH, HEIGHT))
    fade_surface.set_alpha(30)
    fade_surface.fill((0, 0, 0))
    screen.blit(fade_surface, (0, 0))

    center = (WIDTH // 2, HEIGHT // 2)

    # Draw frequency curves
    points = []
    colors = []
    for i, (mag, freq) in enumerate(zip(smoothed_magnitudes, fft_frequency)):
        if MIN_FREQUENCY <= freq <= MAX_FREQUENCY:
            angle = 2 * math.pi * (freq - MIN_FREQUENCY) / (MAX_FREQUENCY - MIN_FREQUENCY) + rotation_angle
            radius = BASE_RADIUS + int(mag * (MAX_RADIUS - BASE_RADIUS))
            x = int(center[0] + radius * math.cos(angle))
            y = int(center[1] + radius * math.sin(angle))
            points.append((x, y))
            colors.append(freq_to_color(freq, mag))

    if len(points) > 2:
        for i in range(len(points) - 1):
            pygame.draw.line(screen, colors[i], points[i], points[i+1], 2)
        pygame.draw.line(screen, colors[-1], points[-1], points[0], 2)  # Close the loop

def main():
    screen, clock = initialize_pygame()

    print("Starting audio stream...")
    try:
        audio_stream = sd.InputStream(
            callback=audio_callback,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            device=DEVICE,  # None to use default device
        )
        audio_stream.start()
    except Exception as e:
        print(f"Error starting audio stream: {e}")
        return

    print("Entering main loop...")
    running = True
    frame_count = 0
    start_time = time.time()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if len(audio_data) > 0:
            visualize(screen, audio_data)

        pygame.display.flip()
        clock.tick(FPS)

        frame_count += 1
        if frame_count % 60 == 0:
            print(f"FPS: {frame_count / (time.time() - start_time):.2f}")

    print("Cleaning up...")
    audio_stream.stop()
    pygame.quit()
    print(f"Script finished. Total frames: {frame_count}")

if __name__ == "__main__":
    main()
