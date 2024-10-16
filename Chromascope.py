import numpy as np
import pygame
import sounddevice as sd
import time
import math

# Audio configuration
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
DEVICE = None  # Use the default input device
CHANNELS = 1   # Mono input for simplicity

# Pygame configuration
WIDTH = 800
HEIGHT = 800
FPS = 60

# Visualization configuration
SMOOTHING_FACTOR = 0.5

# Global variables
audio_data = np.zeros(BLOCK_SIZE)
smoothed_magnitudes = np.zeros(BLOCK_SIZE // 2)

def audio_callback(indata, frames, time_info, status):
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
    pygame.display.set_caption("Chromascope")
    clock = pygame.time.Clock()
    print("Pygame initialized successfully")
    return screen, clock

def visualize(screen, data):
    global smoothed_magnitudes

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

    # Map magnitudes to visualization parameters
    avg_magnitude = np.mean(smoothed_magnitudes)
    angle_offset = avg_magnitude * math.pi * 2  # Rotate patterns based on audio magnitude

    # Clear screen
    screen.fill((0, 0, 0))

    # Number of segments in the kaleidoscope
    num_segments = 16
    angle_per_segment = 2 * math.pi / num_segments

    # Generate base pattern
    pattern_surface = pygame.Surface((WIDTH // 2, HEIGHT // 2), pygame.SRCALPHA)
    center = (WIDTH // 4, HEIGHT // 4)

    # Draw shapes based on audio frequencies
    for i, (mag, freq) in enumerate(zip(smoothed_magnitudes, fft_frequency)):
        if mag > 0.1:  # Threshold to reduce noise
            radius = mag * (WIDTH // 4)
            angle = angle_offset + freq / (SAMPLE_RATE / 2) * 2 * math.pi
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)

            color = pygame.Color(0)
            hue = (freq / (SAMPLE_RATE / 2)) * 360
            color.hsva = (hue, 100, 100, 100)

            pygame.draw.circle(pattern_surface, color, (int(x), int(y)), int(mag * 10))

    # Create kaleidoscope effect by rotating and mirroring the pattern
    for i in range(num_segments):
        rotated_surface = pygame.transform.rotate(pattern_surface, -math.degrees(angle_per_segment * i))
        screen.blit(rotated_surface, (WIDTH // 2 - rotated_surface.get_width() // 2, HEIGHT // 2 - rotated_surface.get_height() // 2))

        flipped_surface = pygame.transform.flip(rotated_surface, True, False)
        screen.blit(flipped_surface, (WIDTH // 2 - flipped_surface.get_width() // 2, HEIGHT // 2 - flipped_surface.get_height() // 2))

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
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                print(f"FPS: {frame_count / elapsed_time:.2f}")

    print("Cleaning up...")
    audio_stream.stop()
    pygame.quit()
    print(f"Script finished. Total frames: {frame_count}")

if __name__ == "__main__":
    main()
