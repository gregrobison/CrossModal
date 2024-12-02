import pygame
import numpy as np
import sounddevice as sd
import time

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
MAX_DEPTH = 5    # Adjust for more or fewer circles
SMOOTHING_FACTOR = 0.3
MIN_FREQUENCY = 20
MAX_FREQUENCY = 20000  # Increased to cover the full audible spectrum

# Global variables
audio_data = np.zeros(BLOCK_SIZE)
smoothed_magnitudes = np.zeros(BLOCK_SIZE // 2)

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
    pygame.display.set_caption("Apollonian")
    clock = pygame.time.Clock()
    return screen, clock

def get_frequency_range(depth):
    """
    Map depth levels to frequency ranges.
    Larger circles (lower depths) correspond to lower frequencies.
    """
    # Define frequency bands for each depth
    frequency_bands = [
        (20, 200),          # Depth 0: 20Hz - 200Hz
        (200, 500),         # Depth 1: 200Hz - 500Hz
        (500, 1000),        # Depth 2: 500Hz - 1kHz
        (1000, 5000),       # Depth 3: 1kHz - 5kHz
        (5000, 10000),      # Depth 4: 5kHz - 10kHz
        (10000, 20000),     # Depth 5: 10kHz - 20kHz
    ]
    if depth >= len(frequency_bands):
        depth = len(frequency_bands) - 1  # Use the highest frequency band for deeper levels
    return frequency_bands[depth]

def apollonian_gasket(screen, x, y, r, depth, fft_frequency, smoothed_magnitudes):
    if depth > MAX_DEPTH or r < 1:
        return

    # Get the frequency range for this depth
    freq_min, freq_max = get_frequency_range(depth)

    # Find indices corresponding to this frequency range
    freq_indices = np.where((fft_frequency >= freq_min) & (fft_frequency <= freq_max))[0]

    if len(freq_indices) == 0:
        mag = 0
        freq = (freq_min + freq_max) / 2
    else:
        # Average magnitude and frequency in this range
        mag = np.mean(smoothed_magnitudes[freq_indices])
        freq = np.mean(fft_frequency[freq_indices])

    # Pulse effect based on audio magnitude
    pulse = 1 + mag * 0.7  # Adjust the scaling factor as needed
    r_pulse = r * pulse

    # Movement effect based on audio magnitude
    move_amplitude = mag * 10  # Adjust the scaling factor as needed
    x_move = x + move_amplitude * np.sin(freq * 0.001)
    y_move = y + move_amplitude * np.cos(freq * 0.001)

    # Color based on frequency and magnitude
    hue = (freq - MIN_FREQUENCY) / (MAX_FREQUENCY - MIN_FREQUENCY) * 360
    saturation = 100
    value = min(max(mag * 500, 50), 100)
    color = pygame.Color(0, 0, 0)
    color.hsva = (hue % 360, saturation, value, 100)

    # Draw the circle
    pygame.draw.circle(screen, color, (int(x_move), int(y_move)), int(r_pulse), 2)

    # Recursive calls for inner circles
    new_r = r / 2
    # Positions for the inner circles
    positions = [
        (x - new_r, y),
        (x + new_r, y),
        (x, y - new_r),
        (x, y + new_r)
    ]

    for new_x, new_y in positions:
        apollonian_gasket(screen, new_x, new_y, new_r, depth + 1, fft_frequency, smoothed_magnitudes)

def main():
    global smoothed_magnitudes

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

            smoothed_magnitudes = SMOOTHING_FACTOR * normalized_magnitude + (1 - SMOOTHING_FACTOR) * smoothed_magnitudes[:len(normalized_magnitude)]
        else:
            # Default values if no audio input
            fft_frequency = np.linspace(MIN_FREQUENCY, MAX_FREQUENCY, BLOCK_SIZE // 2)
            smoothed_magnitudes = np.zeros(BLOCK_SIZE // 2)

        # Starting circle
        x_center = WIDTH / 2
        y_center = HEIGHT / 2
        radius = WIDTH / 3

        apollonian_gasket(screen, x_center, y_center, radius, depth=0, fft_frequency=fft_frequency, smoothed_magnitudes=smoothed_magnitudes)

        pygame.display.flip()
        clock.tick(FPS)

    # Clean up
    audio_stream.stop()
    pygame.quit()

if __name__ == "__main__":
    main()
