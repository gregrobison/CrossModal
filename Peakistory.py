import numpy as np
import pygame
import sounddevice as sd
import time

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
NUM_BINS = 60          # Number of frequency bins to display
HISTORY_LENGTH = 50    # Number of historical frames to display
MIN_FREQUENCY = 80
MAX_FREQUENCY = 1000

# Global variables
audio_data = np.zeros(BLOCK_SIZE)
history = []  # Store historical FFT magnitudes

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
    pygame.display.set_caption("3D Frequency Visualization")
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
    global history

    # Calculate FFT
    fft_data = np.fft.fft(data)
    fft_magnitude = np.abs(fft_data[:len(data)//2])
    fft_frequency = np.fft.fftfreq(len(data), 1/SAMPLE_RATE)[:len(data)//2]

    # Filter frequencies within the desired range
    freq_indices = np.where((fft_frequency >= MIN_FREQUENCY) & (fft_frequency <= MAX_FREQUENCY))[0]
    fft_magnitude = fft_magnitude[freq_indices]
    fft_frequency = fft_frequency[freq_indices]

    # Bin the frequencies
    bins = np.linspace(MIN_FREQUENCY, MAX_FREQUENCY, NUM_BINS + 1)
    digitized = np.digitize(fft_frequency, bins)
    binned_magnitude = [fft_magnitude[digitized == i].mean() if np.any(digitized == i) else 0 for i in range(1, len(bins))]

    # Normalize magnitudes
    max_magnitude = np.max(binned_magnitude)
    if max_magnitude > 0:
        normalized_magnitude = binned_magnitude / max_magnitude
    else:
        normalized_magnitude = np.zeros_like(binned_magnitude)

    # Add the new frame to history
    history.insert(0, normalized_magnitude)
    if len(history) > HISTORY_LENGTH:
        history.pop()

    # Clear the screen
    screen.fill((0, 0, 0))

    # Vanishing point coordinates at the top-center
    vanishing_point = (WIDTH / 2, 0)

    # Visualize the history
    for depth, magnitudes in enumerate(history):
        # Calculate scaling factor for perspective
        depth_factor = depth / HISTORY_LENGTH
        scale = 1 - depth_factor  # Scale decreases with depth

        # Calculate Y position (bars move down as they recede)
        y = HEIGHT * depth_factor

        # Adjust alpha for fading effect
        alpha = int(255 * (1 - depth_factor))

        for i, mag in enumerate(magnitudes):
            freq = (bins[i] + bins[i+1]) / 2  # Midpoint of the bin
            color = freq_to_color(freq, mag)
            color.a = alpha  # Set alpha for fading effect

            # Perspective transformation for X position
            # Bars converge towards the vanishing point as they recede
            x_ratio = (i - NUM_BINS / 2) / (NUM_BINS / 2)  # Range from -1 to 1
            x = WIDTH / 2 + x_ratio * (WIDTH / 2) * scale

            # Bar dimensions
            bar_width = (WIDTH / NUM_BINS) * scale
            bar_height = mag * (HEIGHT / 2) * scale

            # Draw the bar
            rect = pygame.Rect(x - bar_width / 2, y, bar_width, bar_height)
            pygame.draw.rect(screen, color, rect)

    # Draw horizon line (optional)
    pygame.draw.line(screen, (50, 50, 50), (0, vanishing_point[1]), (WIDTH, vanishing_point[1]), 2)

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
