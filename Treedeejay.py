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
MIN_FREQUENCY = 20
MAX_FREQUENCY = 2000

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
    print("Initializing Pygame...")
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    print("Pygame display mode set successfully")
    pygame.display.set_caption("Treedeejay")
    clock = pygame.time.Clock()
    print("Pygame initialized successfully")
    return screen, clock

def freq_to_color(freq, mag, time_offset):
    # Map frequency to hue
    norm_freq = (freq - MIN_FREQUENCY) / (MAX_FREQUENCY - MIN_FREQUENCY)
    hue = (norm_freq * 360 + time_offset) % 360
    saturation = 100
    value = min(max(mag * 500, 50), 100)
    color = pygame.Color(0, 0, 0)
    color.hsva = (hue, saturation, value, 100)
    return color

def draw_fractal_tree(screen, x, y, angle, depth, max_depth, length, base_thickness, frequencies, magnitudes, time_offset):
    if depth > max_depth:
        return

    # Map depth to frequency index
    freq_idx = int((depth / max_depth) * (len(frequencies) - 1))
    freq = frequencies[freq_idx]
    mag = magnitudes[freq_idx]

    # Determine color based on frequency and magnitude
    color = freq_to_color(freq, mag, time_offset)

    # Determine line thickness
    thickness = int(base_thickness * (1 - depth / (max_depth + 1)))
    if thickness < 1:
        thickness = 1

    # Calculate new branch length
    branch_length = length * (0.9 + 0.3 * mag)  # Branch length influenced by magnitude

    # Calculate end position of the branch
    end_x = x + branch_length * math.cos(angle)
    end_y = y - branch_length * math.sin(angle)

    # Draw the branch
    pygame.draw.line(screen, color, (x, y), (end_x, end_y), thickness)

    # Recursive calls for the two new branches
    new_depth = depth + 2
    angle_variation = 0.5  # Base angle variation
    angle_variation += mag * 0.5  # Angle variation influenced by magnitude
    new_length = branch_length * 0.7  # Shorten the branch for next depth

    draw_fractal_tree(screen, end_x, end_y, angle - angle_variation, new_depth, max_depth, new_length, base_thickness, frequencies, magnitudes, time_offset)
    draw_fractal_tree(screen, end_x, end_y, angle + angle_variation, new_depth, max_depth, new_length, base_thickness, frequencies, magnitudes, time_offset)

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

    # Compute overall magnitude
    overall_magnitude = np.mean(smoothed_magnitudes)

    # Clear screen with transparency to create a fading effect
    fade_surface = pygame.Surface((WIDTH, HEIGHT))
    fade_surface.set_alpha(30)
    fade_surface.fill((0, 0, 0))
    screen.blit(fade_surface, (0, 0))

    # Time offset for color cycling
    time_offset = time.time() * 50  # Adjust speed of color cycling

    # Prepare frequencies and magnitudes for the tree
    num_levels = 20  # Adjust for tree depth complexity
    freq_indices = np.linspace(0, len(smoothed_magnitudes) - 1, num_levels, dtype=int)
    tree_magnitudes = smoothed_magnitudes[freq_indices]
    tree_frequencies = fft_frequency[freq_indices]

    # Initial tree parameters
    base_length = 250 + overall_magnitude * 300  # Base length influenced by audio magnitude
    base_thickness = 15  # Starting thickness of the trunk
    start_x = WIDTH / 2
    start_y = HEIGHT  # Start from the bottom center
    start_angle = math.pi / 2  # Start pointing upwards

    # Draw the fractal tree
    draw_fractal_tree(
        screen,
        start_x,
        start_y,
        start_angle,
        depth=0,
        max_depth=num_levels,
        length=base_length,
        base_thickness=base_thickness,
        frequencies=tree_frequencies,
        magnitudes=tree_magnitudes,
        time_offset=time_offset
    )

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
