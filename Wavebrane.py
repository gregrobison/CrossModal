import numpy as np
import pygame
import sounddevice as sd
import time
import math
import colorsys

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
SMOOTHING_FACTOR = 0.2
MIN_FREQUENCY = 20
MAX_FREQUENCY = 1000

# Rotation configuration
ROTATION_BASE_SPEED = 0.001
ROTATION_MAX_SPEED = 0.5
ROTATION_SMOOTHING = 0.2

# Grid configuration
GRID_SIZE = 20
GRID_COLOR = (30, 30, 30)  # Darker grid lines
LINE_WIDTH = 3

# Psychedelic color configuration
COLOR_SHIFT_SPEED = 0.05  # Slightly faster color shifting
COLOR_INTENSITY_FACTOR = 5.0  # Increased color intensity
SATURATION_BASE = 0.9  # Higher base saturation for stronger colors

# Trail configuration
TRAIL_LENGTH = 5  # Number of previous frames to keep in the trail
TRAIL_ALPHA_STEP = 40  # Alpha decrease per trail frame (out of 255)

# Global variables
audio_data = np.zeros(BLOCK_SIZE)
smoothed_magnitudes = np.zeros(BLOCK_SIZE // 2)
rotation_angle = 0
rotation_speed = ROTATION_BASE_SPEED
smoothed_rotation_speed = ROTATION_BASE_SPEED
color_offset = 0
trail_surfaces = []

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
    pygame.display.set_caption("Wavebrane")
    clock = pygame.time.Clock()
    print("Pygame initialized successfully")
    return screen, clock

def rotation_matrix(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])

def get_psychedelic_color(height, intensity):
    global color_offset
    hue = (height + color_offset) % 1.0
    saturation = min(SATURATION_BASE + intensity * COLOR_INTENSITY_FACTOR, 1.0)
    value = min(0.7 + intensity * 0.3, 1.0)  # Increased base value for brighter colors
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)

def visualize(screen, data):
    global smoothed_magnitudes, rotation_angle, rotation_speed, smoothed_rotation_speed, color_offset, trail_surfaces

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

    # Adjust rotation speed based on overall magnitude
    overall_magnitude = np.mean(smoothed_magnitudes)
    rotation_speed = ROTATION_BASE_SPEED + (overall_magnitude * (ROTATION_MAX_SPEED - ROTATION_BASE_SPEED))
    smoothed_rotation_speed = (ROTATION_SMOOTHING * rotation_speed) + ((1 - ROTATION_SMOOTHING) * smoothed_rotation_speed)
    rotation_angle += smoothed_rotation_speed

    # Update color offset
    color_offset += COLOR_SHIFT_SPEED * overall_magnitude

    # Create a new surface for the current frame
    current_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    # Generate grid for the plane
    x = np.linspace(-2, 2, GRID_SIZE)
    y = np.linspace(-2, 2, GRID_SIZE)
    X, Y = np.meshgrid(x, y)

    # Create a more reactive surface
    Z = np.zeros_like(X)
    for i in range(len(smoothed_magnitudes)):
        mag = smoothed_magnitudes[i]
        freq = fft_frequency[i]
        if mag > 0.01 and freq > 0:
            Z += mag * np.sin(2 * np.pi * freq * (X + Y) * 0.5)  # Increased frequency impact

    # Normalize Z and apply a more dramatic amplitude
    Z = Z / np.max(np.abs(Z)) if np.max(np.abs(Z)) != 0 else Z
    amplitude = overall_magnitude * 4  # Increased amplitude for more reactivity
    Z *= amplitude

    # Rotate the plane
    angle_x = rotation_angle * 0.9
    angle_y = rotation_angle * 0.9
    angle_z = rotation_angle * 0.9
    R_x = rotation_matrix('x', angle_x)
    R_y = rotation_matrix('y', angle_y)
    R_z = rotation_matrix('z', angle_z)
    R = R_z @ R_y @ R_x

    points = np.vstack((X.ravel(), Y.ravel(), Z.ravel()))
    rotated_points = R @ points
    X_rotated = rotated_points[0, :].reshape(GRID_SIZE, GRID_SIZE)
    Y_rotated = rotated_points[1, :].reshape(GRID_SIZE, GRID_SIZE)
    Z_rotated = rotated_points[2, :].reshape(GRID_SIZE, GRID_SIZE)

    # Project to 2D
    fov = WIDTH / 2
    distance = 5
    x_2d = fov * X_rotated / (Z_rotated + distance) + WIDTH / 2
    y_2d = fov * Y_rotated / (Z_rotated + distance) + HEIGHT / 2

    # Draw the colored grid cells on the current surface
    for i in range(GRID_SIZE - 1):
        for j in range(GRID_SIZE - 1):
            points = [
                (x_2d[i, j], y_2d[i, j]),
                (x_2d[i, j+1], y_2d[i, j+1]),
                (x_2d[i+1, j+1], y_2d[i+1, j+1]),
                (x_2d[i+1, j], y_2d[i+1, j])
            ]
            height = (Z_rotated[i, j] + Z_rotated[i+1, j] + Z_rotated[i, j+1] + Z_rotated[i+1, j+1]) / 4
            color = get_psychedelic_color(height, overall_magnitude)
            pygame.draw.polygon(current_surface, color, points)

    # Draw the grid lines on the current surface
    for i in range(GRID_SIZE):
        pygame.draw.lines(current_surface, GRID_COLOR, False, list(zip(x_2d[i], y_2d[i])), LINE_WIDTH)
        pygame.draw.lines(current_surface, GRID_COLOR, False, list(zip(x_2d[:, i], y_2d[:, i])), LINE_WIDTH)

    # Add the current surface to the trail
    trail_surfaces.append(current_surface)
    if len(trail_surfaces) > TRAIL_LENGTH:
        trail_surfaces.pop(0)

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the trail
    for i, surface in enumerate(trail_surfaces):
        alpha = 255 - (TRAIL_LENGTH - i - 1) * TRAIL_ALPHA_STEP
        surface.set_alpha(alpha)
        screen.blit(surface, (0, 0))

def main():
    screen, clock = initialize_pygame()

    print("Starting audio stream...")
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