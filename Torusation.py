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
SMOOTHING_FACTOR = 0.5
MIN_FREQUENCY = 20
MAX_FREQUENCY = 2000

# Rotation configuration
ROTATION_BASE_SPEED = 0.001   # Base rotation speed
ROTATION_MAX_SPEED = 0.05      # Maximum rotation speed
ROTATION_SMOOTHING = 0.2      # Smoothing factor for rotation speed

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
    pygame.display.set_caption("Torusation")
    clock = pygame.time.Clock()
    print("Pygame initialized successfully")
    return screen, clock

def freq_to_color(freq, mag):
    norm_freq = (freq - MIN_FREQUENCY) / (MAX_FREQUENCY - MIN_FREQUENCY)
    hue = (norm_freq % 1.0) * 360
    saturation = 100  # Set saturation to maximum
    # Adjust value (brightness) based on magnitude
    value = min(max(mag * 200, 50), 100)
    color = pygame.Color(0, 0, 0)
    color.hsva = (hue, saturation, value, 100)
    return color

def rotation_matrix(axis, angle):
    """
    Create a rotation matrix for rotating around a specified axis by a given angle.
    """
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

    # Clear screen with transparency to create a fading effect
    fade_surface = pygame.Surface((WIDTH, HEIGHT))
    fade_surface.set_alpha(10)
    fade_surface.fill((0, 0, 0))
    screen.blit(fade_surface, (0, 0))

    # Generate torus vertices
    NUM_U = 50
    NUM_V = 25
    R = 1.5  # Major radius
    r = 0.5  # Minor radius

    # Map audio magnitudes to modulate the minor radius
    mag_interp = np.interp(np.linspace(0, len(smoothed_magnitudes), NUM_V), np.arange(len(smoothed_magnitudes)), smoothed_magnitudes)
    amplitude_factor = 0.3  # Adjust the pulsation effect

    u = np.linspace(0, 2 * np.pi, NUM_U)
    v = np.linspace(0, 2 * np.pi, NUM_V)
    U, V = np.meshgrid(u, v)

    # Modulate minor radius with audio magnitudes
    r_modulated = r + mag_interp[:, np.newaxis] * amplitude_factor

    X = (R + r_modulated * np.cos(V)) * np.cos(U)
    Y = (R + r_modulated * np.cos(V)) * np.sin(U)
    Z = r_modulated * np.sin(V)

    # Rotate the torus
    angle_x = rotation_angle * 0.3
    angle_y = rotation_angle * 0.5
    angle_z = rotation_angle * 0.2
    R_x = rotation_matrix('x', angle_x)
    R_y = rotation_matrix('y', angle_y)
    R_z = rotation_matrix('z', angle_z)
    R_total = R_z @ R_y @ R_x

    points = np.vstack((X.ravel(), Y.ravel(), Z.ravel()))
    rotated_points = R_total @ points
    X_rotated = rotated_points[0, :].reshape(NUM_V, NUM_U)
    Y_rotated = rotated_points[1, :].reshape(NUM_V, NUM_U)
    Z_rotated = rotated_points[2, :].reshape(NUM_V, NUM_U)

    # Invert the coordinates to create an inside view
    X_rotated *= -1
    Y_rotated *= -1
    Z_rotated *= -1

    # Project to 2D
    fov = WIDTH / 2
    distance = 3  # Adjust to change perspective
    x_2d = fov * X_rotated / (Z_rotated + distance) + WIDTH / 2
    y_2d = fov * Y_rotated / (Z_rotated + distance) + HEIGHT / 2

    # Map frequencies to colors
    num_points = NUM_V
    freq_indices = np.linspace(0, len(smoothed_magnitudes) - 1, num_points, dtype=int)
    point_magnitudes = smoothed_magnitudes[freq_indices]
    point_frequencies = fft_frequency[freq_indices]

    # Draw the torus as a mesh (wireframe)
    for i in range(NUM_V):
        for j in range(NUM_U):
            # Current point
            x_current = x_2d[i, j]
            y_current = y_2d[i, j]

            # Next point in U direction
            x_next_u = x_2d[i, (j + 1) % NUM_U]
            y_next_u = y_2d[i, (j + 1) % NUM_U]

            # Next point in V direction
            x_next_v = x_2d[(i + 1) % NUM_V, j]
            y_next_v = y_2d[(i + 1) % NUM_V, j]

            # Compute color based on the magnitude and frequency
            mag = point_magnitudes[i]
            freq = point_frequencies[i]
            color = freq_to_color(freq, mag)

            # Draw lines to next points to create the mesh
            pygame.draw.line(screen, color, (x_current, y_current), (x_next_u, y_next_u), 1)
            pygame.draw.line(screen, color, (x_current, y_current), (x_next_v, y_next_v), 1)

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
