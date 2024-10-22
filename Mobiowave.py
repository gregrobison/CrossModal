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
ROTATION_BASE_SPEED = 0.0001   # Base rotation speed
ROTATION_MAX_SPEED = 0.2     # Maximum rotation speed
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
    pygame.display.set_caption("Möbiowave")
    clock = pygame.time.Clock()
    print("Pygame initialized successfully")
    return screen, clock

def freq_to_color(freq, mag):
    norm_freq = (freq - MIN_FREQUENCY) / (MAX_FREQUENCY - MIN_FREQUENCY)
    hue = (norm_freq % 1.0) * 360
    saturation = min(max(mag * 200, 50), 100)  # Increased minimum saturation
    value = 100
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

    # Create fading effect
    fade_surface = pygame.Surface((WIDTH, HEIGHT))
    fade_surface.set_alpha(10)
    fade_surface.fill((0, 0, 0))
    screen.blit(fade_surface, (0, 0))

    # Generate Möbius strip vertices
    NUM_U = 100
    NUM_V = 20
    STRIP_WIDTH = 2
    u = np.linspace(0, 2 * np.pi, NUM_U)
    v = np.linspace(-STRIP_WIDTH, STRIP_WIDTH, NUM_V)
    U, V = np.meshgrid(u, v)

    # Map audio magnitudes to modulate the strip
    mag_interp = np.interp(u, np.linspace(0, 2 * np.pi, len(smoothed_magnitudes)), smoothed_magnitudes)
    amplitude_factor = 0.5
    # Modulate the z-coordinate to create undulations
    scale_factor = 2.0  # Adjust this value to change the size
    Z = scale_factor * (V * np.sin(U / 2) + mag_interp[np.newaxis, :] * amplitude_factor)
    X = scale_factor * ((1 + V * np.cos(U / 2)) * np.cos(U))
    Y = scale_factor * ((1 + V * np.cos(U / 2)) * np.sin(U))

    # Rotate the strip
    angle_x = rotation_angle * 0.3
    angle_y = rotation_angle * 0.5
    angle_z = rotation_angle * 0.2
    R_x = rotation_matrix('x', angle_x)
    R_y = rotation_matrix('y', angle_y)
    R_z = rotation_matrix('z', angle_z)
    R = R_z @ R_y @ R_x

    points = np.vstack((X.ravel(), Y.ravel(), Z.ravel()))
    rotated_points = R @ points
    X_rotated = rotated_points[0, :].reshape(NUM_V, NUM_U)
    Y_rotated = rotated_points[1, :].reshape(NUM_V, NUM_U)
    Z_rotated = rotated_points[2, :].reshape(NUM_V, NUM_U)

    # Project to 2D
    fov = WIDTH / 2
    distance = 5 * scale_factor  # Adjust to change perspective
    x_2d = fov * X_rotated / (Z_rotated + distance) + WIDTH / 2
    y_2d = fov * Y_rotated / (Z_rotated + distance) + HEIGHT / 2

    # Map frequencies to colors
    num_points = NUM_U
    freq_indices = np.linspace(0, len(smoothed_magnitudes) - 1, num_points, dtype=int)
    point_magnitudes = smoothed_magnitudes[freq_indices]
    point_frequencies = fft_frequency[freq_indices]

    # Draw the strip
    for i in range(NUM_V - 1):
        for j in range(NUM_U - 1):
            # Get the four corners of the quad
            x1, y1 = x_2d[i, j], y_2d[i, j]
            x2, y2 = x_2d[i, j + 1], y_2d[i, j + 1]
            x3, y3 = x_2d[i + 1, j + 1], y_2d[i + 1, j + 1]
            x4, y4 = x_2d[i + 1, j], y_2d[i + 1, j]

            # Compute color based on the magnitude and frequency
            mag = point_magnitudes[j]
            freq = point_frequencies[j]
            color = freq_to_color(freq, mag)

            # Draw the quad as two triangles
            pygame.draw.polygon(screen, color, [(x1, y1), (x2, y2), (x3, y3)])
            pygame.draw.polygon(screen, color, [(x1, y1), (x3, y3), (x4, y4)])

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
