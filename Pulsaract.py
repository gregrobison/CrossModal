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
SMOOTHING_FACTOR = 0.3
MIN_FREQUENCY = 20
MAX_FREQUENCY = 1000

# Rotation configuration
ROTATION_BASE_SPEED = 0.002   # Base rotation speed
ROTATION_MAX_SPEED = 0.05     # Maximum rotation speed
ROTATION_SMOOTHING = 0.2      # Smoothing factor for rotation speed

# Vibration configuration
VIBRATION_AMPLITUDE = 20  # Maximum vibration amplitude in pixels
THICKNESS_BASE = 5  # Base thickness of the lines
THICKNESS_SCALE = 30  # Scale factor for thickness variation

# Global variables
audio_data = np.zeros(BLOCK_SIZE)
smoothed_magnitudes = np.zeros(BLOCK_SIZE // 2)
rotation_angle = 0
rotation_speed = ROTATION_BASE_SPEED
smoothed_rotation_speed = ROTATION_BASE_SPEED

# Tesseract vertices and edges
vertices_4d = np.array([[x, y, z, w] for x in (-1, 1) for y in (-1, 1)
                        for z in (-1, 1) for w in (-1, 1)])

edges = []
for i in range(len(vertices_4d)):
    for j in range(i + 1, len(vertices_4d)):
        if np.sum(np.abs(vertices_4d[i] - vertices_4d[j])) == 2:
            edges.append((i, j))

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
    pygame.display.set_caption("Pulsaract")
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

def rotation_matrix_4d(angle_xy=0, angle_xz=0, angle_xw=0, angle_yz=0, angle_yw=0, angle_zw=0):
    # Start with identity matrix
    R = np.identity(4)
    # Rotation in xy-plane
    if angle_xy != 0:
        c, s = np.cos(angle_xy), np.sin(angle_xy)
        R_xy = np.array([[ c, -s,  0,  0],
                         [ s,  c,  0,  0],
                         [ 0,  0,  1,  0],
                         [ 0,  0,  0,  1]])
        R = R @ R_xy
    # Rotation in xz-plane
    if angle_xz != 0:
        c, s = np.cos(angle_xz), np.sin(angle_xz)
        R_xz = np.array([[ c,  0, -s,  0],
                         [ 0,  1,  0,  0],
                         [ s,  0,  c,  0],
                         [ 0,  0,  0,  1]])
        R = R @ R_xz
    # Rotation in xw-plane
    if angle_xw != 0:
        c, s = np.cos(angle_xw), np.sin(angle_xw)
        R_xw = np.array([[ c,  0,  0, -s],
                         [ 0,  1,  0,  0],
                         [ 0,  0,  1,  0],
                         [ s,  0,  0,  c]])
        R = R @ R_xw
    # Rotation in yz-plane
    if angle_yz != 0:
        c, s = np.cos(angle_yz), np.sin(angle_yz)
        R_yz = np.array([[ 1,  0,  0,  0],
                         [ 0,  c, -s,  0],
                         [ 0,  s,  c,  0],
                         [ 0,  0,  0,  1]])
        R = R @ R_yz
    # Rotation in yw-plane
    if angle_yw != 0:
        c, s = np.cos(angle_yw), np.sin(angle_yw)
        R_yw = np.array([[ 1,  0,  0,  0],
                         [ 0,  c,  0, -s],
                         [ 0,  0,  1,  0],
                         [ 0,  s,  0,  c]])
        R = R @ R_yw
    # Rotation in zw-plane
    if angle_zw != 0:
        c, s = np.cos(angle_zw), np.sin(angle_zw)
        R_zw = np.array([[ 1,  0,  0,  0],
                         [ 0,  1,  0,  0],
                         [ 0,  0,  c, -s],
                         [ 0,  0,  s,  c]])
        R = R @ R_zw
    return R

def visualize(screen, data):
    global smoothed_magnitudes, rotation_angle, rotation_speed, smoothed_rotation_speed, vertices_4d

    current_time = time.time()  # Get the current time for time-based vibration

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

    rotation_speed = ROTATION_BASE_SPEED + (bass_magnitude * (ROTATION_MAX_SPEED - ROTATION_BASE_SPEED))
    smoothed_rotation_speed = (ROTATION_SMOOTHING * rotation_speed) + ((1 - ROTATION_SMOOTHING) * smoothed_rotation_speed)
    rotation_angle += smoothed_rotation_speed

    # Create fading effect
    fade_surface = pygame.Surface((WIDTH, HEIGHT))
    fade_surface.set_alpha(10)
    fade_surface.fill((0, 0, 0))
    screen.blit(fade_surface, (0, 0))

    center = (WIDTH // 2, HEIGHT // 2)

    # Rotate the tesseract vertices
    angle = rotation_angle
    R = rotation_matrix_4d(
        angle_xy=angle * 0.2,
        angle_xz=angle * 0.3,
        angle_xw=angle * 0.2,
        angle_yz=angle * 0.4,
        angle_yw=angle * 0.1,
        angle_zw=angle * 0.2
    )
    rotated_vertices = vertices_4d @ R.T

    # Project from 4D to 3D
    distance = 3
    projected_3d = []
    for v in rotated_vertices:
        w = 1 / (distance - v[3])
        x = v[0] * w
        y = v[1] * w
        z = v[2] * w
        projected_3d.append([x, y, z])

    # Project from 3D to 2D
    projected_2d = []
    for v in projected_3d:
        fov = WIDTH / 2
        x = int(center[0] + v[0] * fov)
        y = int(center[1] + v[1] * fov)
        projected_2d.append([x, y])

    # Map frequencies to edges
    num_edges = len(edges)
    freq_indices = np.linspace(0, len(smoothed_magnitudes) - 1, num_edges, dtype=int)
    edge_magnitudes = smoothed_magnitudes[freq_indices]
    edge_frequencies = fft_frequency[freq_indices]

    for idx, (i, j) in enumerate(edges):
        mag = edge_magnitudes[idx]
        freq = edge_frequencies[idx]
        color = freq_to_color(freq, mag)

        # Calculate thickness
        thickness = int(THICKNESS_BASE + (mag * THICKNESS_SCALE))

        # Frequency-based vibration
        vibration_x = VIBRATION_AMPLITUDE * mag * np.sin(2 * np.pi * freq * current_time)
        vibration_y = VIBRATION_AMPLITUDE * mag * np.cos(2 * np.pi * freq * current_time)

        start_point = (int(projected_2d[i][0] + vibration_x), int(projected_2d[i][1] + vibration_y))
        end_point = (int(projected_2d[j][0] + vibration_x), int(projected_2d[j][1] + vibration_y))

        pygame.draw.line(screen, color, start_point, end_point, thickness)

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
