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
SMOOTHING_FACTOR = 0.2
MIN_FREQUENCY = 20
MAX_FREQUENCY = 2000

# Rotation configuration
ROTATION_BASE_SPEED = 0.001   # Base rotation speed
ROTATION_MAX_SPEED = 0.05     # Maximum rotation speed
ROTATION_SMOOTHING = 0.2      # Smoothing factor for rotation speed

# Global variables
audio_data = np.zeros(BLOCK_SIZE)
smoothed_magnitudes = np.zeros(BLOCK_SIZE // 2)
rotation_angle = 0
rotation_speed = ROTATION_BASE_SPEED
smoothed_rotation_speed = ROTATION_BASE_SPEED

# Simplex vertices and edges
vertices_4d = np.array([
    [1, 1, 1, -1/np.sqrt(5)],
    [1, -1, -1, -1/np.sqrt(5)],
    [-1, 1, -1, -1/np.sqrt(5)],
    [-1, -1, 1, -1/np.sqrt(5)],
    [0, 0, 0, np.sqrt(5) - (1/np.sqrt(5))]
])

edges = [
    (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 2), (1, 3), (1, 4),
    (2, 3), (2, 4),
    (3, 4)
]

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
    pygame.display.set_caption("Pentaflow")
    clock = pygame.time.Clock()
    print("Pygame initialized successfully")
    return screen, clock

def freq_to_color(freq, mag):
    norm_freq = (freq - MIN_FREQUENCY) / (MAX_FREQUENCY - MIN_FREQUENCY)
    hue = (norm_freq % 1.0) * 360
    saturation = 100  # Max saturation for vivid colors
    value = min(max(mag * 200, 50), 100)  # Brightness varies with magnitude
    color = pygame.Color(0, 0, 0)
    color.hsva = (hue, saturation, value, 100)
    return color

def rotation_matrix_4d(angles):
    """
    Create a combined rotation matrix for rotating in 4D space.
    Angles is a dictionary with plane keys ('xy', 'xz', 'xw', 'yz', 'yw', 'zw') and rotation angles.
    """
    c = {key: np.cos(angle) for key, angle in angles.items()}
    s = {key: np.sin(angle) for key, angle in angles.items()}

    R = np.identity(4)

    # Rotation in xy-plane
    R_xy = np.array([[ c['xy'], -s['xy'], 0, 0],
                     [ s['xy'],  c['xy'], 0, 0],
                     [     0,       0,    1, 0],
                     [     0,       0,    0, 1]])
    R = R @ R_xy

    # Rotation in xz-plane
    R_xz = np.array([[ c['xz'], 0, -s['xz'], 0],
                     [     0,   1,     0,    0],
                     [ s['xz'], 0,  c['xz'], 0],
                     [     0,   0,     0,    1]])
    R = R @ R_xz

    # Rotation in xw-plane
    R_xw = np.array([[ c['xw'], 0, 0, -s['xw']],
                     [     0,   1, 0,     0],
                     [     0,   0, 1,     0],
                     [ s['xw'], 0, 0,  c['xw']]])
    R = R @ R_xw

    # Rotation in yz-plane
    R_yz = np.array([[ 1,     0,     0,    0],
                     [ 0, c['yz'], -s['yz'], 0],
                     [ 0, s['yz'],  c['yz'], 0],
                     [ 0,     0,     0,    1]])
    R = R @ R_yz

    # Rotation in yw-plane
    R_yw = np.array([[ 1,     0,     0,    0],
                     [ 0, c['yw'], 0, -s['yw']],
                     [ 0,     0,   1,    0],
                     [ 0, s['yw'], 0,  c['yw']]])
    R = R @ R_yw

    # Rotation in zw-plane
    R_zw = np.array([[ 1,     0,     0,    0],
                     [ 0,     1,     0,    0],
                     [ 0,     0, c['zw'], -s['zw']],
                     [ 0,     0, s['zw'],  c['zw']]])
    R = R @ R_zw

    return R

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

    # Define rotation angles for different planes
    angle_factors = {
        'xy': 0.7,
        'xz': 0.4,
        'xw': 0.3,
        'yz': 0.6,
        'yw': 0.2,
        'zw': 0.9,
    }
    angles = {plane: rotation_angle * factor for plane, factor in angle_factors.items()}

    # Create rotation matrix
    R = rotation_matrix_4d(angles)

    # Rotate the simplex vertices
    rotated_vertices = vertices_4d @ R.T

    # Project from 4D to 3D using perspective projection
    distance_4d = 2  # Adjust this value to change 4D perspective
    projected_3d = []
    for v in rotated_vertices:
        w = 1 / (distance_4d - v[3])  # Perspective division from 4D to 3D
        x = v[0] * w
        y = v[1] * w
        z = v[2] * w
        projected_3d.append([x, y, z])

    # Project from 3D to 2D
    projected_2d = []
    distance_3d = 2  # Adjust this value to change 3D perspective
    fov = WIDTH / 2  # Field of view
    for v in projected_3d:
        w = 1 / (distance_3d - v[2])  # Perspective division from 3D to 2D
        x = int(WIDTH / 2 + v[0] * w * fov)
        y = int(HEIGHT / 2 + v[1] * w * fov)
        projected_2d.append([x, y])

    # Map frequencies to edges
    num_edges = len(edges)
    freq_indices = np.linspace(0, len(smoothed_magnitudes) - 1, num_edges, dtype=int)
    edge_magnitudes = smoothed_magnitudes[freq_indices]
    edge_frequencies = fft_frequency[freq_indices]

    # Draw edges with varying colors or thickness based on magnitudes
    for idx, (i, j) in enumerate(edges):
        mag = edge_magnitudes[idx]
        freq = edge_frequencies[idx]
        color = freq_to_color(freq, mag)
        thickness = int(1 + mag * 5)
        pygame.draw.line(screen, color, projected_2d[i], projected_2d[j], thickness)

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
