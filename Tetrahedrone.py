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
SMOOTHING_FACTOR = 0.5
MIN_FREQUENCY = 20
MAX_FREQUENCY = 1000

# Global variables
audio_data = np.zeros(BLOCK_SIZE)
smoothed_magnitudes = np.zeros(BLOCK_SIZE // 2)

# Variables for rotation
rotation_x = 0
rotation_y = 0
rotation_z = 0

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
    pygame.display.set_caption("Tetrahedrone")
    clock = pygame.time.Clock()
    print("Pygame initialized successfully")
    return screen, clock

def generate_sierpinski_tetrahedron(order, vertices):
    if order == 0:
        # Return the faces (triangles) of the tetrahedron
        faces = [
            [vertices[0], vertices[1], vertices[2]],
            [vertices[0], vertices[1], vertices[3]],
            [vertices[0], vertices[2], vertices[3]],
            [vertices[1], vertices[2], vertices[3]]
        ]
        return faces
    else:
        # Midpoints of edges
        midpoints = {}
        edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        for i, j in edges:
            midpoints[(i,j)] = (np.array(vertices[i]) + np.array(vertices[j])) / 2

        # New vertices for sub-tetrahedra
        v0 = vertices[0]
        v1 = midpoints[(0,1)]
        v2 = midpoints[(0,2)]
        v3 = midpoints[(0,3)]
        v4 = midpoints[(1,2)]
        v5 = midpoints[(1,3)]
        v6 = midpoints[(2,3)]
        v7 = (np.array(vertices[0]) + np.array(vertices[1]) + np.array(vertices[2]) + np.array(vertices[3])) / 4

        # Recursively generate sub-tetrahedra
        faces = []
        faces += generate_sierpinski_tetrahedron(order - 1, [v0, v1, v2, v3])
        faces += generate_sierpinski_tetrahedron(order - 1, [v1, vertices[1], v4, v5])
        faces += generate_sierpinski_tetrahedron(order - 1, [v2, v4, vertices[2], v6])
        faces += generate_sierpinski_tetrahedron(order - 1, [v3, v5, v6, vertices[3]])
        return faces

def rotation_matrix_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rotation_matrix_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def rotation_matrix_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def project(points):
    # Perspective projection onto 2D plane
    projected_points = []
    distance = 5  # Distance from viewer to projection plane
    for point in points:
        x, y, z = point
        factor = distance / (distance + z)
        x_proj = x * factor * WIDTH / 2 + WIDTH / 2
        y_proj = -y * factor * HEIGHT / 2 + HEIGHT / 2  # Negative y to flip coordinate system
        projected_points.append((x_proj, y_proj))
    return projected_points

def freq_to_color(freq, mag, time_offset):
    # Map frequency to hue
    norm_freq = (freq - MIN_FREQUENCY) / (MAX_FREQUENCY - MIN_FREQUENCY)
    hue = (norm_freq * 360 + time_offset) % 360
    saturation = 100
    value = min(max(mag * 500, 50), 100)
    color = pygame.Color(0, 0, 0)
    color.hsva = (hue, saturation, value, 100)
    return color

def visualize(screen, data):
    global smoothed_magnitudes, rotation_x, rotation_y, rotation_z

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

    # Compute scale factor
    base_scale = 0.5
    amplitude = 0.5  # Adjust as needed
    scale_factor = base_scale + amplitude * overall_magnitude

    # Update rotation angles
    rotation_speed = 0.01
    rotation_x += rotation_speed
    rotation_y += rotation_speed * 0.4
    rotation_z += rotation_speed * 0.3

    # Clear screen with transparency to create a fading effect
    fade_surface = pygame.Surface((WIDTH, HEIGHT))
    fade_surface.set_alpha(10)
    fade_surface.fill((0, 0, 0))
    screen.blit(fade_surface, (0, 0))

    # Define initial tetrahedron vertices
    initial_vertices = [
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1]
    ]

    # Generate Sierpiński tetrahedron faces
    order = 3  # Adjust for complexity (higher order increases detail)
    faces = generate_sierpinski_tetrahedron(order, initial_vertices)

    # Prepare rotation matrices
    Rx = rotation_matrix_x(rotation_x)
    Ry = rotation_matrix_y(rotation_y)
    Rz = rotation_matrix_z(rotation_z)
    R = Rx @ Ry @ Rz

    # Time offset for color cycling
    time_offset = time.time() * 50  # Adjust speed of color cycling

    # Map frequencies to faces
    num_faces = len(faces)
    freq_indices = np.linspace(0, len(smoothed_magnitudes) - 1, num_faces, dtype=int)
    magnitudes = smoothed_magnitudes[freq_indices]
    frequencies = fft_frequency[freq_indices]

    # Draw faces
    for idx, face in enumerate(faces):
        vertices = np.array(face)

        # Apply scaling
        vertices *= scale_factor

        # Apply rotation
        vertices = vertices @ R.T

        # Project to 2D
        projected_vertices = project(vertices)

        # Determine color
        mag = magnitudes[idx]
        freq = frequencies[idx]
        color = freq_to_color(freq, mag, time_offset)

        # Draw the face
        pygame.draw.polygon(screen, color, projected_vertices)

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
