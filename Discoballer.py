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
DISCO_BALL_RADIUS = 200
NUM_LATITUDE = 20  # Number of horizontal divisions
NUM_LONGITUDE = 30  # Number of vertical divisions
SMOOTHING_FACTOR = 0.3  # Reduced for faster response

# Global variables
audio_data = np.zeros(BLOCK_SIZE)
smoothed_magnitudes = np.zeros(BLOCK_SIZE // 2)

# Camera configuration
CAMERA_DISTANCE = 400  # Distance from the camera to the center of the sphere
FOV = 500  # Field of view for perspective projection

# Panel class
class Panel:
    def __init__(self, theta, phi, frequency_indices):
        # Spherical coordinates
        self.theta = theta
        self.phi = phi
        self.frequency_indices = frequency_indices
        self.color = pygame.Color(255, 255, 255)

        # Initial position in 3D space
        self.x, self.y, self.z = self.spherical_to_cartesian(DISCO_BALL_RADIUS, theta, phi)
        self.initial_vertices = self.create_panel_vertices()
        self.vertices = [v.copy() for v in self.initial_vertices]

    def spherical_to_cartesian(self, r, theta, phi):
        x = r * math.sin(theta) * math.cos(phi)
        y = r * math.cos(theta)
        z = r * math.sin(theta) * math.sin(phi)
        return x, y, z

    def create_panel_vertices(self):
        # Small offset for panel size
        delta_theta = math.pi / NUM_LATITUDE / 1.5
        delta_phi = 2 * math.pi / NUM_LONGITUDE / 1.5
        vertices = []
        for dtheta, dphi in [(-delta_theta, -delta_phi), (-delta_theta, delta_phi),
                             (delta_theta, delta_phi), (delta_theta, -delta_phi)]:
            theta = self.theta + dtheta
            phi = self.phi + dphi
            x, y, z = self.spherical_to_cartesian(DISCO_BALL_RADIUS, theta, phi)
            vertices.append(np.array([x, y, z]))
        return vertices

    def update(self, rotation_matrix, magnitudes, frequencies):
        # Apply rotation to initial vertices
        self.vertices = [np.dot(rotation_matrix, v) for v in self.initial_vertices]

        # Get average magnitude and frequency for assigned frequencies
        magnitude = np.mean(magnitudes[self.frequency_indices])
        avg_frequency = np.mean(frequencies[self.frequency_indices])

        # Map frequency to color
        min_freq = 20  # Human hearing range minimum
        max_freq = SAMPLE_RATE / 2  # Nyquist frequency
        normalized_freq = (avg_frequency - min_freq) / (max_freq - min_freq)
        normalized_freq = max(0, min(1, normalized_freq))  # Clamp between 0 and 1
        hue = normalized_freq * 360  # Map normalized frequency to hue (0 to 360 degrees)

        saturation = 100  # Full saturation
        value = min(100, max(0, magnitude * 200))  # Adjusted for higher sensitivity

        self.color.hsva = (hue, saturation, value, 100)

    def project(self, vertex):
        # Perspective projection
        x, y, z = vertex
        z += CAMERA_DISTANCE  # Translate camera
        if z == 0:
            z = 0.0001  # Avoid division by zero
        factor = FOV / z
        x_proj = x * factor + WIDTH // 2
        y_proj = -y * factor + HEIGHT // 2  # Invert y-axis for screen coordinates
        return [int(x_proj), int(y_proj)]

    def draw(self, screen):
        # Project all vertices
        projected_vertices = [self.project(v) for v in self.vertices]

        # Draw the panel as a polygon
        pygame.draw.polygon(screen, self.color, projected_vertices)

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
    pygame.display.set_caption("Discoballer")
    clock = pygame.time.Clock()
    print("Pygame initialized successfully")
    return screen, clock

def visualize(screen, panels, data):
    global smoothed_magnitudes

    # Clear the screen
    screen.fill((0, 0, 0))

    # Calculate FFT
    fft_data = np.fft.fft(data)
    fft_magnitude = np.abs(fft_data[:len(data)//2])

    # Normalize and smooth FFT magnitude
    max_magnitude = np.max(fft_magnitude)
    if max_magnitude > 0:
        normalized_magnitude = fft_magnitude / max_magnitude
    else:
        normalized_magnitude = np.zeros_like(fft_magnitude)

    smoothed_magnitudes = (SMOOTHING_FACTOR * normalized_magnitude +
                           (1 - SMOOTHING_FACTOR) * smoothed_magnitudes)

    # Calculate rotation speed based on overall magnitude
    overall_magnitude = np.mean(smoothed_magnitudes)
    rotation_speed_multiplier = 0.5  # Increased for higher sensitivity
    rotation_speed = max(0.1, overall_magnitude * rotation_speed_multiplier)

    # Get the current time in seconds
    current_time = pygame.time.get_ticks() / 1000.0
    # Compute rotation angles based on time and audio magnitude
    angle_x = current_time * rotation_speed * 0.5  # Adjust as needed
    angle_y = current_time * rotation_speed

    # Create rotation matrices based on angles
    cos_angle_x = math.cos(angle_x)
    sin_angle_x = math.sin(angle_x)
    cos_angle_y = math.cos(angle_y)
    sin_angle_y = math.sin(angle_y)

    rotation_matrix_y = np.array([
        [cos_angle_y, 0, sin_angle_y],
        [0, 1, 0],
        [-sin_angle_y, 0, cos_angle_y]
    ])
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, cos_angle_x, -sin_angle_x],
        [0, sin_angle_x, cos_angle_x]
    ])
    rotation_matrix = np.dot(rotation_matrix_x, rotation_matrix_y)

    # Precompute frequencies for mapping colors
    frequencies = np.linspace(0, SAMPLE_RATE / 2, len(smoothed_magnitudes))

    # Update and draw panels
    for panel in panels:
        panel.update(rotation_matrix, smoothed_magnitudes, frequencies)
        panel.draw(screen)

def main():
    screen, clock = initialize_pygame()

    # Create panels
    panels = []
    num_frequency_bins = BLOCK_SIZE // 2
    total_panels = NUM_LATITUDE * NUM_LONGITUDE
    frequency_bins_per_panel = max(1, num_frequency_bins // total_panels)

    # Generate panels on the sphere
    for i in range(NUM_LATITUDE):
        theta = math.pi * (i + 0.5) / NUM_LATITUDE  # Avoid poles
        for j in range(NUM_LONGITUDE):
            phi = 2 * math.pi * j / NUM_LONGITUDE
            panel_index = i * NUM_LONGITUDE + j
            start_idx = panel_index * frequency_bins_per_panel
            end_idx = start_idx + frequency_bins_per_panel
            frequency_indices = np.arange(start_idx, min(end_idx, num_frequency_bins))
            panel = Panel(theta, phi, frequency_indices)
            panels.append(panel)

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
            visualize(screen, panels, audio_data)

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
