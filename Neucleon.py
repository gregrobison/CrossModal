import numpy as np
import pygame
import sounddevice as sd
import time
import math
import random

# Audio configuration
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
DEVICE = None  # Use the default input device
CHANNELS = 1   # Mono input for simplicity

# Pygame configuration
WIDTH = 800
HEIGHT = 800
FPS = 60

# Visualization configuration
SMOOTHING_FACTOR = 0.8  # Adjusted for proper smoothing

# Atom configuration
NUM_ELECTRONS = 100  # Adjust for desired complexity
NUM_SHELLS = 10      # Number of electron shells

# Global variables
audio_data = np.zeros(BLOCK_SIZE)
smoothed_magnitudes = np.zeros(BLOCK_SIZE // 2)
smoothed_magnitude = 0

# 3D projection parameters
CAMERA_DISTANCE = 1000
FOV = 500  # Field of view
PROJECTION_CENTER = (WIDTH // 2, HEIGHT // 2)

class Electron:
    def __init__(self, shell_radius, inclination, azimuth, speed, freq_range):
        self.shell_radius = shell_radius
        self.inclination = inclination  # Angle from the z-axis
        self.azimuth = azimuth          # Angle around the z-axis
        self.speed = speed
        self.color = (255, 255, 255)
        self.size = 10
        self.freq_range = freq_range  # Tuple of (start_freq, end_freq)

    def update(self, smoothed_magnitudes, fft_frequency):
        # Get the magnitude for the electron's frequency range
        freq_indices = np.where((fft_frequency >= self.freq_range[0]) & (fft_frequency < self.freq_range[1]))[0]
        if len(freq_indices) > 0:
            magnitude = np.mean(smoothed_magnitudes[freq_indices])
        else:
            magnitude = 0

        # Update angles based on speed and audio magnitude
        self.azimuth = (self.azimuth + self.speed * (1 + magnitude * 5)) % (2 * math.pi)
        self.inclination = (self.inclination + self.speed * 0.1 * (1 + magnitude * 5)) % math.pi

        # Update size based on audio magnitude with more pulsing
        self.size = 3 + magnitude * 30 + math.sin(time.time() * 10) * 5

        # Update color based on magnitude and frequency
        hue = (self.freq_range[0] / 20) % 360  # Base hue on frequency
        saturation = 100 - magnitude * 50  # Reduce saturation with higher magnitude
        value = 50 + magnitude * 50  # Increase brightness with higher magnitude
        color = pygame.Color(0)
        color.hsva = (hue, saturation, value, 100)
        self.color = color

    def get_3d_position(self):
        # Calculate 3D position based on spherical coordinates
        x = self.shell_radius * math.sin(self.inclination) * math.cos(self.azimuth)
        y = self.shell_radius * math.sin(self.inclination) * math.sin(self.azimuth)
        z = self.shell_radius * math.cos(self.inclination)
        return np.array([x, y, z])

    def project(self, position):
        # Project 3D position to 2D screen coordinates using perspective projection
        factor = FOV / (CAMERA_DISTANCE + position[2])
        x = int(PROJECTION_CENTER[0] + position[0] * factor)
        y = int(PROJECTION_CENTER[1] - position[1] * factor)  # Invert y-axis for correct orientation
        return x, y

    def draw(self, screen):
        position = self.get_3d_position()
        x, y = self.project(position)
        # Adjust size based on depth
        size = self.size * (FOV / (CAMERA_DISTANCE + position[2]))
        pygame.draw.circle(screen, self.color, (x, y), max(1, int(size)))

def audio_callback(indata, frames, time_info, status):
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
    pygame.display.set_caption("Nucleon")
    clock = pygame.time.Clock()
    print("Pygame initialized successfully")
    return screen, clock

def create_electrons(fft_frequency):
    electrons = []
    shell_spacing = 400 / NUM_SHELLS
    freq_bins = np.logspace(np.log10(20), np.log10(SAMPLE_RATE / 2), NUM_ELECTRONS + 1)
    for i in range(NUM_ELECTRONS):
        shell = i % NUM_SHELLS + 1
        shell_radius = shell * shell_spacing
        inclination = random.uniform(0, math.pi)
        azimuth = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.01, 0.05)
        freq_range = (freq_bins[i], freq_bins[i + 1])
        electron = Electron(shell_radius, inclination, azimuth, speed, freq_range)
        electrons.append(electron)
    return electrons

def rotate_3d(position, angles):
    # Rotate position around x, y, and z axes
    x_angle, y_angle, z_angle = angles
    x, y, z = position

    # Rotation around x-axis
    cos_x, sin_x = math.cos(x_angle), math.sin(x_angle)
    y, z = y * cos_x - z * sin_x, y * sin_x + z * cos_x

    # Rotation around y-axis
    cos_y, sin_y = math.cos(y_angle), math.sin(y_angle)
    x, z = x * cos_y + z * sin_y, -x * sin_y + z * cos_y

    # Rotation around z-axis
    cos_z, sin_z = math.cos(z_angle), math.sin(z_angle)
    x, y = x * cos_z - y * sin_z, x * sin_z + y * cos_z

    return np.array([x, y, z])

def visualize(screen, electrons, nucleus_particles, angles, data):
    global smoothed_magnitudes, smoothed_magnitude

    # Calculate the audio signal's magnitude
    magnitude = np.linalg.norm(data) / len(data)
    # Smooth the magnitude to prevent abrupt changes
    smoothed_magnitude = (SMOOTHING_FACTOR * magnitude) + ((1 - SMOOTHING_FACTOR) * smoothed_magnitude)

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

    smoothed_magnitudes = (SMOOTHING_FACTOR * normalized_magnitude) + ((1 - SMOOTHING_FACTOR) * smoothed_magnitudes)

    # Map the smoothed magnitude to rotation speed
    rotation_speed = 0.01 + smoothed_magnitude * 0.8  # Adjusted for better sensitivity

    # Clear screen
    screen.fill((0, 0, 0))

    # Update and draw electrons
    for electron in electrons:
        electron.update(smoothed_magnitudes, fft_frequency)
        # Get electron 3D position and rotate it
        position = electron.get_3d_position()
        rotated_position = rotate_3d(position, angles)
        electron.project(rotated_position)
        electron.draw(screen)

    # Update and draw nucleus particles
    for particle in nucleus_particles:
        # Rotate particle position
        rotated_position = rotate_3d(particle['position'], angles)
        x, y = project(rotated_position)
        # Adjust size based on depth
        size = particle['size'] * (FOV / (CAMERA_DISTANCE + rotated_position[2]))
        # Update color based on position and audio magnitude
        hue = (particle['position'][0] * 360 / 100 + smoothed_magnitude * 180) % 360
        saturation = 100 - smoothed_magnitude * 50
        value = 50 + smoothed_magnitude * 50
        color = pygame.Color(0)
        color.hsva = (hue, saturation, value, 100)
        pygame.draw.circle(screen, color, (x, y), max(1, int(size)))


    return rotation_speed

def project(position):
    # Project 3D position to 2D screen coordinates using perspective projection
    factor = FOV / (CAMERA_DISTANCE + position[2])
    x = int(PROJECTION_CENTER[0] + position[0] * factor)
    y = int(PROJECTION_CENTER[1] - position[1] * factor)  # Invert y-axis for correct orientation
    return x, y

def create_nucleus_particles():
    # Create particles representing the nucleus in 3D space
    nucleus_particles = []
    num_particles = 100  # Increased number of protons and neutrons for better visual
    for _ in range(num_particles):
        # Random position inside a sphere
        theta = random.uniform(0, math.pi)
        phi = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0, 50)  # Nucleus radius
        x = radius * math.sin(theta) * math.cos(phi)
        y = radius * math.sin(theta) * math.sin(phi)
        z = radius * math.cos(theta)
        position = np.array([x, y, z])
        size = random.randint(3, 8)
        nucleus_particles.append({'position': position, 'size': size})
    return nucleus_particles

def main():
    screen, clock = initialize_pygame()
    # Generate initial FFT frequency array for electron frequency ranges
    temp_data = np.zeros(BLOCK_SIZE)
    fft_frequency = np.fft.fftfreq(len(temp_data), 1/SAMPLE_RATE)[:len(temp_data)//2]
    electrons = create_electrons(fft_frequency)
    nucleus_particles = create_nucleus_particles()
    angles = [0, 0, 0]

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
            rotation_speed = visualize(screen, electrons, nucleus_particles, angles, audio_data)
            # Update rotation angles
            angles[0] += rotation_speed * 1.2  # Rotate around x-axis
            angles[1] += rotation_speed * 1.5  # Rotate around y-axis
            angles[2] += rotation_speed * 1.2  # Rotate around z-axis

        pygame.display.flip()
        clock.tick(FPS)

        frame_count += 1
        if frame_count % 60 == 0:
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                print(f"FPS: {frame_count / elapsed_time:.2f}")

    print("Cleaning up...")
    audio_stream.stop()
    pygame.quit()
    print(f"Script finished. Total frames: {frame_count}")

if __name__ == "__main__":
    main()
