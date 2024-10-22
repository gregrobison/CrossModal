import numpy as np
import pygame
import sounddevice as sd
import time
import math
import random

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
NUM_PARTICLES = 800
MAX_ORBIT_RADIUS = 400
MIN_ORBIT_RADIUS = 50
PARTICLE_SIZE = 3
SMOOTHING_FACTOR = 0.2
BLACK_HOLE_RADIUS = 30  # Radius of the black hole

# Global variables
audio_data = np.zeros(BLOCK_SIZE)
smoothed_magnitudes = np.zeros(BLOCK_SIZE // 2)

# Particle class
class Particle:
    def __init__(self):
        self.reset_particle()
        self.gravity_factor = 1  # Adjust this value to change gravity strength

    def reset_particle(self):
        self.angle = random.uniform(0, 2 * math.pi)
        self.radius = random.uniform(MIN_ORBIT_RADIUS, MAX_ORBIT_RADIUS)
        self.speed = random.uniform(0.001, 0.01)
        self.size = PARTICLE_SIZE
        self.orbit_center = (WIDTH // 2, HEIGHT // 2)

    def update(self, audio_reactive_speed, audio_reactive_radius):
        # Update the angle based on speed
        self.angle += self.speed * audio_reactive_speed
        self.angle %= 2 * math.pi  # Keep angle within 0 to 2π

        # Apply gravitational pull
        self.radius -= self.gravity_factor * (self.radius / MAX_ORBIT_RADIUS)

        # Update radius based on audio input
        self.radius += audio_reactive_radius
        if self.radius > MAX_ORBIT_RADIUS:
            self.radius = MAX_ORBIT_RADIUS
        elif self.radius < BLACK_HOLE_RADIUS:
            # Respawn particle when it reaches the black hole
            self.reset_particle()

    def draw(self, screen):
        x = int(self.orbit_center[0] + self.radius * math.cos(self.angle))
        y = int(self.orbit_center[1] + self.radius * math.sin(self.angle))
        # Color changes based on distance from the center
        distance_ratio = (self.radius - MIN_ORBIT_RADIUS) / (MAX_ORBIT_RADIUS - MIN_ORBIT_RADIUS)
        distance_ratio = max(0, min(distance_ratio, 1))  # Clamp between 0 and 1

        color = pygame.Color(0, 0, 0)
        color.hsva = (distance_ratio * 360, 100, 100, 100)
        pygame.draw.circle(screen, color, (x, y), int(self.size))

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
    pygame.display.set_caption("Soundgularity")
    clock = pygame.time.Clock()
    print("Pygame initialized successfully")
    return screen, clock

def visualize(screen, particles, data):
    global smoothed_magnitudes

    # Fill the screen with a faint black to create a fading trail effect
    fade_surface = pygame.Surface((WIDTH, HEIGHT))
    fade_surface.set_alpha(40)
    fade_surface.fill((0, 0, 0))
    screen.blit(fade_surface, (0, 0))

    center = (WIDTH // 2, HEIGHT // 2)

    # Draw the black hole at the center
    pygame.draw.circle(screen, (0, 0, 0), center, BLACK_HOLE_RADIUS)

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

    # Calculate audio-reactive parameters
    # Use low frequencies to affect radius (bass frequencies)
    bass_freq_indices = np.where((fft_frequency >= 20) & (fft_frequency <= 250))[0]
    bass_magnitude = np.mean(smoothed_magnitudes[bass_freq_indices]) if len(bass_freq_indices) > 0 else 0

    # Use mid to high frequencies to affect speed
    treble_freq_indices = np.where((fft_frequency >= 2500) & (fft_frequency <= 5000))[0]
    treble_magnitude = np.mean(smoothed_magnitudes[treble_freq_indices]) if len(treble_freq_indices) > 0 else 0

    # Increase reactivity
    audio_reactive_radius = (bass_magnitude - 0.3) * 10  # Adjusted sensitivity
    audio_reactive_speed = 1 + treble_magnitude * 50     # Adjusted sensitivity

    # Update and draw particles
    for particle in particles:
        particle.update(audio_reactive_speed, audio_reactive_radius)
        particle.draw(screen)

def main():
    screen, clock = initialize_pygame()

    # Create particles
    particles = [Particle() for _ in range(NUM_PARTICLES)]

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
            visualize(screen, particles, audio_data)

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
