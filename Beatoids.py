import numpy as np
import pygame
import sounddevice as sd
import time
import math
import random
from pygame.math import Vector2

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
NUM_BOIDS = 100
SMOOTHING_FACTOR = 0.2

# Global variables
audio_data = np.zeros(BLOCK_SIZE)
smoothed_magnitudes = np.zeros(BLOCK_SIZE // 2)

# Boid class
class Boid:
    def __init__(self):
        self.position = Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
        self.velocity = Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        if self.velocity.length() == 0:
            self.velocity = Vector2(1, 0)
        else:
            self.velocity.scale_to_length(random.uniform(1, 3))  # Initial speed
        self.acceleration = Vector2(0, 0)
        self.max_speed = 4
        self.max_force = 0.1
        self.perception_radius = 50
        self.size = 5  # Default size
        self.color = (255, 255, 255)  # Default color (white)
        self.base_hue = random.uniform(0, 360)  # Each boid starts with a random hue
    
    def update(self):
        self.velocity += self.acceleration
        if self.velocity.length() > self.max_speed:
            self.velocity.scale_to_length(self.max_speed)
        self.position += self.velocity
        self.acceleration *= 0  # Reset acceleration

        # Wrap around screen edges
        if self.position.x > WIDTH:
            self.position.x = 0
        elif self.position.x < 0:
            self.position.x = WIDTH
        if self.position.y > HEIGHT:
            self.position.y = 0
        elif self.position.y < 0:
            self.position.y = HEIGHT

    def apply_force(self, force):
        self.acceleration += force

    def flock(self, boids):
        alignment = self.align(boids)
        cohesion = self.cohere(boids)
        separation = self.separate(boids)
        self.apply_force(alignment)
        self.apply_force(cohesion)
        self.apply_force(separation)

    def align(self, boids):
        steering = Vector2(0, 0)
        total = 0
        avg_vector = Vector2(0, 0)
        for other in boids:
            if other != self and self.position.distance_to(other.position) < self.perception_radius:
                avg_vector += other.velocity
                total += 1
        if total > 0:
            avg_vector /= total
            if avg_vector.length() > 0:
                avg_vector.scale_to_length(self.max_speed)
                steering = avg_vector - self.velocity
                if steering.length() > 0:
                    if steering.length() > self.max_force:
                        steering.scale_to_length(self.max_force)
        return steering

    def cohere(self, boids):
        steering = Vector2(0, 0)
        total = 0
        center_of_mass = Vector2(0, 0)
        for other in boids:
            if other != self and self.position.distance_to(other.position) < self.perception_radius:
                center_of_mass += other.position
                total += 1
        if total > 0:
            center_of_mass /= total
            desired = center_of_mass - self.position
            if desired.length() > 0:
                desired.scale_to_length(self.max_speed)
                steering = desired - self.velocity
                if steering.length() > 0:
                    if steering.length() > self.max_force:
                        steering.scale_to_length(self.max_force)
        return steering

    def separate(self, boids):
        steering = Vector2(0, 0)
        total = 0
        for other in boids:
            distance = self.position.distance_to(other.position)
            if other != self and distance < self.perception_radius:
                diff = self.position - other.position
                if distance > 0:
                    diff /= distance  # Weight by distance
                steering += diff
                total += 1
        if total > 0:
            steering /= total
            if steering.length() > 0:
                steering.scale_to_length(self.max_speed)
                steering -= self.velocity
                if steering.length() > 0:
                    if steering.length() > self.max_force:
                        steering.scale_to_length(self.max_force)
            else:
                steering = Vector2(0, 0)
        return steering

    def draw(self, screen):
        # Draw boid as a triangle pointing in the direction of velocity
        angle = self.velocity.angle_to(Vector2(1, 0))
        size = self.size
        points = [
            (self.position + Vector2(size, 0).rotate(-angle)),
            (self.position + Vector2(-size / 2, size / 2).rotate(-angle)),
            (self.position + Vector2(-size / 2, -size / 2).rotate(-angle)),
        ]
        pygame.draw.polygon(screen, self.color, points)

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
    pygame.display.set_caption("Beatoids")
    clock = pygame.time.Clock()
    print("Pygame initialized successfully")
    return screen, clock

def visualize(screen, boids, data):
    global smoothed_magnitudes

    # Clear the screen
    screen.fill((0, 0, 0))

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
    bass_indices = np.where((fft_frequency >= 20) & (fft_frequency <= 250))[0]
    mid_indices = np.where((fft_frequency >= 250) & (fft_frequency <= 2000))[0]
    treble_indices = np.where((fft_frequency >= 2000) & (fft_frequency <= 8000))[0]

    bass_magnitude = np.mean(smoothed_magnitudes[bass_indices]) if len(bass_indices) > 0 else 0
    mid_magnitude = np.mean(smoothed_magnitudes[mid_indices]) if len(mid_indices) > 0 else 0
    treble_magnitude = np.mean(smoothed_magnitudes[treble_indices]) if len(treble_indices) > 0 else 0

    perception_radius_factor = bass_magnitude * 100
    size_factor = mid_magnitude * 200
    color_shift = treble_magnitude * 200

    for boid in boids:
        boid.perception_radius = max(30, min(150, 50 + perception_radius_factor))
        boid.size = max(2, min(25, 5 + size_factor))
        hue = (boid.base_hue + color_shift) % 360
        color = pygame.Color(0)
        color.hsla = (hue, 100, 50, 100)
        boid.color = color
        boid.flock(boids)
        boid.update()
        boid.draw(screen)

def main():
    screen, clock = initialize_pygame()

    boids = [Boid() for _ in range(NUM_BOIDS)]

    try:
        audio_stream = sd.InputStream(
            callback=audio_callback,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            device=DEVICE
        )
        audio_stream.start()
    except Exception as e:
        print(f"Error starting audio stream: {e}")
        return

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if len(audio_data) > 0:
            visualize(screen, boids, audio_data)

        pygame.display.flip()
        clock.tick(FPS)

    audio_stream.stop()
    pygame.quit()

if __name__ == "__main__":
    main()
