import numpy as np
import pygame
import sounddevice as sd
import time
import random
import math

# Audio configuration
SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
DEVICE = None  # Use the default input device
CHANNELS = 1   # Mono input for simplicity

# Pygame configuration
WIDTH = 800
HEIGHT = 800
FPS = 60

# Visualization configuration
SMOOTHING_FACTOR = 0.2  # Adjusted for responsiveness
USE_COLOR = True  # Set to False for black and white
MAX_POINT_SIZE = 15  # Maximum size of the spiral points

# Flow field configuration
FLOW_FIELD_SCALE = 20  # Scale of the flow field grid
PARTICLE_COUNT = 500  # Number of particles in the flow field
PARTICLE_BASE_SIZE = 2  # Base size of particles
PARTICLE_MAX_SIZE = 15   # Maximum size of particles on beat
FLOW_SPEED = 2  # Base speed of particles

# Beat detection configuration
beat_threshold = 0.05  # Initial threshold for beat detection
beat_decay = 0.95      # How quickly the threshold decays (0 < beat_decay < 1)
beat_sensitivity = 1.8 # Multiplier to adjust sensitivity to beats

# Global variables
audio_data = np.zeros(BLOCK_SIZE)
smoothed_magnitude = 0
previous_magnitude = 0
beat_detected = False

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
    pygame.display.set_caption("Spirhythm")
    clock = pygame.time.Clock()
    print("Pygame initialized successfully")
    return screen, clock

def create_flow_field(cols, rows):
    """Create a flow field with random vectors."""
    flow_field = []
    for y in range(rows):
        flow_field_row = []
        for x in range(cols):
            angle = random.uniform(0, 2 * np.pi)
            vector = pygame.math.Vector2(np.cos(angle), np.sin(angle))
            flow_field_row.append(vector)
        flow_field.append(flow_field_row)
    return flow_field

class Particle:
    def __init__(self, flow_field, cols, rows):
        self.pos = pygame.math.Vector2(random.uniform(0, WIDTH), random.uniform(0, HEIGHT))
        angle = random.uniform(0, 2 * np.pi)
        self.vel = pygame.math.Vector2(np.cos(angle), np.sin(angle)) * 0.1  # Small initial velocity
        self.acc = pygame.math.Vector2(0, 0)
        self.max_speed = FLOW_SPEED
        self.flow_field = flow_field
        self.cols = cols
        self.rows = rows
        self.base_size = PARTICLE_BASE_SIZE
        self.size = self.base_size
        self.max_size = PARTICLE_MAX_SIZE
        self.color = pygame.Color(255, 255, 255)
        self.hue = random.randint(0, 360)

    def update(self, magnitude, beat_detected):
        # Update particle acceleration based on the flow field
        x = int(self.pos.x / FLOW_FIELD_SCALE)
        y = int(self.pos.y / FLOW_FIELD_SCALE)

        if x >= 0 and x < self.cols and y >= 0 and y < self.rows:
            force = self.flow_field[y][x]
            self.acc = force * magnitude * 5  # Increase the influence of audio magnitude
        else:
            # Assign a small random acceleration if out of bounds
            angle = random.uniform(0, 2 * np.pi)
            self.acc = pygame.math.Vector2(np.cos(angle), np.sin(angle)) * 0.1

        # Adjust max speed based on magnitude
        self.max_speed = FLOW_SPEED + magnitude * 10  # Adjust multiplier as needed

        self.vel += self.acc

        # Only scale if the velocity length is greater than zero
        if self.vel.length() > 0:
            self.vel.scale_to_length(min(self.vel.length(), self.max_speed))
        else:
            # Assign a small random velocity to prevent zero-length vector
            angle = random.uniform(0, 2 * np.pi)
            self.vel = pygame.math.Vector2(np.cos(angle), np.sin(angle)) * 0.1

        self.pos += self.vel
        self.acc *= 0

        # Wrap around screen edges
        if self.pos.x > WIDTH:
            self.pos.x = 0
        elif self.pos.x < 0:
            self.pos.x = WIDTH
        if self.pos.y > HEIGHT:
            self.pos.y = 0
        elif self.pos.y < 0:
            self.pos.y = HEIGHT

        # Update particle size based on beat detection
        if beat_detected:
            self.size = self.max_size
        else:
            # Smoothly decrease size back to base size
            self.size -= 0.5  # Decrease rate adjusted for smoother animation
            if self.size < self.base_size:
                self.size = self.base_size

        # Update color dynamically based on magnitude
        if USE_COLOR:
            self.hue = (self.hue + magnitude * 100) % 360  # Adjust the multiplier as needed
            self.color.hsva = (self.hue, 100, 100, 100)
        else:
            self.color = (255, 255, 255)

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.pos.x), int(self.pos.y)), int(self.size))

def draw_rotating_spiral(screen, center, max_radius, angle_offset, magnitude):
    num_spirals = 6  # Number of spiral arms
    num_points = 3000  # Number of points along the spiral

    # Set a higher minimum point size
    min_point_size = 4  # Adjust as needed
    max_point_size = MAX_POINT_SIZE  # Use the increased MAX_POINT_SIZE

    for i in range(num_points):
        t = i / float(num_points)
        theta = t * num_spirals * 2 * np.pi + angle_offset
        radius = max_radius * t
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)

        # Adjust point size based on magnitude
        point_size = int(min_point_size + magnitude * (max_point_size - min_point_size))
        point_size = max(min_point_size, min(point_size, max_point_size))

        # Generate color based on magnitude
        if USE_COLOR:
            hue = (t * 360 + magnitude * 200) % 360  # Adjust as needed
            color = pygame.Color(0)
            color.hsva = (hue, 100, 100, 100)
        else:
            color = (255, 255, 255)

        # Draw the point with varying size and color
        pygame.draw.circle(screen, color, (int(x), int(y)), point_size)


def visualize(screen, particles, flow_field, data, time_elapsed):
    global smoothed_magnitude, previous_magnitude, beat_threshold, beat_detected

    # Calculate the audio signal's magnitude (RMS)
    magnitude = np.sqrt(np.mean(np.square(data)))

    # Smooth the magnitude to prevent abrupt changes
    smoothed_magnitude = (SMOOTHING_FACTOR * magnitude) + ((1 - SMOOTHING_FACTOR) * smoothed_magnitude)

    # Calculate the magnitude difference to detect sudden changes (possible beats)
    magnitude_diff = smoothed_magnitude - previous_magnitude

    # Beat detection logic based on magnitude difference
    if magnitude_diff > beat_threshold * beat_sensitivity:
        beat_detected = True
        beat_threshold = magnitude_diff
    else:
        beat_detected = False
        beat_threshold *= beat_decay
        if beat_threshold < 0.01:
            beat_threshold = 0.01  # Prevent threshold from becoming too low

    previous_magnitude = smoothed_magnitude

    # Map the smoothed magnitude to rotation speed
    rotation_speed = 0.3 + smoothed_magnitude * 100  # Increased base speed and multiplier

    # Calculate the current angle for rotation
    angle_offset = time_elapsed * rotation_speed

    # Clear the screen with a high-contrast background
    screen.fill((0, 0, 0))

    # Update and draw particles
    for particle in particles:
        particle.update(smoothed_magnitude, beat_detected)
        particle.draw(screen)

    # Draw the rotating spiral on top
    draw_rotating_spiral(screen, (WIDTH // 2, HEIGHT // 2), min(WIDTH, HEIGHT) // 2, angle_offset, smoothed_magnitude)

def main():
    screen, clock = initialize_pygame()

    # Initialize flow field
    cols = int(WIDTH / FLOW_FIELD_SCALE) + 1
    rows = int(HEIGHT / FLOW_FIELD_SCALE) + 1
    flow_field = create_flow_field(cols, rows)

    # Create particles
    particles = [Particle(flow_field, cols, rows) for _ in range(PARTICLE_COUNT)]

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
        time_elapsed = time.time() - start_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if len(audio_data) > 0:
            visualize(screen, particles, flow_field, audio_data, time_elapsed)

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
