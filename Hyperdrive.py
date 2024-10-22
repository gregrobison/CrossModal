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
NUM_STARS = 5000
SMOOTHING_FACTOR = 0.3

# Global variables
audio_data = np.zeros(BLOCK_SIZE)
smoothed_magnitude = 0

class Star:
    def __init__(self):
        self.reset()
    
    def reset(self):
        # Initialize star at a random position
        self.x = random.uniform(-WIDTH / 2, WIDTH / 2)
        self.y = random.uniform(-HEIGHT / 2, HEIGHT / 2)
        self.z = random.uniform(0, WIDTH)
        self.pz = self.z
        self.color = [255, 255, 255]
    
    def update(self, speed):
        # Move the star towards the viewer
        self.z -= speed
        if self.z < 1:
            self.reset()
            self.pz = self.z
    
    def draw(self, screen):
        # Map 3D coordinates to 2D screen positions
        sx = int((self.x / self.z) * (WIDTH / 2) + (WIDTH / 2))
        sy = int((self.y / self.z) * (HEIGHT / 2) + (HEIGHT / 2))
        
        px = int((self.x / self.pz) * (WIDTH / 2) + (WIDTH / 2))
        py = int((self.y / self.pz) * (HEIGHT / 2) + (HEIGHT / 2))
        
        self.pz = self.z
        
        # Draw the star as a line from previous to current position
        if 0 <= sx < WIDTH and 0 <= sy < HEIGHT:
            pygame.draw.line(screen, self.color, (px, py), (sx, sy))

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
    pygame.display.set_caption("Hyperdrive")
    clock = pygame.time.Clock()
    print("Pygame initialized successfully")
    return screen, clock

def visualize(screen, stars, data):
    global smoothed_magnitude
    
    # Calculate the audio signal's magnitude
    magnitude = np.linalg.norm(data) / len(data)
    
    # Smooth the magnitude to prevent abrupt changes
    smoothed_magnitude = (SMOOTHING_FACTOR * magnitude) + ((1 - SMOOTHING_FACTOR) * smoothed_magnitude)
    
    # Map the smoothed magnitude to star speed
    speed = 0.9 + smoothed_magnitude * 100000  # Adjust the multiplier for desired sensitivity
    
    # Map magnitude to color shift for psychedelic effect
    color_shift = int(smoothed_magnitude * 360)
    
    # Clear the screen with a black background
    screen.fill((0, 0, 0))
    
    # Update and draw each star
    for star in stars:
        star.update(speed)
        
        # Update star color based on audio input
        hue = (color_shift + (star.z * 0.1)) % 360  # Vary hue with z to add depth
        color = pygame.Color(0)
        color.hsva = (hue, 100, 100, 100)
        star.color = color
        
        star.draw(screen)

def main():
    screen, clock = initialize_pygame()
    stars = [Star() for _ in range(NUM_STARS)]
    
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
            visualize(screen, stars, audio_data)
        
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
