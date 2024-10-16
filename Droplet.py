import numpy as np
import pygame
import sounddevice as sd
import time
import random
import string

# Audio configuration
SAMPLE_RATE = 44100
BLOCK_SIZE = 2048
DEVICE = None  # Use the default input device
CHANNELS = 1   # Mono input for simplicity

# Pygame configuration
WIDTH = 800
HEIGHT = 600
FPS = 60

# Visualization configuration
FONT_SIZE = 32
SMOOTHING_FACTOR = 0.3

# Global variables
audio_data = np.zeros(BLOCK_SIZE)
smoothed_magnitudes = np.zeros(BLOCK_SIZE // 2)

# Character set for the falling code
CHARACTERS = string.ascii_letters + string.digits + string.punctuation

class Symbol:
    def __init__(self, x, y, speed, font, color):
        self.x = x
        self.y = y
        self.value = random.choice(CHARACTERS)
        self.speed = speed
        self.font = font
        self.color = color
        self.switch_interval = random.randint(5, 20)

    def draw(self, screen):
        char_surface = self.font.render(self.value, True, self.color)
        screen.blit(char_surface, (self.x, self.y))

    def update(self):
        self.y = (self.y + self.speed) % HEIGHT
        if random.randint(0, self.switch_interval) == 0:
            self.value = random.choice(CHARACTERS)

class CodeRain:
    def __init__(self, num_columns, font_size):
        self.num_columns = num_columns
        self.font_size = font_size
        self.font = pygame.font.SysFont('Consolas', self.font_size)
        self.symbols = []
        self.create_symbols()

    def create_symbols(self):
        for i in range(self.num_columns):
            x = i * self.font_size
            y = random.randint(-HEIGHT, 0)
            speed = random.uniform(2, 5)
            color = (0, 255, 70)
            symbol = Symbol(x, y, speed, self.font, color)
            self.symbols.append(symbol)

    def draw(self, screen):
        for symbol in self.symbols:
            symbol.draw(screen)

    def update(self):
        for symbol in self.symbols:
            symbol.update()

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
    pygame.display.set_caption("Droplet")
    clock = pygame.time.Clock()
    print("Pygame initialized successfully")
    return screen, clock

def visualize(screen, code_rain, data):
    global smoothed_magnitudes

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

    # Map magnitudes to brightness and speed
    avg_magnitude = np.mean(smoothed_magnitudes)
    brightness = int(min(255, avg_magnitude * 10000))
    speed_factor = avg_magnitude * 200 + 1

    # Clear screen with semi-transparent overlay to create trail effect
    overlay = pygame.Surface((WIDTH, HEIGHT), flags=pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 25))  # Last value is alpha
    screen.blit(overlay, (0, 0))

    # Update and draw symbols
    for symbol in code_rain.symbols:
        # Adjust speed and color based on audio
        symbol.speed = random.uniform(2, 5) * speed_factor
        symbol.color = (0, brightness, 70)
        symbol.update()
        symbol.draw(screen)

def main():
    screen, clock = initialize_pygame()
    num_columns = WIDTH // FONT_SIZE
    code_rain = CodeRain(num_columns, FONT_SIZE)

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
            visualize(screen, code_rain, audio_data)

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
