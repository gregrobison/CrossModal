import pygame
import sys
import multiprocessing
import time

# Import all the visualization scripts
import Hexacosmatrix
import Pentaflow
import Mobiowave
import Torusation
import Spirhythm
import Frequencsphere
import Soundgularity
import Hyperdrive
import Pulsaract
import Chromascope
import Wavebrane
import Droplet
import Neucleon
import Beatoids
import Peakistory

# List of visualizations to choose from
visualizations = [
    ("Frequencsphere", Frequencsphere.main),
    ("Hexacosmatrix", Hexacosmatrix.main),
    ("Pulsaract", Pulsaract.main),
    ("Soundgularity", Soundgularity.main),
    ("Beatoids", Beatoids.main),
    ("Hyperdrive", Hyperdrive.main),
    ("Torusation", Torusation.main),
    ("Peakistory", Peakistory.main),
    ("Wavebrane", Wavebrane.main),
    ("Mobiowave", Mobiowave.main),
    ("Pentaflow", Pentaflow.main),
    ("Spirhythm", Spirhythm.main),
    ("Chromascope", Chromascope.main),
    ("Droplet", Droplet.main),
    ("Neucleon", Neucleon.main)
]

def display_menu(screen, font):
    selected_index = 0
    while True:
        screen.fill((0, 0, 0))
        title = font.render("Select a Visualization", True, (255, 255, 255))
        screen.blit(title, (250, 50))

        for i, (name, _) in enumerate(visualizations):
            color = (0, 191, 255) if i == selected_index else (150, 150, 150)  # Changed this line
            text = font.render(name, True, color)
            screen.blit(text, (300, 150 + i * 40))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_index = (selected_index - 1) % len(visualizations)
                elif event.key == pygame.K_DOWN:
                    selected_index = (selected_index + 1) % len(visualizations)
                elif event.key == pygame.K_RETURN:
                    # Run the selected visualization
                    run_visualization(screen, font, selected_index)
                    return

def run_visualization(screen, font, index):
    visualization_index = index
    clock = pygame.time.Clock()
    running = True

    while running:
        screen.fill((0, 0, 0))
        visualization_name, visualization_function = visualizations[visualization_index]
        title_font = pygame.font.Font(None, 36)
        title = title_font.render(f"Running {visualization_name}", True, (255, 255, 255))
        screen.blit(title, (250, 50))
        pygame.display.flip()

        # Start the visualization in a separate process
        process = multiprocessing.Process(target=visualization_function)
        process.start()

        # Check for events while the visualization is running
        while process.is_alive():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    process.terminate()
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_ESCAPE):
                        process.terminate()
                        if event.key == pygame.K_LEFT:
                            visualization_index = (visualization_index - 1) % len(visualizations)
                        elif event.key == pygame.K_RIGHT:
                            visualization_index = (visualization_index + 1) % len(visualizations)
                        elif event.key == pygame.K_ESCAPE:
                            running = False
                        break

            # Update the screen and wait a bit
            pygame.display.flip()
            time.sleep(0.1)

        # The process has ended (either naturally or was terminated)
        if not running:
            break

        # Ask user if they want to restart the visualization or return to menu
        screen.fill((0, 0, 0))
        restart_text = font.render("Press R to restart, or any other key for menu", True, (255, 255, 255))
        screen.blit(restart_text, (50, 400))
        pygame.display.flip()

        waiting_for_input = True
        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        waiting_for_input = False
                    else:
                        running = False
                        waiting_for_input = False

        clock.tick(30)  # Limit frame rate to 30 frames per second

    return  # Return to menu

def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("CrossModal Menu")
    font = pygame.font.Font(None, 36)

    while True:
        display_menu(screen, font)

if __name__ == "__main__":
    multiprocessing.freeze_support()  # This is necessary for Windows
    main()
