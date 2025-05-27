import sys

import pygame
import neat

from car import Car
from constants import screen_width, screen_height


class Game:
    def __init__(self):
        pygame.init()

        self.generation = 0
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.clock = pygame.time.Clock()
        self.map = pygame.image.load("assets/map.png")

        self.draw()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

    def draw(self, cars=[]):
        self.screen.blit(self.map, (0, 0))

        generation_font = pygame.font.SysFont("Arial", 70)
        generation_text = generation_font.render(
            "Generation : " + str(self.generation), True, (255, 255, 0)
        )
        generation_text_rect = generation_text.get_rect()
        generation_text_rect.center = (screen_width / 2, 100)
        self.screen.blit(generation_text, generation_text_rect)

        remain_cars = 0
        for car in cars:
            if car.get_alive():
                remain_cars += 1
                car.draw(self.screen)

        remain_font = pygame.font.SysFont("Arial", 30)
        remain_text = remain_font.render(
            "remain cars : " + str(remain_cars), True, (0, 0, 0)
        )
        remain_text_rect = remain_text.get_rect()
        remain_text_rect.center = (screen_width / 2, 200)
        self.screen.blit(remain_text, remain_text_rect)

        pygame.display.flip()
        self.clock.tick(0)

    def run_car(self, genomes, config):
        # Generate nets and cars
        cars = []
        for id, g in genomes:
            g.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(g, config)
            cars.append(Car(net))

        # Main loop
        self.generation += 1
        while True:
            self.handle_events()

            # Input my data and get result from network
            for car in cars:
                car.activate_net()

            # Update car and fitness
            remain_cars = 0
            for i, car in enumerate(cars):
                if car.get_alive():
                    remain_cars += 1
                    car.update(self.map)
                    genomes[i][1].fitness += car.get_reward()

            # Check
            if remain_cars == 0:
                break

            # Drawing
            self.draw(cars)


if __name__ == "__main__":
    # Set NEAT configuration file
    config_path = "./config.txt"
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Create core evolution algorithm class
    p = neat.Population(config)

    # Add reporter for fancy statistical result
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Init the track
    game = Game()

    # Run NEAT
    p.run(game.run_car, 1000)
