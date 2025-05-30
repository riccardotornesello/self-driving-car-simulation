import pygame
import math
from neat.nn import FeedForwardNetwork

from constants import screen_width, screen_height
from car_math import acceleration_from_velocity


class Car:
    def __init__(self, net: FeedForwardNetwork):
        # TODO: reduce car size

        self.surface = pygame.image.load("assets/car.png")
        self.surface = pygame.transform.scale(self.surface, (100, 100))
        self.rotate_surface = self.surface
        self.pos = [700, 650]
        self.center = [self.pos[0] + 50, self.pos[1] + 50]
        self.radars = []
        self.radars_for_draw = []
        self.is_alive = True
        self.distance = 0
        self.time_spent = 0
        self.net = net

        self.angle = 0
        self.speed = 0

        self.ticks_stopped = 0
        self.max_ticks_stopped = 20

        self.ticks_per_second = 10

        self.vmax = 40
        self.acc = 0.1
        self.brake = 0.2

    def update_speed(self, throttle):
        # TODO: use curves to calculate acceleration

        vinc = 0

        if throttle > 0:
            vinc = acceleration_from_velocity(v=self.speed, vmax=self.vmax, k=self.acc)
            vinc = vinc * throttle / self.ticks_per_second

        elif throttle < 0:
            vinc = -self.speed * self.brake

        self.speed = max(0, self.speed + vinc)

    def update_angle(self, steering):
        # TODO: set max angle based on speed

        max_steering = 10

        self.angle += steering * max_steering

    def draw(self, screen):
        screen.blit(self.rotate_surface, self.pos)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for r in self.radars:
            pos, dist = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def check_collision(self, map):
        self.is_alive = True
        for p in self.four_points:
            if map.get_at((int(p[0]), int(p[1]))) == (255, 255, 255, 255):
                self.is_alive = False
                break

    def check_radar(self, degree, map):
        len = 0
        x = int(
            self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len
        )
        y = int(
            self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len
        )

        while not map.get_at((x, y)) == (255, 255, 255, 255) and len < 300:
            len = len + 1
            x = int(
                self.center[0]
                + math.cos(math.radians(360 - (self.angle + degree))) * len
            )
            y = int(
                self.center[1]
                + math.sin(math.radians(360 - (self.angle + degree))) * len
            )

        dist = int(
            math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2))
        )
        self.radars.append([(x, y), dist])

    def update(self, map):
        # check position
        self.rotate_surface = self.rot_center(self.surface, self.angle)
        self.pos[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        if self.pos[0] < 20:
            self.pos[0] = 20
        elif self.pos[0] > screen_width - 120:
            self.pos[0] = screen_width - 120

        self.distance += self.speed
        self.time_spent += 1
        self.pos[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        if self.pos[1] < 20:
            self.pos[1] = 20
        elif self.pos[1] > screen_height - 120:
            self.pos[1] = screen_height - 120

        # caculate 4 collision points
        self.center = [int(self.pos[0]) + 50, int(self.pos[1]) + 50]
        len = 40
        left_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * len,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * len,
        ]
        right_top = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * len,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * len,
        ]
        left_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * len,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * len,
        ]
        right_bottom = [
            self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * len,
            self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * len,
        ]
        self.four_points = [left_top, right_top, left_bottom, right_bottom]

        self.check_collision(map)
        self.radars.clear()
        for d in range(-90, 120, 45):
            self.check_radar(d, map)

        # check speed
        if self.speed < 1:
            self.ticks_stopped += 1

        if self.ticks_stopped > self.max_ticks_stopped:
            self.is_alive = False

    def get_data(self):
        radars = self.radars
        ret = [0, 0, 0, 0, 0, 0]
        for i, r in enumerate(radars):
            ret[i] = int(r[1] / 30)
        ret[5] = self.speed / self.vmax

        return ret

    def get_alive(self):
        return self.is_alive

    def get_reward(self):
        return self.distance / 50.0

    def rot_center(self, image, angle):
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image

    def activate_net(self):
        # TODO: prevent frequent steering changes

        output = self.net.activate(self.get_data())
        steering = output[0]
        throttle = output[1]

        self.update_speed(throttle)
        self.update_angle(steering)
