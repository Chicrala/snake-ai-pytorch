import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from drone_map import MovingMapper, NoFlyMapper
pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    #STOP = 0


Point = namedtuple('Point', 'x, y')
#Rectangle = namedtuple('Rectangle', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
GREEN1 = (32,139,57)

background = pygame.image.load('background2.png')
background = pygame.transform.scale(background,(931,783))

# The hospital coordinates.
#HOSPITALS = [[63,321], [311,344], [149,500]]
HOSPITALS = [[65,320], [310,345], [140,500],[665, 675],[740,760], [845,80]]

#MOTHER_BASE = [[655,360],[680,380]]
MOTHER_BASE = [665,360]

nfz_heyshan = [575,750]
nfz_cark = [500,400]
nfz_barrow = [70,520]

BLOCK_SIZE = 5
SPEED = 80
DG_RADIUS = 15

class DroneGameAI:
    
    def __init__(self, w=931, h=783):
        self.w = w
        self.h = h
        #self.nfz_maps = NoFlyMapper().no_fly_zone
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Drone deliveries')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        # This might be redundant now.
        self.head = Point(200, 400)
        self.snake = [self.head]
        self.fuel = 100
        self.score = 0
        #self.dronepositionmap = MovingMapper([self.h,self.w],self.head.y,self.head.x,BLOCK_SIZE).mapped
        #self.foodmap = MovingMapper([self.h,self.w],self.head.y,self.head.x,BLOCK_SIZE).mapped
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.distance_to_food = None # Here we add the initial distance to the food
        self._distance_to_food()
        self.acc_reward = 0
        self.danger=False
        self.distance_to_heyshan = np.sqrt((self.head.x - nfz_heyshan[0]) ** 2 + (self.head.y - nfz_heyshan[1]) ** 2)
        self.distance_to_cark = np.sqrt((self.head.x - nfz_cark[0]) ** 2 + (self.head.y - nfz_cark[1]) ** 2)
        self.distance_to_barrow = np.sqrt((self.head.x - nfz_barrow[0]) ** 2 + (self.head.y - nfz_barrow[1]) ** 2)

    def _place_food(self):
        x = random.randint(0,len(HOSPITALS)-1)
        self.food = Point(HOSPITALS[x][0], HOSPITALS[x][1])
        if self.food in self.snake:
            self._place_food()

    def _distance_to_food(self):
        distance = np.sqrt((self.food.x-self.head.x)**2+(self.food.y-self.head.y)**2)
        if np.isnan(distance):
            print(self.food.x,self.head.x,self.food.y,self.head.y)
            # Set as maximum possible distance.
            self.distance_to_food = np.sqrt(self.w**2+self.h**2)
        else:
            self.distance_to_food = distance

    def play_step(self, action):
        self.frame_iteration +=1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action) # update the head
        # update maps
        self.dronepositionmap = MovingMapper([self.h, self.w], self.head.y, self.head.x, BLOCK_SIZE).mapped
        self.foodmap = MovingMapper([self.h, self.w], self.head.y, self.head.x, BLOCK_SIZE).mapped
        self.snake.insert(0, self.head)

        self.distance_to_heyshan = np.sqrt((self.head.x - nfz_heyshan[0]) ** 2 + (self.head.y - nfz_heyshan[1]) ** 2)
        self.distance_to_cark = np.sqrt((self.head.x - nfz_cark[0]) ** 2 + (self.head.y - nfz_cark[1]) ** 2)
        self.distance_to_barrow = np.sqrt((self.head.x - nfz_barrow[0]) ** 2 + (self.head.y - nfz_barrow[1]) ** 2)

        # Check if it's flying in the danger zone (plays top gun's danger zone).
        # Was not in danger, is not in danger.
        if not self.danger and self.danger_zone():
            self.danger = True
        # Was in danger, is not in danger.
        if self.danger and not self.danger_zone():
            self.danger = False
            reward = 2
            self.acc_reward += reward
            # Reward



        # 3. check if a collision happened.
        reward = 0
        game_over = False

        # if self.is_collision() or self.frame_iteration > 1000*len(self.snake):
        if self.is_collision():
            game_over = True
            reward = -100
            self.acc_reward += reward
            return reward, game_over, self.score

        #cond = (self.head.x > MOTHER_BASE[0] - BLOCK_SIZE) and (self.head.x < MOTHER_BASE[0] + BLOCK_SIZE*4)\
        #        and (self.head.y > MOTHER_BASE[1] - BLOCK_SIZE) and (self.head.y < MOTHER_BASE[1] + BLOCK_SIZE*4)

        #self.fuel = 100 if cond else self.fuel-0.1
        # Check if drone is in the MOTHER_BASE and refuel if so.
        #if (self.head.x > MOTHER_BASE[0] - BLOCK_SIZE) and (self.head.x < MOTHER_BASE[0] + BLOCK_SIZE*4)\
        #        and (self.head.y > MOTHER_BASE[1] - BLOCK_SIZE) and (self.head.y < MOTHER_BASE[1] + BLOCK_SIZE*4):
        #    self.fuel = 100
        #else:
        #    self.fuel -= .25
        self.fuel -= .25
        if self._out_of_fuel():
            game_over = True
            reward = -10
            self.acc_reward += reward
            return reward, game_over, self.score

        # 4. place new food/refuel or just move
        if self.head == self.food:
            reward = 10 + self.fuel
            self.acc_reward += reward
            self.fuel = 100
            self.score += 1
            self._place_food()
            # Recalculate distance?
            self._distance_to_food()

        else:
            # Calculate distance to food.
            #self.distance = np.sqrt((self.food.x - self.head.x)**2 + (self.food.y - self.head.y)**2)

            distance = np.sqrt((self.food.x - self.head.x)**2 + (self.food.y - self.head.y)**2)
            if not np.isnan(distance):
                # Didn't hit food but walked in the wrong direction.
                if distance > self.distance_to_food:
                    #reward = -1
                    #self.acc_reward += reward
                    self.distance_to_food = distance
                    #pass
                # Didn't hit food but walked in the right direction.
                if distance < self.distance_to_food:
                    reward = 1
                    self.acc_reward += reward
                    self.distance_to_food = distance

            else:
                print(self.food.x, self.head.x, self.food.y, self.head.y)

        if self.score > 150:
            game_over = True
            reward = 1000
            self.acc_reward += reward
            return reward, game_over, self.score

        # Trim the snake!
        self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def danger_zone(self):
        # hits boundary
        conditions = [
            self.head.x+DG_RADIUS > self.w - BLOCK_SIZE,
            self.head.x+DG_RADIUS < 0,
            self.head.y+DG_RADIUS > self.h - BLOCK_SIZE,
            self.head.y < 0,
            np.sqrt((self.head.x - nfz_heyshan[0]) ** 2 + (self.head.y - nfz_heyshan[1]) ** 2)-DG_RADIUS < 40,
            np.sqrt((self.head.x - nfz_cark[0]) ** 2 + (self.head.y - nfz_cark[1]) ** 2)-DG_RADIUS < 80,
            np.sqrt((self.head.x - nfz_barrow[0]) ** 2 + (self.head.y - nfz_barrow[1]) ** 2)-DG_RADIUS < 60
        ]

        if any(conditions):
            return True
        else:
            return False

    def is_collision(self, pt=None):
        # hits boundary
        conditions = [
            self.head.x > self.w - BLOCK_SIZE,
            self.head.x < 0, self.head.y > self.h - BLOCK_SIZE,
            self.head.y < 0,
            np.sqrt((self.head.x - nfz_heyshan[0]) ** 2 + (self.head.y - nfz_heyshan[1]) ** 2) < 40,
            np.sqrt((self.head.x - nfz_cark[0]) ** 2 + (self.head.y - nfz_cark[1]) ** 2) < 80,
            np.sqrt((self.head.x - nfz_barrow[0]) ** 2 + (self.head.y - nfz_barrow[1]) ** 2) < 60
        ]

        if any(conditions):
            return True
        else:
            return False

    def _out_of_fuel(self):
        # Runs out of fuel.
        return True if self.fuel <= 0 else False
        
    def _update_ui(self):
        self.display.blit(background, (0, 0))

        # Drawing the motherbase.
        pygame.draw.rect(surface=self.display, color=GREEN1,
                         rect=pygame.Rect(MOTHER_BASE[0], MOTHER_BASE[1], BLOCK_SIZE * 4, BLOCK_SIZE * 4))

        # Drawing the refueling text.
        text = font.render("Refuel", True, BLACK)
        self.display.blit(text,[MOTHER_BASE[0]-10,MOTHER_BASE[1]-25])

        pygame.draw.circle(surface=self.display, color=RED, center=nfz_heyshan, radius=40, width=1)
        pygame.draw.circle(surface=self.display, color=RED, center=nfz_cark, radius=80, width=1)
        pygame.draw.circle(surface=self.display, color=RED, center=nfz_barrow, radius=60, width=1)

        pygame.draw.circle(surface=self.display, color=WHITE,
                           center=[self.head.x+BLOCK_SIZE/2,self.head.y+BLOCK_SIZE/2],
                           radius=DG_RADIUS, width=1)

        text = font.render("Danger", True, BLACK)
        self.display.blit(text, [0, 30])
        if self.danger:
            pygame.draw.circle(surface=self.display, color=RED,
                               center=[100, 50],
                               radius=DG_RADIUS)
        else:
            pygame.draw.circle(surface=self.display, color=RED,
                               center=[100, 50],
                               radius=DG_RADIUS, width=1)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render(f"Score: {self.score} Fuel: {self.fuel:.2f} Distance: {self.distance_to_food:.2f} Reward: {self.acc_reward:.2f}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
