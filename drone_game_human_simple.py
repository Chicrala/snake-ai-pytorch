import pygame
import random
from enum import Enum
from collections import namedtuple

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

BLOCK_SIZE = 5
SPEED = 10

class DroneGame:
    
    def __init__(self, w=931, h=783):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        # init game state.
        #self.direction = Direction.STOP
        self.direction = Direction.LEFT

        # This might be redundant now.
        self.head = Point(665, 360)
        self.snake = [self.head]

        # Getting a fuel.
        self.fuel = 100

        self.score = 0
        self.food = None
        self._place_food()
        
    def _place_food(self):
        x = random.randint(0,len(HOSPITALS)-1)
        self.food = Point(HOSPITALS[x][0], HOSPITALS[x][1])
        if self.food in self.snake:
            self._place_food()

    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
                #elif event.key == pygame.K_s:
                #    self.direction = Direction.STOP

        # 2. move
        self._move(self.direction) # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        game_over = False

        if self._is_collision():
            game_over = True
            return game_over, self.score

        if (self.head.x > MOTHER_BASE[0] - BLOCK_SIZE) and (self.head.x < MOTHER_BASE[0] + BLOCK_SIZE*4)\
                and (self.head.y > MOTHER_BASE[1] - BLOCK_SIZE) and (self.head.y < MOTHER_BASE[1] + BLOCK_SIZE*4):
            self.fuel = 100
        else:
            self.fuel -= .5

        if self._out_of_fuel():
            game_over = True
            return game_over, self.score

        # 4. place new food/refuel or just move
        if self.head == self.food:
            self.fuel = 100
            self.score += 1
            self._place_food()
        #else:
        self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return game_over, self.score
    
    def _is_collision(self):
        # hits boundary
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        
        return False

    def _out_of_fuel(self):
        # Runs out of fuel.
        if self.fuel <= 0:
            return True

        return False
        
    def _update_ui(self):
        self.display.blit(background, (0, 0))

        # Drawing the motherbase.
        pygame.draw.rect(surface=self.display, color=GREEN1,
                         rect=pygame.Rect(MOTHER_BASE[0], MOTHER_BASE[1], BLOCK_SIZE * 4, BLOCK_SIZE * 4))

        text = font.render("Refuel", True, BLACK)
        self.display.blit(text,[MOTHER_BASE[0]-10,MOTHER_BASE[1]-25])

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score)+" Fuel: " + str(self.fuel), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, direction):
        x = self.head.x
        y = self.head.y

        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        else:
            pass
            
        self.head = Point(x, y)
            

if __name__ == '__main__':
    game = DroneGame()
    
    # game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over == True:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()