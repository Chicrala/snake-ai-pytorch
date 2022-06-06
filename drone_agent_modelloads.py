import torch
import numpy as np
#from drone_game_simple import DroneGameAI, Direction, Point
from drone_game_nfz import DroneGameAI, Direction, Point

import pickle


class Agent:

    def __init__(self):
        self.n_games = 0
        #self.model = pickle.load(open("./model/newmodel.sav", "rb"))
        self.model = pickle.load(open("./model/ggmodel.sav", "rb"))


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        #dir_s = game.direction == Direction.STOP

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #game.fuel,
            # Food location 
            #game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            #game.food.y > game.head.y,  # food down
            ]

        return np.array(state, dtype=float)

    def get_action(self, state, tradeoff=120):
        # random moves: tradeoff exploration / exploitation
        final_move = [0,0,0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move[move] = 1

        return final_move


def run():

    agent = Agent()
    game = DroneGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        if done:

            print('Game', agent.n_games, 'Score', score, 'Reward:', game.acc_reward)

            game.reset()


if __name__ == '__main__':
    run()