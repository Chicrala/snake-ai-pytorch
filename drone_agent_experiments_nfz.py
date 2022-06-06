import torch
import random
import numpy as np
from collections import deque
#from drone_game_simple import DroneGameAI, Direction, Point
from drone_game_nfz import DroneGameAI, Direction, Point
from drone_model_experiments import QTrainer,Deeper_Linear_QNet #Deeeeper_Linear_QNet ##,Linear_QNet
from helper import plot
from os import environ
import pickle
environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

MAX_MEMORY = 1000000#100_000
BATCH_SIZE = 5000 #1000
LR = 0.001 #df 0.001
BLOCK_SIZE = 5 # 20

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.7#0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        #self.model = Linear_QNet(11,256,3)
        self.model = Deeper_Linear_QNet(9,3)
        #self.model = Deeeeper_Linear_QNet(9,3)
        #self.model = LinearRelu(12,3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
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

        #return np.array(state, dtype=int)
        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state,max_score=1):
        # random moves: tradeoff exploration / exploitation
        #self.epsilon = tradeoff - self.n_games
        self.epsilon = 50*np.exp(-max_score/20)
        final_move = [0,0,0]
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    plot_rewards = []
    #plot_mean_rewards = []
    #total_rewards = 0
    record_reward = -300

    # Counterhow
    i = 0
    imax = 150

    agent = Agent()
    game = DroneGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old,record)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            plot_rewards.append(game.acc_reward)
            agent.n_games += 1
            agent.train_long_memory()

            if score > record and game.acc_reward > record_reward:
                record = score
                record_reward = game.acc_reward
                # agent.model.save()
                pickle.dump(agent.model, open("./model/nfzmodel.sav", "wb"))


            elif score > record:
                record = score
                # agent.model.save()
                pickle.dump(agent.model, open("./model/nfzmodel.sav", "wb"))

            elif game.acc_reward > record_reward:
                record_reward = game.acc_reward
                # agent.model.save()
                pickle.dump(agent.model, open("./model/nfzmodel.sav", "wb"))

            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Reward:', game.acc_reward, 'Record:', record_reward)

            game.reset()

            # Calculating the mean score.
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            #plot_rewards.append(game.acc_reward)
            # total_rewards += game.acc_reward
            # mean_reward = total_rewards / agent.n_games
            # plot_mean_rewards.append(mean_reward)

            # Plotting.
            plot(plot_scores, plot_mean_scores)

            i += 1
            if i > imax:
                np.save(f'scores_imax{imax}.npy', plot_scores)
                np.save(f'rewards_imax{imax}.npy', plot_rewards)
                break


if __name__ == '__main__':
    train()