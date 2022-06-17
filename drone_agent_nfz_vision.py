import torch
import random
import numpy as np
from collections import deque
#from drone_game_simple import DroneGameAI, Direction, Point
from drone_game_nfz_vision import DroneGameAI, Direction, Point
from drone_model_vision import QTrainer,DVision
from helper import plot
from os import environ
import pickle
import numpy as np

environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

MAX_MEMORY = 1000000#100_000
BATCH_SIZE = 1000 #1000
LR = 0.001 #df 0.001
BLOCK_SIZE = 5 # 20

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.75#0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = DVision(1,3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):

        state = [game.nfz_maps + 2*game.dronepositionmap + 3*game.foodmap]
        #print('nfz shape:', np.shape(game.nfz_maps))
        #print('droneposmap shape:', np.shape(game.dronepositionmap))
        #print('foodmap shape:', np.shape(game.foodmap))
        #state = np.stack([game.nfz_maps, game.dronepositionmap, game.foodmap])
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        #states, actions, rewards, next_states,dones = map(np.array, zip(*mini_sample))

        #for state, action, reward, next_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, max_score=1):
        # random moves: tradeoff exploration / exploitation
        #self.epsilon = tradeoff - self.n_games
        self.epsilon = 50*np.exp(-max_score/20)
        final_move = [0,0,0]
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
            print('random move >>>>>', move)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            #print(prediction)
            #move = torch.argmax(prediction).item()
            move = (prediction==torch.max(prediction)).nonzero()[0][1]
            final_move[move] = 1
            print('real move >>>>>',move)

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
    imax = 300

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