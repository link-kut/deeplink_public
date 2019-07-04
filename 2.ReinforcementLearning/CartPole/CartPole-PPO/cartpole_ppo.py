# Initial framework taken from https://github.com/OctThe16th/PPO-Keras/blob/master/Main.py

import numpy as np
import gym
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

print(tf.__version__)



env = gym.make(ENV)
CONTINUOUS = False
# num_states = env.observation_space.shape[0]

LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
NOISE = 1.0  # Exploration noise

GAMMA = 0.99

BUFFER_SIZE = 256
BATCH_SIZE = 64
NUM_ACTIONS = 2
NUM_STATE = 4
HIDDEN_SIZE = 128
NUM_LAYERS = 2
ENTROPY_LOSS = 1e-3
LR = 1e-4  # Lower lr stabilises training greatly

'''def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new'''


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob / (old_prob + 1e-10)
        return -K.mean(
            K.minimum(
                r * advantage,
                K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage
            ) + ENTROPY_LOSS * (prob * K.log(prob + 1e-10))
        )
    return loss


class Agent:
    def __init__(self):
        self.critic = self.build_critic()
        self.actor = self.build_actor()

        self.env = gym.make(ENV)
        print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space')
        self.episode = 0
        self.observation = self.env.reset()
        self.val = False
        self.reward = []
        self.reward_over_time = []
        self.name = self.get_name()
        self.scores = []
        self.episode_reward = 0

    def get_name(self):
        name = 'AllRuns/'
        name += 'discrete/'
        name += ENV
        return name

    def build_actor(self):
        state_input = Input(shape=(NUM_STATE,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))

        x = Dense(units=HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_actions = Dense(units=NUM_ACTIONS, activation='softmax', name='output')(x)

        model = Model(
            inputs=[state_input, advantage, old_prediction],
            outputs=[out_actions],
            name="actor_model"
        )
        model.compile(
            optimizer=Adam(lr=LR),
            loss=[proximal_policy_optimization_loss(advantage=advantage, old_prediction=old_prediction)]
        )
        model.summary()

        return model

    def build_critic(self):
        state_input = Input(shape=(NUM_STATE,))
        x = Dense(units=HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(units=HIDDEN_SIZE, activation='tanh')(x)

        out_value = Dense(units=1)(x)

        model = Model(
            inputs=[state_input],
            outputs=[out_value],
            name="critic_model"
        )
        model.compile(
            optimizer=Adam(lr=LR),
            loss='mse'
        )

        model.summary()

        return model

    def reset_env(self):
        self.episode += 1
        if self.episode % 100 == 0:
            self.val = True
        else:
            self.val = False
        self.observation = self.env.reset()
        self.reward = []
        self.episode_reward = 0

    def get_action(self):
        DUMMY_VALUE = np.zeros((1, 1))
        DUMMY_ACTION = np.zeros((1, NUM_ACTIONS))

        p = self.actor.predict([self.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_ACTION])

        if self.val is False:
            action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0]))
        else:
            action = np.argmax(p[0])
        action_matrix = np.zeros(NUM_ACTIONS)
        action_matrix[action] = 1
        return action, action_matrix, p

    def transform_reward(self):
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA

    def get_batch(self):
        batch = [[], [], [], []]

        tmp_batch = [[], [], []]

        while len(batch[0]) < BUFFER_SIZE:
            action, action_matrix, actor_p = self.get_action()
            observation, reward, done, info = self.env.step(action)

            self.reward.append(reward)
            self.episode_reward = self.episode_reward + reward

            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(actor_p)
            self.observation = observation

            if done:
                self.transform_reward()
                if self.val is False:
                    for i in range(len(tmp_batch[0])):
                        obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                        r = self.reward[i]
                        batch[0].append(obs)
                        batch[1].append(action)
                        batch[2].append(pred)
                        batch[3].append(r)
                tmp_batch = [[], [], []]
                #print("EPISODE REWARD     ", self.episode_reward)
                self.scores.append(self.episode_reward)
                self.reset_env()

        obs = np.array(batch[0])
        action = np.array(batch[1])
        pred = np.array(batch[2])
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))

        reward = np.reshape(np.array(batch[3]), (len(batch[3]), 1))

        return obs[:BUFFER_SIZE], action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[:BUFFER_SIZE]

    def run(self):

        total_episodes = 100000
        epochs = 10

        while self.episode < total_episodes:
            if len(self.scores) > 1:
                print("EPISODE    ", self.episode, self.scores[-1])

            obs, action, pred, reward = self.get_batch()
            old_prediction = pred
            pred_values = self.critic.predict(obs)

            advantage = reward - pred_values
            # advantage = (advantage - advantage.mean()) / advantage.std()

            actor_loss = self.actor.fit(
                x=[obs, advantage, old_prediction],
                y=[action],
                batch_size=BATCH_SIZE,
                shuffle=True,
                epochs=epochs,
                verbose=0
            )

            critic_loss = self.critic.fit(
                x=[obs],
                y=[reward],
                batch_size=BATCH_SIZE,
                shuffle=True,
                epochs=epochs,
                verbose=0
            )

            if self.episode % 10 == 0:
                print('(episode, score) = ' + str((self.episode, self.episode_reward)))

            # Solved condition
            if len(self.scores) >= 110:
                if np.mean(self.scores[-100:]) >= 195.0:
                    print(' \ Solved after ' + str(self.episode - 100) + ' episodes')
                    break
        plt.plot(self.scores)


if __name__ == '__main__':
    ag = Agent()
    ag.run()