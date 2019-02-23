import copy
import matplotlib
matplotlib.use("TkAgg")
import pylab
import numpy as np
from environment import Env
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

EPISODES = 2500


# this is REINFORCE Agent for GridWorld
class ReinforceAgent:
    def __init__(self):
        self.load_model = False
        # actions which agent can do
        self.action_space = [0, 1, 2, 3, 4]
        # get size of state and action
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.optimizer = self.optimizer()
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights('./save_model/reinforce_trained.h5')

    # state is input and probability of each action(policy) is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.summary()
        return model

    # create error function and training function to update policy network
    def optimizer(self):
        action_placeholder = K.placeholder(shape=[None, 5])
        discounted_rewards_placeholder = K.placeholder(shape=[None, ])

        # Calculate cross entropy error function
        action_prob = K.sum(action_placeholder * self.model.output, axis=1)
        cross_entropy = K.log(action_prob) * discounted_rewards_placeholder
        loss = -K.sum(cross_entropy)

        # create training function
        optimizer = Adam(lr=self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        updates_func = K.function(
            [self.model.input, action_placeholder, discounted_rewards_placeholder],
            [],
            updates=updates
        )

        return updates_func

    # get action from policy network
    # numpy.random.choice(a, size=None, replace=True, p=None)
    # a: 배열이면 원래의 데이터, 정수이면 range(a) 명령으로 데이터 생성
    # size: 정수. 샘플링해야 할 개수
    # replace: True or False. True이면 한번 선택한 데이터를 다시 선택 가능
    # p: 배열.각 데이터가 선택될 수 있는 확률
    def get_action(self, state):
        predict_result = self.model.predict(state)
        policy = predict_result[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # calculate discounted rewards
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save states, actions and rewards for an episode
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # update policy neural network
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []


if __name__ == "__main__":
    env = Env()
    agent = ReinforceAgent()

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        # fresh env
        state = env.reset()
        state = np.reshape(state, [1, 15])

        local_step = 0
        while not done:
            global_step += 1
            local_step += 1
            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, 15])
            agent.append_sample(state, action, reward)
            score += reward
            state = copy.deepcopy(next_state)

            if done:
                # update policy neural network for each episode
                agent.train_model()
                scores.append(score)
                episodes.append(e)
                score = round(score, 2)
                print("episode: {0}, score: {1}, local step: {2}, global_step: {3}".format(e, score, local_step, global_step))
                local_step = 0

        if e % 100 == 0:
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/reinforce.png")
            agent.model.save_weights("./save_model/reinforce.h5")
