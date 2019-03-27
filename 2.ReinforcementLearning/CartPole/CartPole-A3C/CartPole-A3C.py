import math
from collections import deque

import gym
import threading
from utils.logger import get_logger
import numpy as np
import tensorflow as tf
import pylab
import time
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, LSTM
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import os
import pickle

print(tf.__version__)
series_size = 1 # MLP에서는 사용하지 않음
feature_size = 4
# x : -0.061586
# θ : -0.75893141
# dx/dt : 0.05793238
# dθ/dt : 1.15547541

action_size = 2

model = "MLP"
#model = "LSTM"
#model = "CNN"

load_model = False  # 훈련할 때
#load_model = True    # 훈련이 끝나고 Play 할 때

MAX_EPISODES = 1000

SUCCESS_CONSECUTIVE_THRESHOLD = 15

global_logger = get_logger("./cartpole_a3c")

class A3C_LSTM:
    def __init__(self):
        self.actor_lr = 0.001
        self.critic_lr = 0.001

        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 10 ** (-8)

        self.discount_factor = .99
        self.hidden1, self.hidden2, self.hidden3 = 64, 32, 16

        self.global_score_list = []
        self.__global_score_list_lock = threading.RLock()

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        if not os.path.exists("./save_model/"):
            os.makedirs("./save_model/")

        if not os.path.exists("./save_graph/"):
            os.makedirs("./save_graph/")

        if not os.path.exists("~/git/auto_trading/web_app/static/img"):
            os.makedirs("~/git/auto_trading/web_app/static/img")

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        self.num_consecutive_success = 0

    def play(self):
        env = gym.make("CartPole-v0")
        state = env.reset()

        self.actor.load_weights("./save_model/bithumb_a3c_actor.h5")
        self.critic.load_weights("./save_model/bithumb_a3c_critic.h5")

        done = False
        step_counter = 0
        reward_sum = 0

        try:
            while not done:
                env.render(mode='rgb_array')
                action = self.get_action(state)
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()

    def build_model(self):
        if model == "MLP":
            input = Input(batch_shape=(None, feature_size), name="state")
            shared_1 = Dense(units=self.hidden1, activation='relu')(input)
            shared_2 = Dense(units=self.hidden2, activation="relu")(shared_1)
        elif model == "LSTM":
            input = Input(batch_shape=(None, series_size, feature_size), name="state")
            shared_1 = LSTM(
                units=self.hidden1,
                input_shape=(series_size, feature_size),  # (타임스텝, 속성)
                activation='relu',
                dropout=0.2,
                return_sequences=True
            )(input)

            shared_2 = LSTM(
                units=self.hidden2,
                activation="relu",
                dropout=0.3,
                return_sequences=False
            )(shared_1)
        else:
            input = Input(batch_shape=(None, series_size, feature_size, 1), name="state")
            shared_1 = Conv2D(
                filters=32,
                kernel_size=(int(math.floor(series_size/4)), 2),
                activation='relu',
                input_shape=(1, series_size, feature_size)
            )(input)

            shared_2 = Conv2D(
                filters=32,
                kernel_size=(int(math.floor(series_size/4)), 2),
                activation='relu',
            )(shared_1)

            shared_2 = Flatten()(shared_2)

        actor_hidden = Dense(self.hidden3, activation='relu', kernel_initializer='glorot_normal')(shared_2)
        actor_hidden_2 = Dropout(rate=0.35)(actor_hidden)
        action_prob = Dense(action_size, activation='softmax', kernel_initializer='glorot_normal')(
            actor_hidden_2)

        value_hidden = Dense(self.hidden3, activation='relu', kernel_initializer='glorot_normal')(shared_2)
        value_hidden_2 = Dropout(rate=0.35)(value_hidden)
        state_value = Dense(1, activation='linear', kernel_initializer='glorot_normal')(value_hidden_2)

        actor = Model(inputs=input, outputs=action_prob)
        critic = Model(inputs=input, outputs=state_value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    def actor_optimizer(self):
        action = K.placeholder(shape=(None, action_size), name="action")
        advantages = K.placeholder(shape=(None,), name="advantages")
        policy = self.actor.output
        action_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(action_prob + 1e-10) * K.stop_gradient(advantages)
        loss = -K.sum(eligibility)

        # policy의 entropy를 더하여 Exploration 행동 확률 추가.
        entropy = K.sum(policy * K.log(policy + 1e-10))

        actor_loss = loss + 0.01 * entropy

        optimizer = Adam(lr=self.actor_lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        l2_norm = K.l2_normalize(self.actor.trainable_weights[-1])

        train = K.function(
            [self.actor.input, action, advantages],
            [actor_loss, loss, policy, action_prob, l2_norm],
            updates=updates
        )

        global_logger.info("action: {0}".format(action))
        global_logger.info("advantages: {0}".format(advantages))
        global_logger.info("policy: {0}".format(policy))
        global_logger.info("action_prob: {0}".format(action_prob))
        global_logger.info("eligibility: {0}".format(eligibility))
        global_logger.info("loss: {0}".format(loss))
        global_logger.info("entropy: {0}".format(entropy))
        global_logger.info("self.actor.trainable_weights: {0}".format(self.actor.trainable_weights))

        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None,), name="discounted_reward")
        value = self.critic.output
        critic_loss = K.mean(K.square(discounted_reward - value))
        optimizer = Adam(lr=self.critic_lr, beta_1=self.beta_1, beta_2=self.beta_2, epsilon=self.epsilon)

        updates = optimizer.get_updates(self.critic.trainable_weights, [], critic_loss)
        l2_norm_0 = K.l2_normalize(self.actor.trainable_weights[0])
        l2_norm_1 = K.l2_normalize(self.actor.trainable_weights[1])
        l2_norm_2 = K.l2_normalize(self.actor.trainable_weights[2])
        l2_norm_3 = K.l2_normalize(self.actor.trainable_weights[3])

        train = K.function(
            [self.critic.input, discounted_reward],
            [value, critic_loss, l2_norm_0, l2_norm_1, l2_norm_2, l2_norm_3],
            updates=updates
        )

        global_logger.info("discounted_reward: {0}".format(discounted_reward))
        global_logger.info("critic ourput value: {0}".format(value))
        global_logger.info("self.critic.trainable_weights: {0}".format(self.critic.trainable_weights))

        return train

    def train(self):
        agents = [
            Agent(idx, self, self.actor, self.critic, self.optimizer, self.discount_factor, self.sess) for idx in range(4)
        ]

        for agent in agents:
            agent.start()

        while True:
            if self.num_consecutive_success >= SUCCESS_CONSECUTIVE_THRESHOLD:
                for agent in agents:
                    agent.running = False
                print("SUCCESS!!!")
                break

            is_anyone_alive = True
            for agent in agents:
                is_anyone_alive = agent.is_alive()

            if not is_anyone_alive:
                break

            time.sleep(1)

    @staticmethod
    def exp_moving_average(values, window):
        """ Numpy implementation of EMA
        """
        if window >= len(values):
            sma = np.mean(np.asarray(values))
            a = [sma] * len(values)
        else:
            weights = np.exp(np.linspace(-1., 0., window))
            weights /= weights.sum()
            a = np.convolve(values, weights, mode='full')[:len(values)]
            a[:window] = a[window]
        return a

    def save_model(self):
        with self.__global_score_list_lock:
            self.actor.save_weights("./save_model/bithumb_a3c_actor.h5")
            self.critic.save_weights("./save_model/bithumb_a3c_critic.h5")

    def append_global_score_list(self, idx, episode, score, mean_score):
        with self.__global_score_list_lock:

            if score >= 195 and self.global_score_list[-1] >= 195:
                self.num_consecutive_success += 1
            else:
                self.num_consecutive_success = 0

            self.global_score_list.append(score)

            global_logger.info("{0:>5}-Episode {1:>3d}: SCORE {2:.6f}, MEAN SCORE {3:.6f}, num_consecutive_success: {4}".format(
                idx,
                episode,
                score,
                mean_score,
                self.num_consecutive_success
            ))

            pylab.clf()
            pylab.plot(range(len(self.global_score_list)), self.global_score_list, 'b')
            pylab.plot(range(len(self.global_score_list)), self.exp_moving_average(self.global_score_list, 10), 'r')
            pylab.savefig("./save_graph/bithumb_a3c_lstm.png")

            with open('./save_graph/global_score_list.pickle', 'wb') as f:
                pickle.dump(self.global_score_list, f)

    def get_action(self, state):
        if model == "MLP":
            policy = self.actor.predict(np.reshape(state, [1, feature_size]))[0]
        elif model == "LSTM":
            policy = self.actor.predict(np.reshape(state, [1, series_size, feature_size]))[0]
        else:
            policy = self.actor.predict(np.reshape(state, [1, series_size, feature_size, 1]))[0]
        return np.random.choice(action_size, 1, p=policy)[0]


class Agent(threading.Thread):
    def __init__(self, idx, global_a3c, actor, critic, optimizer, discount_factor, sess):
        threading.Thread.__init__(self)

        self.state_list = []
        self.action_list = []
        self.reward_list = []
        self.new_state_list = []
        self.done_list = []

        self.sess = sess

        self.global_a3c = global_a3c

        self.idx = idx
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.local_score_list = []
        self.local_logger = get_logger("./cartpole_a3c_" + str(self.idx))
        self.env = gym.make('CartPole-v0')

        self.state_series = deque([], series_size)
        self.new_state_series = deque([], series_size)
        for _ in range(series_size):
            self.state_series.append(np.zeros(shape=(feature_size,)).tolist())
            self.new_state_series.append(np.zeros(shape=(feature_size,)).tolist())

        self.running = True

    # Thread interactive with environment
    def run(self):
        local_episode = 0

        while local_episode < MAX_EPISODES and self.running:
            state = self.env.reset()
            self.state_series.append(state.tolist())

            local_score = 0
            local_step = 0

            while self.running:
                policy, argmax, action = self.get_action(self.state_series)
                new_state, reward, done, info = self.env.step(action)

                # self.local_logger.info("{0} - policy: {1}|{2}, Action: {3} --> State: {3}, Reward: {4}, Done: {5}, Info: {6}".format(
                #     self.idx, policy, argmax, action, new_state, reward, done, info
                # ))

                local_score += reward
                local_step += 1

                self.append_memory(state, action, reward, new_state, done)

                state = new_state

                if local_step % 5 == 0 and self.running:
                    self.train_episode()
                    self.global_a3c.save_model()

                if done and self.running:
                    if len(self.state_list) > 0:
                        self.train_episode()
                        self.global_a3c.save_model()

                    local_episode += 1

                    self.local_score_list.append(local_score)
                    mean_local_score = np.mean(self.local_score_list)

                    self.local_logger.info("{0:>5}-Episode {1:>3d}: SCORE {2:.6f}, MEAN SCORE {3:.6f}".format(
                        self.idx,
                        local_episode,
                        local_score,
                        mean_local_score
                    ))

                    self.global_a3c.append_global_score_list(self.idx, local_episode, local_score, mean_local_score)
                    break

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def get_discount_rewards(self):
        discounted_rewards = np.zeros_like(self.reward_list)

        for t in reversed(range(0, len(self.reward_list))):
            if self.done_list[t]:
                running_add = self.reward_list[t]
            else:
                if model == "MLP":
                    running_add = self.critic.predict(
                        np.reshape(self.new_state_list[t], (1, feature_size))
                    )[0]
                elif model == 'LSTM':
                    running_add = self.critic.predict(
                        np.reshape(self.new_state_list[t], (1, series_size, feature_size))
                    )[0]
                else:
                    running_add = self.critic.predict(
                        np.reshape(self.new_state_list[t], (1, series_size, feature_size, 1))
                    )[0]
                running_add = self.reward_list[t] + self.discount_factor * running_add
            discounted_rewards[t] = running_add
        return discounted_rewards

    def append_memory(self, state, action, reward, new_state, done):
        self.state_series.append(state.tolist())
        self.state_list.append(self.state_series.copy())

        act = np.zeros(action_size)
        act[action] = 1
        self.action_list.append(act)

        self.reward_list.append(reward)

        self.new_state_series.append(new_state.tolist())
        self.new_state_list.append(self.new_state_series.copy())

        self.done_list.append(done)

    def remove_memory(self):
        self.state_list.clear()
        self.action_list.clear()
        self.reward_list.clear()
        self.new_state_list.clear()
        self.done_list.clear()

    # update policy network and value network every episode
    def train_episode(self):
        discounted_rewards = self.get_discount_rewards()

        if model == "MLP":
            values = self.critic.predict(
                np.reshape(self.state_list, (len(self.state_list), feature_size))
            )
        elif model == "LSTM":
            values = self.critic.predict(
                np.reshape(self.state_list, (len(self.state_list), series_size, feature_size))
            )
        else:
            values = self.critic.predict(
                np.reshape(self.state_list, (len(self.state_list), series_size, feature_size, 1))
            )
        values = np.reshape(values, len(values))  # values.shape --> (65,) when len(self.state_list) == 65

        advantages = discounted_rewards - values
        advantages_clipped = np.clip(advantages, -5.0, 5.0)

        if model == "MLP":
            local_input = np.reshape(self.state_list, (len(self.state_list), feature_size))
        elif model == "LSTM":
            local_input = np.reshape(self.state_list, (len(self.state_list), series_size, feature_size))
        else:
            local_input = np.reshape(self.state_list, (len(self.state_list), series_size, feature_size, 1))

        actor_loss, loss, policy, action_prob, l2_norm = self.optimizer[0]([
            local_input,
            np.reshape(self.action_list, (len(self.action_list), action_size)),
            advantages
        ])

        value, critic_loss, l2_norm_0, l2_norm_1, l2_norm_2, l2_norm_3 = self.optimizer[1]([
            local_input,
            discounted_rewards
        ])

        if not np.array_equal(advantages, advantages_clipped):
            self.local_logger.info("{0}: Advantage: Original: {1} - Clipped: {2}".format(self.idx, advantages, advantages_clipped))
            # self.local_logger.info("actor_loss:{0}\nloss:{1}\npolicy:{2}\naction_prob:{3}\nl2_norm:{4}".format(
            #     actor_loss, loss, policy, action_prob, l2_norm
            # ))
            # self.local_logger.info("value:{0}\ncritic_loss:{1}\nl2_norm_0:{2}\nl2_norm_1:{3}\nl2_norm_2:{4}\nl2_norm_3".format(
            #     value, critic_loss, l2_norm_0, l2_norm_1, l2_norm_2, l2_norm_3
            # ))

        self.remove_memory()

    def get_action(self, state):
        state = np.asarray(state)
        if model == "MLP":
            state = np.reshape(state, (1, feature_size))
        elif model == "LSTM":
            state = np.reshape(state, (1, series_size, feature_size))
        else:
            state = np.reshape(state, (1, series_size, feature_size, 1))

        policy = self.actor.predict(state)[0]
        action = np.random.choice(action_size, 1, p=policy)[0]

        return policy, np.argmax(policy), action


if __name__ == "__main__":
    global_a3c = A3C_LSTM()

    if load_model:
        global_a3c.play()
    else:
        global_a3c.train()
