import threading
from collections import deque
from utils.logger import get_logger
import gym
import numpy as np


class Worker(threading.Thread):
    def __init__(self, idx, global_a3c, actor, critic, optimizer, discount_factor, sess, series_size, feature_size, action_size, max_episodes, model_type):
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

        self.series_size = series_size
        self.feature_size = feature_size
        self.action_size = action_size
        self.max_episodes = max_episodes
        self.model_type = model_type

        self.state_series = deque([], self.series_size)
        self.new_state_series = deque([], self.series_size)
        for _ in range(series_size):
            self.state_series.append(np.zeros(shape=(self.feature_size,)).tolist())
            self.new_state_series.append(np.zeros(shape=(self.feature_size,)).tolist())

        self.running = True

    # Thread interactive with environment
    def run(self):
        local_episode = 0

        while local_episode < self.max_episodes and self.running:
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
                    actor_loss, critic_loss = self.train_episode()
                    self.global_a3c.save_model()

                if done and self.running:
                    if len(self.state_list) > 0:
                        actor_loss, critic_loss = self.train_episode()
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

                    self.global_a3c.append_global_score_list(self.idx, local_episode, local_score, actor_loss, critic_loss)
                    break

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    # def get_discount_rewards(self):
    #     discounted_rewards = np.zeros_like(self.reward_list)
    #
    #     for t in reversed(range(0, len(self.reward_list))):
    #         if self.done_list[t]:
    #             running_add = self.reward_list[t]
    #         else:
    #             if model_type == "MLP":
    #                 running_add = self.critic.predict(
    #                     np.reshape(self.new_state_list[t], (1, feature_size))
    #                 )[0]
    #             elif model_type == 'LSTM':
    #                 running_add = self.critic.predict(
    #                     np.reshape(self.new_state_list[t], (1, series_size, feature_size))
    #                 )[0]
    #             else:
    #                 running_add = self.critic.predict(
    #                     np.reshape(self.new_state_list[t], (1, series_size, feature_size, 1))
    #                 )[0]
    #             running_add = self.reward_list[t] + self.discount_factor * running_add
    #         discounted_rewards[t] = running_add
    #     return discounted_rewards

    def get_discount_rewards(self):
        discounted_rewards = np.zeros_like(self.reward_list)

        for t in reversed(range(0, len(self.reward_list))):
            if self.done_list[t]:
                running_add = self.reward_list[t]
            else:
                if self.model_type == "MLP":
                    running_add_2 = self.critic.predict(
                        np.reshape(self.new_state_list[t], (1, self.feature_size))
                    )[0]
                elif self.model_type == 'LSTM':
                    running_add_2 = self.critic.predict(
                        np.reshape(self.new_state_list[t], (1, self.series_size, self.feature_size))
                    )[0]
                else:
                    running_add_2 = self.critic.predict(
                        np.reshape(self.new_state_list[t], (1, self.series_size, self.feature_size, 1))
                    )[0]
                running_add = self.reward_list[t] + self.discount_factor * running_add_2
            discounted_rewards[t] = running_add
        return discounted_rewards

    def append_memory(self, state, action, reward, new_state, done):
        self.state_series.append(state.tolist())
        self.state_list.append(self.state_series.copy())

        act = np.zeros(self.action_size)
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

        if self.model_type == "MLP":
            values = self.critic.predict(
                np.reshape(self.state_list, (len(self.state_list), self.feature_size))
            )
        elif self.model_type == "LSTM":
            values = self.critic.predict(
                np.reshape(self.state_list, (len(self.state_list), self.series_size, self.feature_size))
            )
        else:
            values = self.critic.predict(
                np.reshape(self.state_list, (len(self.state_list), self.series_size, self.feature_size, 1))
            )
        values = np.reshape(values, len(values))  # values.shape --> (65,) when len(self.state_list) == 65

        advantages = discounted_rewards - values
        advantages_clipped = np.clip(advantages, -5.0, 5.0)

        if self.model_type == "MLP":
            local_input = np.reshape(self.state_list, (len(self.state_list), self.feature_size))
        elif self.model_type == "LSTM":
            local_input = np.reshape(self.state_list, (len(self.state_list), self.series_size, self.feature_size))
        else:
            local_input = np.reshape(self.state_list, (len(self.state_list), self.series_size, self.feature_size, 1))

        actor_loss, loss, policy, action_prob, weights = self.optimizer[0]([
            local_input,
            np.reshape(self.action_list, (len(self.action_list), self.action_size)),
            advantages
        ])

        critic_loss, value, weights_0, weights_1, weights_2, weights_3 = self.optimizer[1]([
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

        return abs(actor_loss), abs(critic_loss)

    def get_action(self, state):
        state = np.asarray(state)
        if self.model_type == "MLP":
            state = np.reshape(state, (1, self.feature_size))
        elif self.model_type == "LSTM":
            state = np.reshape(state, (1, self.series_size, self.feature_size))
        else:
            state = np.reshape(state, (1, self.series_size, self.feature_size, 1))

        policy = self.actor.predict(state)[0]
        action = np.random.choice(self.action_size, 1, p=policy)[0]

        return policy, np.argmax(policy), action
