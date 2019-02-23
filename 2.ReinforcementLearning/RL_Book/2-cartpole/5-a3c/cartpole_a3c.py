import threading
import numpy as np
import tensorflow as tf
import pylab
import time
import gym
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

# global variables for threading
episode = 0

EPISODES = 2000


# This is A3C(Asynchronous Advantage Actor Critic) agent(global) for the Cartpole
# In this example, we use A3C algorithm
class A3CAgent:
    def __init__(self, state_size, action_size, env_name):
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # get gym environment name
        self.env_name = env_name

        # these are hyper parameters for the A3C
        self.actor_lr = 0.001
        self.critic_lr = 0.001
        self.discount_factor = .99
        self.hidden1, self.hidden2 = 24, 24
        self.threads = 8

        self.global_score_list = []
        self.__global_score_list_lock = threading.RLock()

        # create model for actor and critic network
        self.actor, self.critic = self.build_model()

        # method for training actor and critic network
        self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

        if self.load_model:
            self.actor.load_weights("./save_model/cartpole_a3c_actor.h5")
            self.critic.load_weights("./save_model/cartpole_a3c_critic.h5")
        else:
            self.sess = tf.InteractiveSession()
            K.set_session(self.sess)
            self.sess.run(tf.global_variables_initializer())

    # approximate policy and value using Neural Network
    # actor -> state is input and probability of each action is output of network
    # critic -> state is input and value of state is output of network
    # actor and critic network share first hidden layer
    def build_model(self):
        state = Input(batch_shape=(None, self.state_size))
        shared = Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_uniform')(
            state)

        actor_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform')(shared)
        action_prob = Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform')(actor_hidden)

        value_hidden = Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform')(shared)
        state_value = Dense(1, activation='linear', kernel_initializer='he_uniform')(value_hidden)

        actor = Model(inputs=state, outputs=action_prob)
        critic = Model(inputs=state, outputs=state_value)

        actor._make_predict_function()
        critic._make_predict_function()

        actor.summary()
        critic.summary()

        return actor, critic

    # make loss function for Policy Gradient
    # [log(action probability) * advantages] will be input for the back prop
    # we add entropy of action probability to loss
    def actor_optimizer(self):
        action = K.placeholder(shape=(None, self.action_size))
        advantages = K.placeholder(shape=(None,))

        policy = self.actor.output

        action_prob = K.sum(action * policy, axis=1)
        eligibility = K.log(action_prob + 1e-10) * K.stop_gradient(advantages)
        loss = -K.sum(eligibility)

        # policy의 entropy를 더하여 각 행동에 대한 확률을 가급적 동일하게 맞추려고 노력함.
        entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
        actor_loss = loss + 0.01 * entropy

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], actor_loss)
        train = K.function([self.actor.input, action, advantages], [], updates=updates)
        return train

    # make loss function for Value approximation
    def critic_optimizer(self):
        discounted_reward = K.placeholder(shape=(None,))

        value = self.critic.output

        loss = K.mean(K.square(discounted_reward - value))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, discounted_reward], [], updates=updates)
        return train

    # make agents(local) and start training
    def train(self):
        # self.load_model()
        agents = [Agent(i, self, self.actor, self.critic, self.optimizer,
                        self.env_name, self.discount_factor, self.action_size, self.state_size)
                  for i in range(self.threads)]

        for agent in agents:
            agent.start()

        while True:
            time.sleep(5)

            print("Global Save!!!")
            self.save_model()

            is_anyone_alive = True
            for agent in agents:
                is_anyone_alive = agent.is_alive()

            if not is_anyone_alive:
                break

    def save_model(self):
        with self.__global_score_list_lock:
            pylab.plot(range(len(self.global_score_list)), self.global_score_list, 'b')
            pylab.savefig("./save_graph/cartpole_a3c.png")

            self.actor.save_weights("./save_model/cartpole_a3c_actor.h5")
            self.critic.save_weights("./save_model/cartpole_a3c_critic.h5")

    def append_global_score_list(self, score):
        with self.__global_score_list_lock:
            self.global_score_list.append(score)

    def get_action(self, state):
        policy = self.actor.predict(np.reshape(state, [1, self.state_size]))[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]


# This is Agent(local) class for threading
class Agent(threading.Thread):
    def __init__(self, index, global_agent, actor, critic, optimizer, env_name, discount_factor, action_size,
                 state_size):
        threading.Thread.__init__(self)

        self.env = gym.make(env_name)
        self.states = []
        self.rewards = []
        self.actions = []

        self.global_agent = global_agent

        self.index = index
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer
        self.discount_factor = discount_factor
        self.action_size = action_size
        self.state_size = state_size
        self.local_score_list = []

    # Thread interactive with environment
    def run(self):
        local_episode = 0

        while local_episode < EPISODES:
            state = self.env.reset()
            local_score = 0
            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                local_score += reward

                self.append_memory(state, action, reward)

                state = next_state

                if done:
                    local_episode += 1
                    print("agent: {0}, local_episode: {1}, local_score: {2}, mean_local_score: {3}".format(
                        self.index,
                        local_episode,
                        local_score,
                        np.mean(self.local_score_list[-min(10, len(self.local_score_list)):]
                                )))
                    self.local_score_list.append(local_score)

                    self.global_agent.append_global_score_list(local_score)

                    self.train_episode(done=local_score != 500)
                    time.sleep(0.05)
                    break

            if np.mean(self.local_score_list[-min(10, len(self.local_score_list)):]) > 495:
                print("### agent {0} comes to converge successfully at episode {1}!".format(self.index, local_episode))
                self.global_agent.save_model()
                break

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def get_discount_rewards(self, rewards, done=True):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        if not done:
            running_add = self.critic.predict(np.reshape(self.states[-1], (1, self.state_size)))[0]
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    # this is used for calculating discounted rewards
    def append_memory(self, state, action, reward):
        self.states.append(state)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
        self.rewards.append(reward)

    # update policy network and value network every episode
    def train_episode(self, done):
        discounted_rewards = self.get_discount_rewards(self.rewards, done=done)

        values = self.critic.predict(np.array(self.states))  # values.shape --> (65, 1) when len(self.states) == 65
        values = np.reshape(values, len(values))  # values.shape --> (65,) when len(self.states) == 65

        advantages = discounted_rewards - values

        self.optimizer[0]([self.states, self.actions, advantages])
        self.optimizer[1]([self.states, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

    def get_action(self, state):
        policy = self.actor.predict(np.reshape(state, [1, self.state_size]))[0]
        return np.random.choice(self.action_size, 1, p=policy)[0]


if __name__ == "__main__":
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    global_agent = A3CAgent(state_size, action_size, env_name)

    if not global_agent.load_model:
        global_agent.train()
    else:
        for e in range(EPISODES):
            done = False
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            score = 0

            while not done:
                env.render()

                action = global_agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])

                score += reward
                state = next_state

                if done:
                    score = score if score == 500 else score + 100
                    global_agent.append_global_score_list(score)
                    print(
                        "episode: {0:3d}, score: {1:6.2f}, mean_score: {2:6.2f}".format(
                            int(e),
                            score,
                            np.mean(global_agent.global_score_list[-min(10, len(global_agent.global_score_list)):])
                        ))
