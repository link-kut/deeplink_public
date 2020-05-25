import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 3
T_horizon = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, s_size, hidden_size, a_size):
        super(ActorCritic, self).__init__()
        self.fc0 = nn.Linear(s_size, hidden_size[0])
        self.fc1 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc2 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc3 = nn.Linear(hidden_size[2], a_size)
        self.fc3_v = nn.Linear(hidden_size[2], 1)

    def pi(self, x, softmax_dim=0):
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=softmax_dim)

    def v(self, x):
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        v = self.fc3_v(x)
        return v

class PPO:
    def __init__(self, env, worker_id, logger, n_inputs, hidden_size, n_outputs, gamma, trajectories_sampling, verbose):
        self.env = env

        self.worker_id = worker_id

        # discount rate
        self.gamma = gamma

        # initially 90% exploration, 10% exploitation
        self.epsilon = 0.5

        # learning rate
        self.learning_rate = 0.002

        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.n_outputs = n_outputs

        self.data = []

        # learning rate
        self.learning_rate = 0.002

        self.model = self.build_model(self.n_inputs, self.hidden_size, self.n_outputs)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # Policy Network is 256-256-256-2 MLP
    def build_model(self, n_inputs, hidden_size, n_outputs):
        model = ActorCritic(n_inputs, hidden_size, n_outputs).to(device)
        return model

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.model.v(s_prime) * done_mask
            delta = td_target - self.model.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.model.pi(s, softmax_dim=1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.model.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

env = gym.make('CartPole-v0')
n_inputs = int(env.observation_space.shape[0] / 2)
n_outputs = env.action_space.n
HIDDEN_1_SIZE = 128
HIDDEN_2_SIZE = 128
HIDDEN_3_SIZE = 128
GAMMA = 1.0 # discount factor
VERBOSE = False
TRAJECTORIES_SAMPLING = 2

def main():
    agent = PPO(
        env,
        0,
        None,
        n_inputs=n_inputs,
        hidden_size=[HIDDEN_1_SIZE, HIDDEN_2_SIZE, HIDDEN_3_SIZE],
        n_outputs=n_outputs,
        gamma=GAMMA,
        trajectories_sampling=TRAJECTORIES_SAMPLING,
        verbose=VERBOSE
    )
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s = env.reset()
        s = s[2:]
        done = False
        while not done:
            for t in range(T_horizon):
                prob = agent.model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                s_prime = s_prime[2:]

                agent.put_data((s, a, r / 100.0, s_prime, prob[a].item(), done))
                s = s_prime

                score += r
                if done:
                    break

            agent.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()