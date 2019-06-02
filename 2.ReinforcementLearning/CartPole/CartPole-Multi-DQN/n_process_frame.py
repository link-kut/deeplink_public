# -*- coding: utf-8 -*-
# https://github.com/openai/gym/wiki/CartPole-v0
import math
import threading
import time
import zlib

import tensorflow as tf
from logger import get_logger
import sys
import json

from tensorflow.python.keras.layers import Dropout
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers import LeakyReLU

print(tf.__version__)
tf.config.gpu.set_per_process_memory_fraction(0.4)

from tensorflow.python.keras.layers import Dense, Input

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import gym
from multiprocessing import Process
import zmq
import matplotlib.pyplot as plt
import os
import pickle
import socket

from numpy.random import seed
seed(1)
tf.random.set_seed(2)

import warnings
warnings.filterwarnings("ignore")

ddqn = True
num_hidden_layers = 3
num_weight_transfer_hidden_layers = 4
num_workers = 4
score_based_transfer = True
loss_based_transfer = False
soft_transfer = True
verbose = False

class DQNAgent:
    def __init__(self, worker_idx, max_episodes):
        self.worker_idx = worker_idx
        self.max_episodes = max_episodes

    def start_rl(self, socket):
        for episode in range(self.max_episodes):
            if episode == 5 and self.worker_idx == 3:
                solved = True
            else:
                solved = False
            self.send_episode_info_and_adapt_best_weights(episode, socket, solved)

    def send_episode_info_and_adapt_best_weights(self, episode, socket, solved):
        if solved:
            episode_msg = {
                "type": "solved",
                "episode": episode,
                "worker_idx": self.worker_idx,
            }
        else:
            episode_msg = {
                "type": "episode",
                "episode": episode,
                "worker_idx": self.worker_idx,
            }

        try:
            episode_msg = pickle.dumps(episode_msg, protocol=-1)
            episode_msg = zlib.compress(episode_msg)
            socket.send(episode_msg)

            episode_ack_msg = socket.recv()
            episode_ack_msg = zlib.decompress(episode_ack_msg)
            episode_ack_msg = pickle.loads(episode_ack_msg)

            continue_loop = True
            if episode_ack_msg["type"] == "episode_ack":
                print(self.worker_idx, "acked_episode", episode_ack_msg["acked_episode"])
            elif episode_ack_msg["type"] == "solved_ack":
                print(self.worker_idx, "acked_solved", episode_ack_msg["acked_solved"])
            else:
                pass

        except zmq.error.ZMQError as e:
            print("Client: zmq.error.ZMQError!!!!!!!!!!!!!!!!!!!! - ", self.worker_idx)
            continue_loop = False


        return continue_loop



def worker_func(worker_idx, max_episodes, port):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://127.0.0.1:' + str(port))

    dqn_agent = DQNAgent(worker_idx, max_episodes)
    dqn_agent.start_rl(socket)


class MultiDQN:
    def __init__(self):
        self.continue_loop = True
        pass


def server_func(multi_dqn):
    context = zmq.Context()
    sockets = {}

    for worker_idx in range(num_workers):
        sockets[worker_idx] = context.socket(zmq.REP)
        sockets[worker_idx].bind('tcp://127.0.0.1:' + str(20000 + worker_idx))

    solved_notification_per_workers = 0

    while True:
        try:
            for worker_idx in range(num_workers):
                episode_msg = sockets[worker_idx].recv()
                episode_msg = zlib.decompress(episode_msg)
                episode_msg = pickle.loads(episode_msg)

                if episode_msg["type"] == "episode":
                    if multi_dqn.continue_loop:
                        episode_ack_msg = {
                            "type": "episode_ack",
                            "acked_episode": int(episode_msg["episode"]) + 100
                        }
                    else:
                        print("!!!!", solved_notification_per_workers)
                        solved_notification_per_workers += 1
                        episode_ack_msg = {
                            "type": "solved_ack",
                            "acked_solved": solved_notification_per_workers
                        }
                    send_to_worker(sockets[worker_idx], episode_ack_msg)
                elif episode_msg["type"] == "solved":
                    print("SOLVED!!!! - ", episode_msg["worker_idx"], solved_notification_per_workers)
                    solved_notification_per_workers = 1
                    multi_dqn.continue_loop = False
                else:
                    pass

            if solved_notification_per_workers == num_workers:
                break
        except zmq.error.ZMQError as e:
            print("zmq.error.ZMQError!!!!!!!!!!!!!!!!!!!!")
            break


def send_to_worker(socket, episode_ack_msg):
    episode_ack_msg = pickle.dumps(episode_ack_msg, protocol=-1)
    episode_ack_msg = zlib.compress(episode_ack_msg)
    socket.send(episode_ack_msg)


if __name__ == '__main__':
    num_experiments = 1
    max_episodes = 10

    for _ in range(num_experiments):
        multi_dqn = MultiDQN()

        server = Process(target=server_func, args=(multi_dqn,))
        server.start()

        clients = []
        for worker_idx in range(num_workers):
            client = Process(target=worker_func, args=(
                worker_idx,
                max_episodes,
                20000 + worker_idx
            ))

            clients.append(client)
            client.start()

        while True:
            is_anyone_alive = True
            for client in clients:
                is_anyone_alive = client.is_alive()
            is_anyone_alive = server.is_alive()

            if not is_anyone_alive:
                break

            time.sleep(1)