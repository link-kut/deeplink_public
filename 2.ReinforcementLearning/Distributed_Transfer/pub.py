import pickle
import zlib

import zmq
import random
import sys
import time

port = "5556"
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://localhost:%s" % port)

while True:
    episode_msg = {
        "type": "episode",
        "episode": 50,
        "worker_idx": 1,
        "loss": 10,
        "score": 20,
        "weights": {0: 100}
    }

    print(episode_msg)

    episode_msg = pickle.dumps(episode_msg, protocol=-1)
    episode_msg = zlib.compress(episode_msg)
    socket.send(episode_msg)

    time.sleep(1)
