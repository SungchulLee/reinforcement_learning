import tensorflow as tf
import numpy as np

from config import Config
from agent import Agent
from environment import Environment


def main():
    sess = tf.Session()
    config = Config()
    env = Environment()
    agent = Agent(sess, config, env)
    agent.train()


if __name__ == "__main__":
    main()