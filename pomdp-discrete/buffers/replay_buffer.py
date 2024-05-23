import abc
import numpy as np

def compute_rewards_to_go(rewards, gamma):
    """
    Compute rewards to go, based on a sequence of rewards.
    :param rewards: a sequence of rewards, shape (L, 1)
    :param gamma: discount factor
    :return: a sequence of rewards to go, shape (L, 1)
    """
    assert rewards.ndim == 2 and rewards.shape[1] == 1
    L = rewards.shape[0]
    rewards_to_go = np.zeros_like(rewards, dtype=np.float32)
    rewards_to_go[-1] = rewards[-1]
    for i in reversed(range(L - 1)):
        rewards_to_go[i] = rewards[i] + gamma * rewards_to_go[i + 1]
    return rewards_to_go
    

class ReplayBuffer(object, metaclass=abc.ABCMeta):
    """
    A class used to save and replay data.
    """

    @abc.abstractmethod
    def add_sample(
        self, observation, action, reward, next_observation, terminal, **kwargs
    ):
        """
        Add a transition tuple.
        """
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """
        Let the replay buffer know that the episode has terminated in case some
        special book-keeping has to happen.
        :return:
        """
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass
