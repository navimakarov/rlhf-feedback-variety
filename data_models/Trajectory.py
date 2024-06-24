import numpy as np
import pickle


class Trajectory:
    def __init__(self):
        self.trajectory = []
        self.total_reward = 0

    def __repr__(self):
        return self.trajectory.__repr__()

    def add(self, state: np.ndarray, action: np.ndarray, reward):
        self.trajectory.append((state, action))
        self.total_reward += reward

    def __eq__(self, other):
        if not isinstance(other, Trajectory):
            return False

        return self.trajectory.__repr__() == other.trajectory.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    @staticmethod
    def serialize_trajectories(trajectories, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(trajectories, file)

    @staticmethod
    def deserialize_trajectories(file_name):
        with open(file_name, 'rb') as file:
            return pickle.load(file)
