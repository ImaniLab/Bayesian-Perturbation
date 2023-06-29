from enum import IntEnum
from typing import Tuple, Optional, List
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
import numpy
from itertools import product
import pickle

number_of_states = 10
noise_num = 0.001
dna_dsb = np.array(np.mat('0 0 0 0 0 0 0 0 0 0'))

nums_missing = 5

with open('GRNBOVals10Gene5Missing.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    ConnectivitySpace = pickle.load(f)

combs = product(range(2), repeat=number_of_states)
combs = list(combs)
A = numpy.array(combs)


def register_env() -> None:
    register(id="TenGene1-v0", entry_point="TenGene1:TenGene1", max_episode_steps=100) #459, 10000 was bad, 1000 was okay


class Action(IntEnum):

    GENE0 = 0
    GENE4 = 1
    GENE7 = 2
    GENE2 = 3

class TenGene1(Env):

    def __init__(self) -> None:
        self.stateNum = 2**number_of_states  # 0-1023, 1024 in total
        self.modelNum = 3**nums_missing  # 243 models, with 5 unknown
        self.qNum = 10  # 10 qs needed

        self.start_pos = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3)
        self.agent_pos = None

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2),spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2), spaces.Box(0, 1, (self.qNum,), dtype=np.float64))
        )

    def seed(self, seed: Optional[int] = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> Tuple[int, int]:
        self.agent_pos = self.start_pos

        return self.agent_pos

    def step(self, action: Action) -> Tuple[Tuple[int, int, int, int, int, int, int, int, int, int, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64, np.float64], float, bool, dict]:

        mycombs = product(range(3), repeat=nums_missing)
        Combinations = np.float64(numpy.array(list(mycombs)) - 1)

        for u in range(nums_missing):

            numpy.place(Combinations[:, u], Combinations[:, u] == -1, (self.agent_pos)[u*2+0+number_of_states])
            numpy.place(Combinations[:, u], Combinations[:, u] == 0, (self.agent_pos)[u*2+1+number_of_states])
            numpy.place(Combinations[:, u], Combinations[:, u] == 1, 1-(self.agent_pos)[u*2+0+number_of_states]-(self.agent_pos)[u*2+1+number_of_states])

        model_probs = np.ones(self.modelNum)

        for v in range(nums_missing):

            model_probs = np.multiply(Combinations[:, v], model_probs)

        numpy.place(model_probs, model_probs<0, 0)
        model_probs = model_probs/np.sum(model_probs)
        model_probs = numpy.nan_to_num(model_probs, copy=True, nan=0.0, posinf=None, neginf=None)
        if np.sum(model_probs) == 0:
            model_probs = (1/self.modelNum)*np.ones(self.modelNum)

        model_choice = np.random.choice(np.arange(self.modelNum), 1, p=model_probs)#222#np.random.choice(np.arange(self.modelNum), 1, p=model_probs)#int((self.agent_pos)[7])#np.random.choice(np.arange(self.modelNum), 1, p=model_probs)

        Cnext = np.atleast_2d(ConnectivitySpace[model_choice, :])
        CnextReshape = np.reshape(Cnext, (number_of_states, number_of_states))

        x_old = np.zeros(number_of_states)
        for t in range(number_of_states):
            x_old[t] = int((self.agent_pos)[t])
        x_old = np.atleast_2d(x_old)
        x_old.reshape((1,number_of_states))
        x_new = 1 * (np.matmul(x_old, (CnextReshape.T) + dna_dsb) > 0)

        perturb = np.zeros(number_of_states)
        if action == Action.GENE0:
            perturb[0] = 1
        if action == Action.GENE4:
            perturb[4] = 1
        if action == Action.GENE7:
            perturb[7] = 1
        if action == Action.GENE2:
            perturb[2] = 1
        perturb.reshape((1,number_of_states))
        noise = np.random.binomial(1, noise_num, size=(1, number_of_states))

        x_new = 1 * np.logical_xor(x_new, perturb)
        x_new = 1 * np.logical_xor(x_new, noise)

        x_new_index = (A == x_new).all(axis=1).nonzero()
        x_new_index = np.array(x_new_index[0])

        M_indices = np.ones(self.modelNum)
        for j in range(self.modelNum):

            Cnext = np.atleast_2d(ConnectivitySpace[j, :])
            CnextReshape = np.reshape(Cnext, (number_of_states, number_of_states))

            x_expected = 1 * (np.matmul(x_old, (CnextReshape.T) + dna_dsb) > 0)
            x_expected = 1 * np.logical_xor(x_expected, perturb)

            for k in range(number_of_states):

                if x_expected[0, k] == x_new[0, k]:
                    temp1 = 1 - noise_num
                else:
                    temp1 = noise_num

                M_indices[j] = M_indices[j] * temp1

        model_probs_new = np.multiply(model_probs, M_indices)
        model_probs_new = model_probs_new/np.sum(model_probs_new) #normalized

        new_qvals = np.zeros(self.qNum)
        mycombs2 = product(range(3), repeat=nums_missing)
        Combinations2 = np.float64(numpy.array(list(mycombs2)) - 1)

        for w in range(nums_missing):

            indices = np.array(np.atleast_2d(Combinations2[:, w] == -1).all(axis=0).nonzero())
            new_qvals[2*w+0] = np.sum(model_probs_new[indices])

            indices = np.array(np.atleast_2d(Combinations2[:, w] == 0).all(axis=0).nonzero())
            new_qvals[2*w+1] = np.sum(model_probs_new[indices])

        done = False
        reward = np.amax(model_probs_new) - np.amax(model_probs)

        new_state = np.zeros(number_of_states + self.qNum)
        for h in range(number_of_states+self.qNum):
            if h < number_of_states:
                new_state[h] = x_new[0,h]
            else:
                new_state[h] = new_qvals[h-number_of_states]

        self.agent_pos = tuple(new_state)

        return self.agent_pos, reward, done, {}
