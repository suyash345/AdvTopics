import gym
from gym import spaces
import numpy as np
import qiskit
from stable_baselines3.common.callbacks import BaseCallback
from qiskit import Aer, QuantumCircuit
from stable_baselines3 import PPO
from qiskit.quantum_info import Operator

def Init_of_gates():
    identity = np.identity(4)
    not_gate = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
    entangle = Operator(1 / np.sqrt(2) * (identity + 1j * not_gate))
    unentangle = Operator(1 / np.sqrt(2) * (identity - 1j * not_gate))
    return entangle, unentangle

class QuantumGameEnv(gym.Env):
    def __init__(self, numQs, CalculateReward, limitOfStepsinEpisode=100):
        super().__init__()
        self.numberOfStepsTakenSoFar = 0
        self.numQs = numQs
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2 * 2**self.numQs,), dtype=np.float32)
        self.CalculateReward = CalculateReward
        self.actionAssocaitedWithRewards = {}
        self.Qcircuit = QuantumCircuit(self.numQs)
        self.prevAction = None
        self.limitOfStepsinEpisode = limitOfStepsinEpisode
        self.entangle, self.InverseEntangle = Init_of_gates()
        self.action_space = spaces.Box(low=np.array([0, 0, 0] * self.numQs), high=np.array([2*np.pi, 2*np.pi, 2*np.pi] * self.numQs), dtype=np.float32)
        self.biggestRewardAction = None


    def reset(self):
        self.numberOfStepsTakenSoFar = 0
        self.Qcircuit = QuantumCircuit(2, 2)
        self.Qcircuit.append(self.entangle, [0, 1])
        return self._get_state()

    def step(self, action):
        self.prevAction = action
        self._apply_gates(action)
        if not self.action_space.contains(action):
            print("action not found!")
        counts = self.measure()
        reward = self.CalculateReward(counts)
        self.actionAssocaitedWithRewards[tuple(action)] = reward  # Using tuple as a dictionary key
        self.biggestRewardAction = max(self.actionAssocaitedWithRewards, key=self.actionAssocaitedWithRewards.get)
        self.numberOfStepsTakenSoFar = self.numberOfStepsTakenSoFar + 1
        done = False
        if(self.numberOfStepsTakenSoFar >= self.limitOfStepsinEpisode):
            done = True
        elif(self.numberOfStepsTakenSoFar < self.limitOfStepsinEpisode):
            done = False
        newState = self._get_state()
        self.prevObs = newState
        return newState, reward, done, {}

    def _apply_gates(self, actions):
        for qubit in range(self.numQs):
            lam = actions[3*qubit + 2]
            phi = actions[3*qubit + 1]
            theta = actions[3*qubit]
            self.Qcircuit.u(theta, phi, lam, qubit)

    def _get_state(self):
        print("getState")
        backend = Aer.get_backend('statevector_simulator')
        statevector = qiskit.execute(self.Qcircuit, backend).result().get_statevector()
        imagPart = np.imag(statevector)
        realPart = np.real(statevector)
        return_val = np.concatenate([realPart, imagPart])
        return return_val

    def render(self, mode='human'):
        print("rendered")

    def measure(self):
        self.Qcircuit.append(self.InverseEntangle.to_instruction(), [0, 1])
        backend = Aer.get_backend('qasm_simulator')
        self.Qcircuit.measure([0, 1], [0, 1])
        result = qiskit.execute(self.Qcircuit, backend, shots=1).result()
        return result.get_counts()

def battleofsexes_reward(state):
    dict_key = list(state.keys())[0]
    if dict_key == '00':
        return 0
    if dict_key == '01':
        return 5
    if dict_key == '10':
        return 5
    if dict_key == '11':
        return 0

env = QuantumGameEnv(numQs=2, CalculateReward=battleofsexes_reward)
env.reset()

nActions = env.action_space.shape[-1]

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_log/", ent_coef=0.005, n_steps=2048, gamma=0.95, learning_rate=0.0003)
model.learn(total_timesteps=100)
episodes = 20
for ep in range(episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        env.render()
    counts = env.measure()
env.close()
