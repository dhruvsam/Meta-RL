# Policy gradient solution with MC return
# Please view the full source code here: https://github.com/lguye/openai-exercise/tree/master/MountainCar-v0/PolicyGradient
# to run this script 'train.pkl' must be generated using generateData.py first

import pickle
import numpy as np
from random import shuffle

import tensorflow as tf
import network as q

with open('train.pkl','rb') as mf:
    train = pickle.load(mf)

def getMC(r,gamma=1):
    targets = []
    for i, _ in enumerate(r):
        target = np.sum([gamma**j*item for j,item in enumerate(r[i:])])
        targets.append([target])
    return targets

class OneHotEncoder():
    def __init__(self,span):
        self.span = np.array(span).astype(np.int)
        self.count = self.span.shape[0]

    def encode(self,vals):
        # return boolean
        vals = np.array(vals).astype(np.int)
        base = np.matlib.repmat(self.span,vals.shape[0],1)
        return (base.T == vals).T


saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

gamma = 1
batch_size = 20

encoder = OneHotEncoder([0,1,2])

for epoch in range(2):
    for i,memory in enumerate(train[:2500]):
        memory = memory[-500:]
        observations, actions, rewards = zip(*memory)
        observations = np.array(observations)
        actions = encoder.encode(actions)
        targets = np.array(getMC(rewards,gamma))
        targets = targets/np.max(np.abs(targets)) + 1

        idx = list(range(len(memory)))
        shuffle(idx)

        for chunk in qn.chunks(idx,batch_size):
            obsChunk = observations[chunk,:]
            targetChunk = targets[chunk]
            actionChunk = actions[chunk,:]

            sess.run(q.train_action,feed_dict={q.observation:obsChunk,
            q.G:targetChunk, q.action:actionChunk})

#%% evaluate learned model

import gym
env = gym.make('MountainCar-v0')

for i_episode in range(500):
    obs = env.reset()
    for t in range(int(1e10)):

        action_prob = sess.run(q.action_prob,feed_dict={q.observation:[obs]})
        action = np.argmax(action_prob[0])

        newObs, reward, done, info = env.step(action)
        obs = newObs
        if done:
            print("{}th episode finished after {} timesteps"\
            .format(i_episode, t+1))
            break
