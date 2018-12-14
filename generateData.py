import gym
import pickle

env = gym.make('MountainCar-v0')

keep_size = 1000

total = []
for i_episode in range(int(2500)):
    obs = env.reset()
    memory = []
    for t in range(int(1e11)):
        action = env.action_space.sample()
        newObs, reward, done, info = env.step(action)
        memory.append([list(obs),action,reward])
        obs = newObs
        if done:
            print("{}th episode finished after {} timesteps"\
            .format(i_episode, t+1))
            break

    if len(memory) > keep_size:
        memory = memory[-keep_size:]

    total.append(memory)

with open('train.pkl','wb') as mf:
    pickle.dump(total,mf)
