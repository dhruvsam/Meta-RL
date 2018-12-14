
from gym.utils import seeding
def randomize_pendulum(env):
    m = config['pendulum_mass_min'] + np.random.rand()*(config['pendulum_mass_max'] - config['pendulum_mass_min'])
    l = config['pendulum_len_min'] + np.random.rand()*(config['pendulum_len_max'] - config['pendulum_len_min'])
    env.m = m
    env.l = l

import yaml
cfg_filename = 'hopper-config.yml'
with open(cfg_filename,'r') as ymlfile:
    config = yaml.load(ymlfile)

def sample_tasks(num_tasks):
    random = seeding.np_random(seed = None)
    goals = random.uniform(-0.5, 0.5, size=(num_tasks, 2))
    tasks = [{'goal': goal} for goal in goals]
    return tasks

def reset_task(env, task):
        return env.unwrapped.reset_task(task)
