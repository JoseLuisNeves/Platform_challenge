from ppo_vec import Agent
from platform_env import PlatformEnv
import numpy as np
if __name__ == '__main__':
    env = PlatformEnv()
    cum_cum_rew = []
    N = 100000
    for _ in range(N):
        state = env.reset()
        cum_rew = 0
        while True: 
            action = env.action_space.sample()
            disc = action[0]
            conti = tuple([action[1][0][0],action[1][1][0],action[1][2][0]])
            act = (disc,conti)
            n_state, reward, done, info = env.step(act)
            cum_rew += reward
            if done: 
                #print('Episode reward', cum_rew)
                break
            #env.render()
        cum_cum_rew.append(cum_rew)
    cum_cum_rew = np.array(cum_cum_rew)
    print('Mean',np.mean(cum_cum_rew), 'Std', np.std(cum_cum_rew))
        