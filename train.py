from ppo_vec import Agent
from platform_env import PlatformEnv
if __name__ == '__main__':
    envs = []
    num_envs = [1,1] # [num_normal_envs,num_leap_training_envs]
    for i in range(num_envs[0]):
        env = PlatformEnv()
        env.seed(i)
        envs.append(env)
    for j in range(num_envs[1]):
        env = PlatformEnv(type = 'leap')
        env.seed(j)
        envs.append(env)
    
    lr = 0.0001
    gamma = 0.99
    K_epochs = 40
    eps_clip = 0.2
    max_training_steps = 500000
    model_name = 'exp'

    model = Agent(envs, lr, gamma, K_epochs, eps_clip)
    model.learn(max_training_steps, model_name, num_envs)
