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

    # selecting hyperparameters
    lr = 0.0001
    gamma = 0.99
    K_epochs = 40
    eps_clip = 0.2
    model_name = '254units_norm1_leap1_400ksteps'

    # building agent and start learning
    model = Agent(envs, lr, gamma, K_epochs, eps_clip)
    model.load(model_name)

    # running
    model.test(num_envs)




