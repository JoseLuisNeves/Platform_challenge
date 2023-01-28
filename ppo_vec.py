import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical,MultivariateNormal
import numpy as np
import time

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class RolloutBuffer:
    def __init__(self,n_envs):
        self.n_envs = n_envs
        self.actions = [[]]*n_envs
        self.param = [[]]*n_envs
        self.states = [[]]*n_envs
        self.action_logprobs = [[]]*n_envs
        self.param_logprobs = [[]]*n_envs
        self.rewards = [[]]*n_envs
        self.state_values = [[]]*n_envs
        self.is_done = [[]]*n_envs
    
    def clear(self):
        self.actions = [[]]*self.n_envs
        self.param = [[]]*self.n_envs
        self.states = [[]]*self.n_envs
        self.action_logprobs = [[]]*self.n_envs
        self.param_logprobs = [[]]*self.n_envs
        self.rewards = [[]]*self.n_envs
        self.state_values = [[]]*self.n_envs
        self.is_done = [[]]*self.n_envs

#  STEP 1: Define the neural network class that is going to be approximate the policies
#  Notes: We will need an output for the critic, for the discrete actions and for the continuous one.
#         This output layers will come from the same feature extraction layers, because the features 
#         needed are the same and its less computationally expensive.
#         The discrete head will output a value for each action possible, from which we will create a 
#         distribution using softmax from which we will sample. The continuous head will give us a mean
#         and std to build a Gaussian distribution from which we will sample the parameters values for 
#         each possible action. The critic will have as input only the statue to which it will associate 
#         a value. We use the state-value function as the critic instead of the action-value
#         function is that action-value function suffers from the overparameterization problem in parameterized action space.  

class ActorsCritic(nn.Module):
    def __init__(self, state_dim, action_dim, parameter_init, parameter_std_init): 
        super(ActorsCritic, self).__init__()
        self.parameter_std = parameter_std_init
        self.parameter_init = parameter_init
        self.parameters_var = self.parameter_std*self.parameter_std
        self.parameters_var = self.parameters_var
        units = 256
        # Feature extraction layers
        self.layer1 = nn.Linear(state_dim, units)
        self.layer2 = nn.Linear(units, units)

        # NN heads
        self.param = nn.Linear(units, action_dim)
        self.action = nn.Linear(units, action_dim)
        self.state =  nn.Linear(units, 1) 

    def update_param_std(self, param_std_decay_rate): 
        self.parameter_std = self.parameter_std*(1-param_std_decay_rate)
        self.parameters_var = self.parameter_std*self.parameter_std
        self.parameters_var = self.parameters_var

    def forward(self, obs): 
        # Feature extraction 
        x = F.relu(self.layer1(obs)) 
        x = F.relu(self.layer2(x)) 

        # getting the value of the state
        self.state_value = self.state(x)
        # we use detach here because we want to get read of the graphs and just have the outputs as numpy arrays

        # getting action distribution to sample, and sample
        self.action_probs = F.softmax(self.action(x), dim = -1)
        dist_action = Categorical(self.action_probs)
        action = dist_action.sample()
        action_log_prob = dist_action.log_prob(action)

        # getting parameters distribution to sample, and sample
        self.param_means = self.param(x)
        cov_mat = torch.diag(self.parameters_var)
        dist_param = MultivariateNormal(self.param_means, cov_mat)
        parameter = dist_param.sample()
        parameter_log_prob = dist_param.log_prob(parameter)
        return action.detach(),action_log_prob.detach(), parameter.detach(), parameter_log_prob.detach(), self.state_value.detach()
    
    def evaluate(self, obs, action, parameter):
        # Feature extraction 
        x = F.relu(self.layer1(obs)) 
        x = F.relu(self.layer2(x)) 

        # state value
        state_values = self.state(x)

        # action
        action_probs = F.softmax(self.action(x),dim = -1)
        dist_action = Categorical(action_probs)
        action_log_prob = dist_action.log_prob(action)

        # parameter
        param_means = self.param(x)
        cov_mat = torch.diag(self.parameters_var)
        dist_param = MultivariateNormal(param_means, cov_mat)
        parameter_log_prob = dist_param.log_prob(parameter) 

        # entropy
        dist_entropy_action = dist_action.entropy() 
        #print('discrete entropy', dist_entropy_action[0])
        dist_entropy_param = dist_param.entropy()
        #print('continuous entropy', dist_entropy_param[0])
        
        return action_log_prob, parameter_log_prob, state_values, dist_entropy_action, dist_entropy_param

class Agent:
    def __init__(self, envs, lr, gamma, K_epochs, eps_clip, parameter_init=[22.5,540,322.5], parameter_std_init = [22.5,540,430], std_decay_rate=0.001):
        # args
        self.envs = envs
        self.std_decay_rate = std_decay_rate
        self.n_envs = len(envs)
        self.lr = lr
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.parameter_init = torch.FloatTensor(np.array(parameter_init))
        self.parameter_std_init = torch.FloatTensor(np.array(parameter_std_init))
        env = envs[0]
        # action space and observation space:
        #   lets split the action space into the discrete actions and continuous parameters
        self.discrete_action_space = env.action_space.spaces[0]
        self.parameter_space = env.action_space.spaces[1]
        self.num_actions = self.discrete_action_space.n 
        self.obs_dim = env.observation_space[0].shape[0]
        # state = (player_enemy_features + platform_features,steps) = 
        # = ([self.player.position[0], self.player.velocity[0], enemy.position[0], enemy.dx] + [wd1, wd2, gap, pos, diff], steps)

        self.buffer = RolloutBuffer(self.n_envs)

        self.policy = ActorsCritic(self.obs_dim, self.num_actions, self.parameter_init,  self.parameter_std_init).to(device) 
        # we will not include the step number in the observation because it is not useful
        # and just complicates things
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.parameters(), 'lr': lr}
                    ])

        self.policy_old = ActorsCritic(self.obs_dim, self.num_actions, self.parameter_init, self.parameter_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

        # Policy initialized! STEP 1 COMPLETED!

# STEP 2: Define the learning loop. 
# Notes:  Lets create our Agent and start building the learn function with a while training loop.
#         
# STEP 3: Run a bunch of trajectories and gather their results.
# Notes:  We will store the results of the current policies on rollout. The data gathered each time we call rollout will be a batch.
#         From each step we need to store state, action, parameter, rewards-on-the-go, policy for the action(action_log_prob), 
#         policy for the parameter(parameter_log_prob) and dones.   

    def learn(self, max_training_timesteps,model_name,num_envs):

        print_running_norm_reward = 0
        print_running_leap_reward = 0
        print_running_episodes = 0

        time_step = 0
        i_episode = 0

        max_ep_len = 200 # steps (200 limit is an estimate by the env authors)
        print_freq = max_ep_len 
        update_timestep = max_ep_len*2
        decay_std_freq =  max_ep_len*2

        # training loop
        while time_step <= max_training_timesteps: # Training loop done, STEP 2 COMPLETED
            # lets start a new episode by reseting env and getting our first observation
            obs = []
            for env_idx0 in range(self.n_envs):
                obsi = self.envs[env_idx0].reset()[0]
                obs.append(obsi)
            current_ep_norm_reward = 0
            current_ep_leap_reward = 0
            # time to initialize our episode loop and collect some experiences
            for t in range(1, max_ep_len+1):
                if t == max_ep_len:
                    print('step limit reached')
                step_reward_norm_sum = 0
                step_reward_leap_sum = 0
                obs = torch.FloatTensor(np.array(obs)).to(device) # obs.shape = (n_envs,obs_dim)
                # we use FloatTensor to avoid possible Runtime errors 
                # lets apply our policy to the state, see what it suggests, and collect the results on rollout
                actions, action_log_prob, parameters, parameter_log_prob, state_value = self.policy_old(obs)  
                # action.shape = action_log_prob = parameter_log_prob = (n_envs,), state_value.shape = (n_envs,1)
                # parameters.shape = (n_envs,num_actions)
                '''
                actions = actions.reshape((self.n_envs,1))
                action_log_prob = action_log_prob.reshape((self.n_envs,1))
                parameter_log_prob = parameter_log_prob.reshape((self.n_envs,1))
                '''
                state_value = state_value.squeeze()
                new_obs = []
                for env_idx in range(self.n_envs):

                    self.buffer.states[env_idx].append(obs[env_idx])
                    self.buffer.actions[env_idx].append(actions[env_idx])
                    self.buffer.param[env_idx].append(parameters[env_idx])
                    self.buffer.action_logprobs[env_idx].append(action_log_prob[env_idx])
                    self.buffer.param_logprobs[env_idx].append(parameter_log_prob[env_idx])
                    self.buffer.state_values[env_idx].append(state_value[env_idx])

                    parameter = torch.abs(parameters[env_idx]) + self.parameter_init
                    param = tuple(parameter.cpu().numpy()) 
                    action = actions[env_idx].item() 
                    total_action = (action,param)
                    # lets apply our action and parameter to the enviroment 
                    state, reward, done, _ = self.envs[env_idx].step(total_action)
                    #print('reward ', reward)
                    new_obs.append(state[0])
                    # appending the results of the policy in the enviroment into the buffer, completing this step's batch
                    self.buffer.rewards[env_idx].append(reward)
                    self.buffer.is_done[env_idx].append(done)
                    if num_envs[1]==0:
                        step_reward_norm_sum+=reward
                    else:
                        if env_idx+1>num_envs[0]:
                            step_reward_leap_sum+=reward
                        else:
                            step_reward_norm_sum+=reward
                obs = new_obs
                time_step +=1
                current_ep_norm_reward += step_reward_norm_sum/num_envs[0]
                if num_envs[1]!=0:
                    current_ep_leap_reward += step_reward_leap_sum/num_envs[1]


                if time_step % update_timestep == 0:
                    print('updating!')
                    for env_idx in range(num_envs[0]):
                        print('norm test')
                        self.envs[env_idx].render()
                        #self.envs[env_idx].save_render_states()
                    for env_idx in range(num_envs[0], num_envs[0]+num_envs[1]):
                        print('leap test')   
                        self.envs[env_idx].render()
                        #self.envs[env_idx].save_render_states()  
                                      
                    self.save(model_name, time_step)
                    self.update()

                if time_step % decay_std_freq == 0:
                    print('decay param std!')
                    self.policy.update_param_std(self.std_decay_rate)

                if time_step % print_freq == 0:

                    # print average reward till last episode
                    print_avg_norm_reward = print_running_norm_reward / print_running_episodes
                    print_avg_norm_reward = round(print_avg_norm_reward, 2)

                    if num_envs[1]!=0:
                        print_avg_leap_reward = print_running_leap_reward / print_running_episodes
                        print_avg_leap_reward = round(print_avg_leap_reward, 2)

                        print("Episode : {} \t\t Timestep : {} \t\t Average Reward(Normal/Leap) : {} \t {}".format(i_episode, time_step, print_avg_norm_reward, print_avg_leap_reward))
                    
                    else:
                        print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_norm_reward))

                        print_running_norm_reward = 0
                        print_running_leap_reward = 0
                        print_running_episodes = 0
                    

                # break; if the episode is over
                if done:
                    break

            print_running_norm_reward += current_ep_norm_reward
            if num_envs[1]!=0:
                print_running_leap_reward += current_ep_leap_reward
            print_running_episodes += 1

            i_episode += 1

# STEP 3 COMPLETED! We have a training loop that collects experiences needed for policy update and evaluation
# STEP 4 and 5 are computing the rewards-on-the-go and advantages respectively

    def calculate_rtgs_n_advantages(self,env_idx): # STEP 4 and STEP 5 COMPLETED
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards[env_idx]), reversed(self.buffer.is_done[env_idx])):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
            
        # we then stack the tensors them to have tensor([1,2,3],[4,5,6],...) instead of [tensor[1,2,3],tensor[4,5,6],...]
        self.old_states = torch.stack(self.buffer.states[env_idx], dim=0).detach().to(device)
        self.old_actions = torch.stack(self.buffer.actions[env_idx], dim=0).detach().to(device)
        self.old_param = torch.stack(self.buffer.param[env_idx], dim=0).detach().to(device)
        self.old_action_logprobs = torch.stack(self.buffer.action_logprobs[env_idx], dim=0).detach().to(device)
        self.old_param_logprobs = torch.stack(self.buffer.param_logprobs[env_idx], dim=0).detach().to(device)
        self.old_state_values = torch.stack(self.buffer.state_values[env_idx], dim=0).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - self.old_state_values.detach()
        return rewards, advantages

# STEP 6: consists in maximizing the clip objective
# Note:   we need for that to calculate the ratio between policies(one for the action policy and another for the parameter policy)  
#         The actor networks share the first few layers to
#         encode the state and each of them generates either a stochastic discrete policy or a stochastic continuous policy. During
#         training, the actors are updated as separate policies
# STEP 7: consists in calculating the loss of the value network through the distance between predicted returns and actual returns

    def update(self):
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            for env_idx in range(self.n_envs):
                advantages, rewards = self.calculate_rtgs_n_advantages(env_idx)

                # Evaluating old actions and values
                action_log_prob, parameter_log_prob, state_values, dist_entropy_action, dist_entropy_param = self.policy.evaluate(self.old_states, self.old_actions, self.old_param)

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
            
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios_action = torch.exp(action_log_prob - self.old_action_logprobs.detach())
                ratios_param = torch.exp(parameter_log_prob - self.old_param_logprobs.detach())

                # Finding Surrogate Loss for action policy 
                surr1_action = ratios_action * advantages
                surr2_action = torch.clamp(ratios_action, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # Finding Surrogate Loss for parameter policy 
                surr1_param = ratios_param * advantages
                surr2_param = torch.clamp(ratios_param, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # final loss of clipped objective PPO
                loss = -torch.min(surr1_action, surr2_action) - torch.min(surr1_param, surr2_param) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy_action - 0.01 * dist_entropy_param
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            
                # Copy new weights into old policy
                self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, model_name,time_step, directory='C:\\Users\\josen\\OneDrive\\Ambiente de Trabalho\\good\\project\\models\\'):
        checkpoint_path = directory + model_name + str(time_step) + ".pth"
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, model_name, directory='C:\\Users\\josen\\OneDrive\\Ambiente de Trabalho\\good\\project\\models\\'):
        checkpoint_path = directory + model_name + ".pth"
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def test(self, num_envs):
        obs = []
        for env_idx0 in range(self.n_envs):
            obsi = self.envs[env_idx0].reset()[0]
            obs.append(obsi)
        current_ep_norm_reward = 0
        current_ep_leap_reward = 0
        max_ep_len = 200
        # time to initialize our episode loop and collect some experiences
        for t in range(1, max_ep_len+1):
            if t == max_ep_len:
                print('step limit reached')
            step_reward_norm_sum = 0
            step_reward_leap_sum = 0
            obs = torch.FloatTensor(np.array(obs)).to(device) # obs.shape = (n_envs,obs_dim)
            # we use FloatTensor to avoid possible Runtime errors 
            # lets apply our policy to the state, see what it suggests, and collect the results on rollout
            actions, action_log_prob, parameters, parameter_log_prob, state_value = self.policy_old(obs)  
            # action.shape = action_log_prob = parameter_log_prob = (n_envs,), state_value.shape = (n_envs,1)
            # parameters.shape = (n_envs,num_actions)
            state_value = state_value.squeeze()
            new_obs = []
            for env_idx in range(self.n_envs):
                    parameter = torch.abs(parameters[env_idx]) + self.parameter_init
                    param = tuple(parameter.cpu().numpy()) 
                    action = actions[env_idx].item() 
                    total_action = (action,param)
                    # lets apply our action and parameter to the enviroment 
                    state, reward, done, _ = self.envs[env_idx].step(total_action)
                    #print('reward ', reward)
                    new_obs.append(state[0])
                    # appending the results of the policy in the enviroment into the buffer, completing this step's batch
                    if env_idx+1>num_envs[0]:
                        step_reward_norm_sum+=reward
                    else:
                        step_reward_leap_sum+=reward
                    obs = new_obs
                    current_ep_norm_reward += step_reward_norm_sum/num_envs[0]
                    if num_envs[1]!=0:
                        current_ep_leap_reward += step_reward_leap_sum/num_envs[1]
  
            if done:
                    print("Average Reward(Normal/Leap) : {} \t {}".format(current_ep_norm_reward, current_ep_leap_reward))
                    for env_idx in range(num_envs[0]):
                        print('norm test')
                        self.envs[env_idx].render()
                        #self.envs[env_idx].save_render_states()
                    for env_idx in range(num_envs[0], num_envs[0]+num_envs[1]):
                        print('leap test')   
                        self.envs[env_idx].render()
                        #self.envs[env_idx].save_render_states()
                    break
















