# Platform_challenge
## State of the Art
Parameterized action spaces are a complicated but fundamental topic to deal with. The first temptation I felt, imagine that to be the same for many engineers when dealing with the problem for the first time, is to either force it to be discrete or continuous, taking the advantages that led the environment to be described in this way in the first place away: the hierarchic organization of subspaces of complex actions. In "Reinforcement Learning with Parameterized Actions" researchers demonstrated that an approach consisting of value-based and policy gradient methods to deal with the discrete and continuous portion of the action space, respectively, surpassed the direct policy search and Sarsa with fixed parameters approach. This paper is from 2016. Today, Proximal Policy Optimization-based models(PPO) are the most successful algorithms in reinforcement learning. In 2019, the paper "Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space" presented an Actor-Critic approach to the problem. This approach consists in attributing an "actor" to each layer of the action space and a "critic" that critiques them all. It is an approach with many advantages: its simplicity(how intuitive it is), its scalability(an extra layer of the action space can be dealt with by adding an extra actor), and the possibility of incorporating proximal policy optimization to update the policy. To choose an action, one actor gives us the discrete portion, and the other outputs a parameter value for each possible discrete act. Later, we calculate Clip PPO Loss for the discrete and continuous policy and incorporate it with the Critic's loss. The paper compared it's results with the extended DDPG for parameterized action space by Hausknecht and Stone [2016], the P-DQN algorithm [2018], and DQN which first discretizes the parameterized action space.

## Methods
The code was organized according to the steps in OpenAI's Spinning Up's pseudocode for PPO's clipped version(https://spinningup.openai.com/en/latest/algorithms/ppo.html):

![image](https://user-images.githubusercontent.com/90193839/213691509-74084808-298e-4d6f-8a1b-3862f004a05c.png)

I also took inspiration from PyTorch's "Minimal PPO" implementation(https://github.com/nikhilbarhate99/PPO-PyTorch) and the PPO-for-Beginners implementation(https://github.com/ericyangyu/PPO-for-Beginners) described in the series "Coding PPO from Scratch with PyTorch"(https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8). 

Lets start by reading the first step and start the explanation:

![image](https://user-images.githubusercontent.com/90193839/213692086-2080e5c1-f8fb-470e-bd93-777a6c12c81d.png)

It means "initialize your policy neural network". Let's look at our neural network class. I called it "ActorsCritic" because we have two actors and one critic.

![image](https://user-images.githubusercontent.com/90193839/213694890-f575e258-4eb1-457d-b2c6-4c8c9630f853.png)

Our neural network starts with two feature extraction layers that will diverge into heads corresponding to each task in which we will use our neural network: outputting a discrete action, outputting a parameter value for each possible discrete action, and associating a value to the state.Our neural network starts with two feature extraction layers that will diverge into heads corresponding to each task in which we will use our neural network: outputting a discrete action, outputting a parameter value for each possible discrete action, and associating a value to the state. Above, we can see the initialization of the initial values and standard deviations of the parameters distribution.




