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

![image](https://user-images.githubusercontent.com/90193839/214864720-e3ed64fa-615c-4806-ba89-f2258dc69503.png)

Our neural network starts with two feature extraction layers that will diverge into heads corresponding to each task, which means outputting a discrete action, outputting a parameter value for each possible discrete action, and associating a value to the state. Above, we can see the initialization of the initial values and standard deviations of the parameters distribution. These standard deviations will be reduced throughout the training since we want our agent to become more deterministic with time. The choice of the standard deviation decay rate, and initial values, and standard deviation of the parameters are hyperparameters. 

Let us call ActorsCritic to initialize our policy and old policy, respectively, the policy before and after the update( we will need that for the update step to make sure we stay in the trust region). Additionally, we collect our hyperparameter values and process environment's information.

![image](https://user-images.githubusercontent.com/90193839/214865640-e8c5ee3a-60a8-4287-83bf-f557bc127e43.png)

This information will be used to initialize our neural network. The environment observation spaces are a tuple of which the first element consists of platform features+player+enemy features and the second the step number on the episode. I saw no purpose in including the steps, so the observation just includes the first part of the tuple.

We can also notice the attribute "envs" and how the buffer has the length of this attribute as an argument. That's because we are training with multiple environments in parallel to accelerate and stabilize training. The buffer organizes its stored information by the environment to which it belongs.

![image](https://user-images.githubusercontent.com/90193839/213717995-b4174a16-a6cc-4a48-ab75-fda1bb4c91ff.png)

We completed step 1. Step 2 consists in initializing our training loop, and step 3 of collecting the agent's results upon iterating over the environment.

![image](https://user-images.githubusercontent.com/90193839/213719770-9a503791-d909-4e66-9447-1393bb7c11a8.png)

I want to have a "learn" function like the models on stable baselines. The arguments of this function are the total number of steps to run, the name we want to give to the model we are training upon saving(by default, the code stores the model every time it updates and calls it model_name+nº of steps), and something called "num_envs". During my first learning attempts, I observed that the agent, upon learning how to jump over the first enemy, would have the probability of leaping so diminished that it would not have enough exploration potential to learn how to jump between platforms. To deal with this problem, part of the environments in which the agent will train will start from the end of the platform with the enemy in the beginning. This strategy greatly improved training. However, the ratio between the number of default environments and what I called "leap"(short for "leap training") environments is a hyperparameter, as is the total number of environments used. num_envs' first element gives us the number of normal environments, and the second the number of "leap" environments.

![image](https://user-images.githubusercontent.com/90193839/213723863-c6180013-5daa-4f1e-aaaa-ebb9d524ed2c.png)

In the image, you can see how I separate the reward from the normal environments and the ones from the "leap" environments. This is just for monitoring purposes. The rewards are mixed in the buffer because we want the update to improve the agent's performance in both environment types. We want it to learn how to leap to apply it in normal environments. However, what we want to get in the end is an agent that can start at the beginning of the first platform and achieve the end of the third, so we have to monitor that reward separately. 

The image also shows how I use multiple environments. Before the episode starts, I collect the first observation of every environment to introduce it into the neural network. The outputs will come in Tensor form with an element for each environment inside. Let's look at ActorsCritic forward pass to understand how we get these outputs.

![image](https://user-images.githubusercontent.com/90193839/214866798-84aad944-70e6-44a3-8f34-99d3da7cc3bb.png)

We first apply the ReLU activation function with the feature extraction layers we talked about before. The output of these layers will then be received by each of the heads for each distinct purpose. The state value layer will output a value for each environment referent to the current state, a value approximating the future returns. From the "discrete" layer, it will come out a value for each possible action. Softmax will transform those into probabilities. With this list of probabilities(one for each possible action), we will create a probability distribution with PyTorch's Categorical, from which we will sample our chosen action. We then save the logarithm of the probability associated with that action. We then do the equivalent for the continuous head. After we apply our layer, we get a parameter value for each possible discrete action. We then take our currently stored standard deviation and these values and build a MultiVariate Gaussian distribution from which to sample.

We then detach the returns of the function of their graphs. We have the evaluate function of the neural network class for use when it's time for a policy update. After getting the returns from the neural network it's time to apply them to the environments. We do a step in each environment before we move to the following because we have to collect all the observations before we can run the neural network again. We collect it all in the buffer and separate the rewards we talked about.

![image](https://user-images.githubusercontent.com/90193839/214867987-9957940d-533e-4c6d-bedf-c71b074bf716.png)


Steps 4 and 5 are calculating the rewards-on-the-go and the advantages. I perform these operations on the same function. 

![image](https://user-images.githubusercontent.com/90193839/213740930-f68a7242-0f1f-43e7-994f-2e9ff300325a.png)

Finally, step 6 corresponds to updating the actors( to which we are calculating separate clip loss functions and ratios with the same advantage), and step 7 corresponds to updating the critic. 

![image](https://user-images.githubusercontent.com/90193839/213745918-403f810f-162a-4ad8-8e66-cd83359b1da1.png)

We do this in the update function, which gets the neural network graphs through the evaluate function, which does not detach() its returns like the forward pass.


