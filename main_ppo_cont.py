import gym
import numpy as np
import pybulletgym
from gym.wrappers import Monitor
from ppo_continuous_gae import Agent

env_name = "HalfCheetahPyBulletEnv-v0"
env = gym.make(env_name)
env.reset()

input_dims = env.observation_space.shape[0]
output_dims = env.action_space.shape[0]
critic_alpha = 0.0003
actor_alpha = 0.0001
n_update_steps = 1024
minibatch_size = 128
K_epochs = 30
eps_clip = 0.2
gae_lambda = 0.95


num_episodes = 5000
Episode_Returns = []
num_runs = 3
scores = np.zeros((num_runs,num_episodes))

for run in range(num_runs):
    ppo_agent = Agent(input_dims=input_dims,fc1_dims=256,fc2_dims=128,output_dims=output_dims,critic_alpha=critic_alpha,actor_alpha=actor_alpha,
                      eps_clip=eps_clip,K_epochs=K_epochs,n_update_steps=n_update_steps,gae_lambda=gae_lambda,minibatch_size=minibatch_size)
    t = 0
    print(run)
    Episode_Returns = []
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        episode_return = 0

        while not done:
            action = ppo_agent.choose_action(observation)
            next_observation,reward,done,_ = env.step(action)
            ppo_agent.store_transitions(observation,reward,next_observation,done)
            observation = next_observation
            episode_return += reward

            if (t + 1) % n_update_steps == 0:
                ppo_agent.update()

            t += 1

            if done:
                break

        Episode_Returns.append(episode_return)
        print(run + 1,episode,t + 1, episode_return, sum(Episode_Returns[-10:])/len(Episode_Returns[-10:]), sum(Episode_Returns[-100:])/len(Episode_Returns[-100:]))

        scores[run][episode] = sum(Episode_Returns[-100:])/len(Episode_Returns[-100:])

        #if sum(Episode_Returns[-100:])/len(Episode_Returns[-100:]) >= 2500:
        #    break
    np.save('test_'+str(env_name)+"_"+ str(run) + '.npy', scores)
    del(ppo_agent)

env.close()
print()

print(scores)
