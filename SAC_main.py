env_name = "HalfCheetahPyBulletEnv-v0"
env = gym.make(env_name)
input_dims = env.observation_space.shape[0]
action_dims = env.action_space.shape[0]

num_episodes = 1000
fc1_dims=512
fc2_dims=256
batch_size=128
gamma = 0.99
tau = 0.005
buffer_size = 100000
SAC_alpha = 0.5
n_update_iter = 4
actor_alpha = 0.0003
critic_alpha = 0.0003
SAC_alpha_lr = 0.0003
tune_alpha = True
device='cpu'




SAC_agent = Agent(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,action_dims=action_dims,gamma=gamma,tau=tau,
                  SAC_alpha=SAC_alpha,actor_alpha=actor_alpha,critic_alpha=critic_alpha,SAC_alpha_lr=SAC_alpha_lr,
                  batch_size=batch_size,buffer_size=buffer_size,tune_alpha=tune_alpha,n_update_iter=n_update_iter,device=device)

scores = []
Scores_Array = np.zeros(num_episodes,)
for i in range(num_episodes):
  done = False
  score = 0
  observation = env.reset()

  while not done:
    action = SAC_agent.choose_action(observation)
    observation_,reward,done,_ = env.step(action)
    SAC_agent.store_transitions(observation,action,reward,observation_,done)
    score += reward
    observation = observation_

    SAC_agent.update()

  scores.append(score)

  avg_score = np.mean(scores[max(0, i-10):(i+1)])
  avg_score_100 = np.mean(scores[max(0, i-100):(i+1)])
  print('episode: ', i+1,'score: ', score,' average_score_10 %.3f' % avg_score,' average_score_100 %.3f' % avg_score_100)
  Scores_Array[i] = score
  torch
  np.save('SAC_'+str(env_name)+"_"+ str(0) + '.npy', Scores_Array)
  SAC_agent.save_model(i)
  print()


print(Scores_Array)
