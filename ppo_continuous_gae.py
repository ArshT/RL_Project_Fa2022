import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self,input_dims,fc1_dims,fc2_dims,output_dims,device,log_std_min = -3, log_std_max = 0,log_std_init=-0.5):
        super(Actor, self).__init__()

        self.fc1 = layer_init(nn.Linear(input_dims,fc1_dims))
        self.fc2 = layer_init(nn.Linear(fc1_dims,fc2_dims))
        self.action_final_layer = layer_init(nn.Linear(fc2_dims,output_dims))

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = torch.device(device)
        self.to(self.device)

        log_std = log_std_init * np.ones(output_dims, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.device = torch.device(device)
        self.output_dims = output_dims
        self.to(self.device)


    def forward(self,observation):
        state = torch.Tensor(observation).to(self.device)

        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        action_mean = F.tanh(self.action_final_layer(x))

        action_logstd = torch.clamp(self.log_std,self.log_std_min,self.log_std_max)

        return action_mean,action_logstd



class Critic(nn.Module):
    def __init__(self,input_dims,fc1_dims,fc2_dims,device):
        super(Critic, self).__init__()

        self.fc1 = layer_init(nn.Linear(input_dims,fc1_dims))
        self.fc2 = layer_init(nn.Linear(fc1_dims,fc2_dims))
        self.value_final_layer = layer_init(nn.Linear(fc2_dims,1))

        self.device = torch.device(device)
        self.to(self.device)

    def forward(self,observation):
        state = torch.Tensor(observation).to(self.device)

        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        state_value = self.value_final_layer(x)

        return state_value



class ActorCritic(nn.Module):
    def __init__(self,input_dims,fc1_dims,fc2_dims,output_dims,device='cpu',log_std_min = -3, log_std_max = 0,log_std_init=-0.5):
        super(ActorCritic, self).__init__()

        self.actor = Actor(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,output_dims=output_dims,
                           device=device,log_std_min=log_std_min,log_std_max=log_std_max,log_std_init=log_std_init)

        self.critic = Critic(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,device=device)


    def forward(self,state):

        action_mean,action_logstd = self.actor(state)
        state_value = self.critic(state)

        return action_mean,action_logstd,state_value



class Agent:
    def __init__(self,input_dims,fc1_dims,fc2_dims,output_dims,actor_alpha,critic_alpha,K_epochs,eps_clip,n_update_steps,gae_lambda,minibatch_size,gamma=0.99,device='cpu'):
        self.output_dims = output_dims
        self.input_dims = input_dims
        self.device = torch.device(device)
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.minibatch_size = minibatch_size
        self.n_update_steps = n_update_steps

        self.AC = ActorCritic(input_dims=input_dims,fc1_dims=fc1_dims,fc2_dims=fc2_dims,output_dims=output_dims,device=device,log_std_init=-0.1)
        self.optimizer = optim.Adam([
        {"params": self.AC.actor.parameters(),"lr": actor_alpha},
        {"params": self.AC.critic.parameters(),"lr": critic_alpha}
        ])

        self.memory_counter = 0

        self.logprob_memory = torch.zeros((self.n_update_steps,))
        self.state_memory = torch.zeros((self.n_update_steps,self.input_dims))
        self.next_state_memory = torch.zeros((self.n_update_steps,self.input_dims))
        self.action_memory = torch.zeros((self.n_update_steps,self.output_dims))
        self.reward_memory = []
        self.terminal_memory = []
        self.actor_std = []

    def store_transitions(self,state,reward,next_state,terminal):
        self.reward_memory.append(reward)
        self.terminal_memory.append(1 - int(terminal))

        state_tensor = torch.Tensor(state).to(self.device)
        self.state_memory[self.memory_counter] = state_tensor

        next_state_tensor = torch.Tensor(next_state).to(self.device)
        self.next_state_memory[self.memory_counter] = next_state_tensor

        self.memory_counter += 1

    def choose_action(self,state):
        action_mean,action_logstd,_ = self.AC.forward(state)

        action_mean = action_mean.float()

        action_std = torch.exp(action_logstd)
        self.actor_std.append(torch.mean(action_std))
        action_var = (action_std * action_std).to(self.device)

        cov_mat = torch.diag(action_var).unsqueeze(dim=0).to(self.device)
        cov_mat = cov_mat.float()

        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        self.logprob_memory[self.memory_counter] = action_logprob
        self.action_memory[self.memory_counter] = action

        return action.detach().cpu().numpy().flatten()


    def evaluate(self,states,actions,next_states):
        action_means,action_logstds,state_values = self.AC.forward(states)
        _,_,next_state_values = self.AC.forward(next_states)

        action_stds = torch.exp(action_logstds)
        action_vars = (action_stds * action_stds).to(self.device)
        action_vars = action_vars.expand_as(action_means)

        cov_mat = torch.diag_embed(action_vars).to(self.device)
        dist = torch.distributions.MultivariateNormal(action_means, cov_mat)

        logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()


        return logprobs,torch.squeeze(state_values),torch.squeeze(next_state_values),dist_entropy


    def GAE(self,state_values,next_state_values):

        advantages = torch.zeros(self.n_update_steps,)
        lastgaelam = 0
        for t in reversed(range(self.n_update_steps)):
            delta = (self.reward_memory[t] + self.gamma * self.terminal_memory[t] * next_state_values[t] - state_values[t])
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * self.terminal_memory[t] * lastgaelam

        returns = (advantages + state_values)
        return returns,advantages


    def update(self):
        old_logprobs = self.logprob_memory
        old_states = self.state_memory
        old_actions = self.action_memory
        old_next_states = self.next_state_memory

        reward_batch = torch.Tensor(np.array(self.reward_memory))
        terminal_batch = torch.Tensor(np.array(self.terminal_memory))

        _,old_state_values,old_next_state_values,_ = self.evaluate(old_states,old_actions,old_next_states)
        old_state_values = old_state_values.detach()
        old_next_state_values = old_next_state_values.detach()

        returns,advantages = self.GAE(old_state_values,old_next_state_values)

        batch_indices = np.arange(self.n_update_steps)
        for k in range(self.K_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0,self.n_update_steps,self.minibatch_size):
                end = start + self.minibatch_size
                minibatch_indices = batch_indices[start:end]

                new_logprobs,state_values,next_state_values,dist_entropy = self.evaluate(old_states[minibatch_indices],old_actions[minibatch_indices],old_next_states[minibatch_indices])
                ratios = torch.exp(new_logprobs - old_logprobs[minibatch_indices].detach())

                td_difference = returns[minibatch_indices] - state_values

                surr1 = ratios * advantages[minibatch_indices]
                surr2 = torch.clamp(ratios,1 - self.eps_clip,1 + self.eps_clip) * advantages[minibatch_indices]

                loss = - torch.min(surr1,surr2) + 0.5 * td_difference.pow(2) - 0.003 * dist_entropy

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()



        self.memory_counter = 0
        self.logprob_memory = torch.zeros((self.n_update_steps,))
        self.state_memory = torch.zeros((self.n_update_steps,self.input_dims))
        self.next_state_memory = torch.zeros((self.n_update_steps,self.input_dims))
        self.action_memory = torch.zeros((self.n_update_steps,self.output_dims))
        self.reward_memory = []
        self.terminal_memory = []
        print(sum(self.actor_std)/len(self.actor_std))
        print()
        self.actor_std = []
