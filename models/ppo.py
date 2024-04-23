import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class PolicyNet(nn.Module):  # 输入当前状态，输出动作的概率分布
    def __init__(self, n_states, n_hiddens, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, n_actions)
    def forward(self, x):  # [b,n_states]
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc3(x)  # [b,n_hiddens]-->[b,n_actions]
        x = F.softmax(x, dim=1)  # 每种动作选择的概率
        return x

# ----------------------------------------- #
# 价值网络--critic
# ----------------------------------------- #

class ValueNet(nn.Module):  # 评价当前状态的价值
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 1)
    def forward(self, x):  # [b,n_states]
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc3(x)  # [b,n_hiddens]-->[b,1]
        return x
    
class PPO:
    def __init__(self,n_states,n_hiddens,n_actions,actor_lr,critic_lr,lmbda,eps,gamma,device) -> None:
        self.n_hiddens = n_hiddens
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.lmbda = lmbda
        self.eps = eps
        self.gamma = gamma
        self.device = device
        self.policy_net = PolicyNet(n_states,n_hiddens,n_actions).to(device)
        self.value_net = ValueNet(n_states,n_hiddens).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(),lr=self.actor_lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(),lr=self.critic_lr)
    
    def take_action(self,state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)  # [1,n_states]
        probs = self.actor(state)  # 当前状态的动作概率 [b,n_actions]
        action_dist = torch.distributions.Categorical(probs)  # 构造概率分布
        action = action_dist.sample().item()  # 从概率分布中随机取样 int
        return action
    
    def update(self,transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).to(self.device)

        # 价值网络

        next_states_value = self.value_net(next_states)
        td_target = rewards + self.gamma * next_states_value * (1 - dones)
        td_value = self.value_net(states)
        td_delta = td_target - td_value

        # GAE 
        advantages = 0
        advantages_list = []
        td_delta = td_delta.detach().cpu().numpy()
        for delta in td_delta[::-1]:
            advantages = self.gamma * self.lmbda * advantages + delta[0]
            advantages_list.append([advantages])
        advantages_list.reverse()
        advantages = torch.tensor(advantages_list,dtype=torch.float).to(self.device)
    
        # 策略网络
        old_log_probs = torch.log(self.policy_net(states).gather(1,actions))
        log_probs = torch.log(self.policy_net(states).gather(1,actions))
        ratio =log_probs / old_log_probs

        #clip截断
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio,1-self.eps,1+self.eps) * advantages

        #loss 
        actor_loss = -torch.mean(torch.min(surr1,surr2))
        critic_loss = torch.mean(F.mse_loss(td_target,td_value))

        #update
        self.policy_optimizer.zero_grad()
        self.value_net.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()
        