import gymnasium as gym
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from gymnasium import wrappers
from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim",type=int,default=128)
    parser.add_argument("--num-episodes",type=int,default=1500)
    parser.add_argument("--lr",type=float,default=2e-4)
    parser.add_argument("--gamma",type=float,default=0.95)
    parser.add_argument("--env-name",type=str,default="CartPole-v0")
    parser.add_argument("--device",type=str,
        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_known_args()[0]

class PolicyNet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(PolicyNet,self).__init__()
        self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x),dim=1)

class REINFORCE():
    def __init__(self,state_dim,hidden_dim,action_dim,lr,gamma,device):
        self.PolicyNet = PolicyNet(state_dim,hidden_dim,action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.PolicyNet.parameters(),lr=lr)
        self.gamma = gamma
        self.device = device

    def take_action(self,state):
        state = torch.tensor([state],dtype=torch.float).to(self.device)
        probs = self.PolicyNet(state)
        action_dist = torch.distributions.Categorical(probs) # 生成一个分布,用于采样,probs是概率
        action = action_dist.sample()
        return action.item()

    def update(self,transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]],dtype=torch.long).view(-1,1).to(self.device)

            log_prob = torch.log(self.PolicyNet(state).gather(1,action))
            G = self.gamma * G + reward
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()
        return loss

def train(args=get_args()):
    env = gym.make(args.env_name)

    random_seed = 0
    np.random.seed(0)
    torch.manual_seed(0)
    env.reset(seed=0)
    #init
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n
    #实例化
    agent = REINFORCE(state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        lr=args.lr,
        gamma=args.gamma,
        device=args.device
    )

    return_list = []
    #draw process
    for i in range(10):
        log_dir = f"ReinforcementLearning/classic/runs/log/gmmma_{args.gamma} lr_{args.lr}/epoch_{i}"
        writer = SummaryWriter(log_dir=log_dir)
        with tqdm(total=(args.num_episodes/10),desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(args.num_episodes/10)):
                episode_return = 0
                state = env.reset()
                state = state[0]
                done = False
                transition_dict = {
                    'states':[],
                    'actions':[],
                    'rewards':[],
                    'next_states':[],
                    'dones':[]
                }
                while not done:
                    action = agent.take_action(state)
                    next_state,reward,done, _ , _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['rewards'].append(reward)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                episode_loss = agent.update(transition_dict)
                return_list.append(episode_return)

                writer.add_scalar("REINFORCE-Return",episode_return,i_episode)
                writer.add_scalar("REINFORCE-Loss",episode_loss,i_episode)
                #draw
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':'%d' % (args.num_episodes/10 * i + i_episode + 1),
                        'return':'%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
if __name__ == "__main__":
    train()