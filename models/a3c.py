import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
import collections
from gymnasium import wrappers
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import argparse
import torch.multiprocessing as mp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim",type=int,default=128)
    parser.add_argument("--actor-lr",type=float,default=1e-3)
    parser.add_argument("--critic-lr",type=float,default=1e-2)
    parser.add_argument("--gamma",type=float,default=0.98)
    parser.add_argument("--num-episodes",type=int,default=1500)
    parser.add_argument("--env-name",type=str,default="CartPole-v0")
    parser.add_argument("--device",type=str,
        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_known_args()[0]

class Actor(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x),dim=1)

class Crtic(nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super(Crtic,self).__init__()
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class A2C():
    def __init__(self,state_dim,hidden_dim,action_dim,actor_lr,critic_lr,gamma,device):
        self.actor = Actor(state_dim,hidden_dim,action_dim).to(device)
        self.critic = Crtic(state_dim,hidden_dim).to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)#对Actor进行优化
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(),lr=critic_lr)#对Critic进行优化
        self.gamma = gamma
        self.device = device

    def share_memory(self):
        self.actor.share_memory()
        self.critic.share_memory()

    def load_actor_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict)

    def load_critic_state_dict(self, state_dict):
        self.critic.load_state_dict(state_dict)

    #Actor进行决策，仿照reinforce
    def take_action(self,state):
        state = torch.tensor(np.array([state]),dtype=torch.float).to(self.device)
        action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    #对Actor和Critic进行更新
    def update(self,transition_dict,args,global_model):
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)

        #计算时序差分目标
        v_value = self.critic(states)
        next_v_value = self.critic(next_states)
        td_target = rewards + self.gamma * next_v_value * (1 - dones)
        td_delta = td_target - v_value
        #对actor进行优化
        log_probs = torch.log(self.actor(states).gather(1,actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        #对critic进行优化
        critic_loss = torch.mean(F.mse_loss(v_value,td_target.detach()))

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.optimizer_critic.step()
        self.optimizer_actor.step()

        loacal_model = A2C(
            state_dim=args.state_dim,
            hidden_dim=args.hidden_dim,
            action_dim=args.action_dim,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            device=args.device
        )
        
        loacal_model.load_actor_state_dict(global_model.actor.state_dict())
        loacal_model.load_critic_state_dict(global_model.critic.state_dict())

def worker(global_model,i,args):
    #env = gym.make(args.env_name)s
    
    worker_agent = A2C(
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        device=args.device)

    worker_agent.load_actor_state_dict(global_model.actor.state_dict())
    worker_agent.load_critic_state_dict(global_model.critic.state_dict())

    return_list = []
    for i in range(10):
        #log_dir = f"ReinforcementLearning/A3C/runs/gamma_{args.gamma} alr_{args.actor_lr} clr_{args.critic_lr}/_iter_{i}"
        #writer = SummaryWriter(log_dir=log_dir)
        with tqdm(total=int(args.num_episodes/10),desc="Iteration %d" % i ) as pbar:
            for i_episode in range(int(args.num_episodes/10)):
                env = gym.make('CartPole-v0')
                episode_return = 0
                state = env.reset()
                state = state[0]
                done = False
                transition_dict = {
                        "states":[],
                        "actions":[],
                        "rewards":[],
                        "next_states":[],
                        "dones":[],
                    }
                while not done:
                    action = worker_agent.take_action(state)
                    next_state,reward,done, _ , _ = env.step(action)
                    transition_dict["states"].append(state)
                    transition_dict["actions"].append(action)
                    transition_dict["rewards"].append(reward)
                    transition_dict["next_states"].append(next_state)
                    transition_dict["dones"].append(done)

                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                worker_agent.update(transition_dict,args,global_model)

                #draw
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':'%d' % (args.num_episodes/10 * i + i_episode + 1),
                        'return':'%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    #env = gym.make(args.env_name,render_mode='rgb_array')
    #env = wrappers.RecordVideo(env=env,video_folder='/tmp/exp-4',episode_trigger= lambda i_episcode: i_episcode % 98 == 0 )

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env.reset()

    args = get_args()
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n

    global_model = A2C( 
        state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        device=args.device
        )

    global_model.share_memory()
    mp.set_start_method('spawn')
    cpu_count = mp.cpu_count()
    print("cpu_count %d" % cpu_count)
    processes = []
    for i in range(4):
        p = mp.Process(target=worker, args=(global_model,i,args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join() 