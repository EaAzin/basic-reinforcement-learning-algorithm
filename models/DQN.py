import os
import argparse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import gymnasium as gym
from gymnasium import wrappers
import numpy as np
import collections
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

#参数
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim",type=int,default=128)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--target-update",type=int,default=10)
    parser.add_argument("--buffer-size",type=int,default=10000)
    parser.add_argument("--minimal-size",type=int,default=500)
    parser.add_argument("--num-episodes",type=int,default=500)
    parser.add_argument("--per-record",type=int,default=98)
    parser.add_argument("--lr",type=float,default=2e-3)
    parser.add_argument("--gamma",type=float,default=0.9)
    parser.add_argument("--epsilon",type=float,default=0.01)
    parser.add_argument("--env-name",type=str,default="CartPole-v0")
    parser.add_argument("--reward-threshold",type=float,default=None)
    parser.add_argument("--device",type=str,
        default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_known_args()[0]

#DL网络
class Qnet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Qnet,self).__init__()
        self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

#ReplayBuffer
class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer = collections.deque(maxlen=capacity)
        
    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

    def sample(self,batch_size):
        transitions = random.sample(self.buffer,batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

#Q-learning
class DQN:
    '''
    DLNetwork:
        state_dim,
        hidden_dim,
        action_dim,
    Off-policy:
        count,
        target_update,
    ReplayBuffer:
        capacity:batch_size,
    Training:
        Adam:lr,
        gamma:gamma,
        device:device,
    epsilon-greedy:
        epsilon
    '''
    def __init__(self,state_dim,hidden_dim,action_dim,target_update,
        batch_size,lr,gamma,epsilon,device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim,hidden_dim,self.action_dim).to(device)
        self.target_q_net = Qnet(state_dim,hidden_dim,self.action_dim).to(device)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr)
        self.gamma = gamma
        self.target_update = target_update
        self.epsilon = epsilon
        self.device = device
        self.count = 0

    def take_action(self,state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array([state]),dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    
    def update(self,transition_dict):
        states= torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1,actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1,1)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values,q_targets))

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict()
            )
        self.count += 1

#Trainning
def train(args=get_args()):
    '''
    Replaybuffer:
        minimal_size
    EarlyStopping:
        reward_threshold
    TensorBoard:
        summarywriter
    '''
    env = gym.make(args.env_name,render_mode='rgb_array')
    env = wrappers.RecordVideo(env=env,
        video_folder='/tmp/exp-1',
        episode_trigger=lambda i_episcode: i_episcode % args.per_record == 0 )
    #random
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env.reset(seed=0)
    #init
    writer = SummaryWriter()
    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n
    if args.reward_threshold is None:
        args.reward_threshold = env.spec.reward_threshold
    #实例化
    replay_buffer = ReplayBuffer(args.buffer_size)
    agent = DQN(state_dim=args.state_dim,
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim, 
        lr=args.lr,
        gamma=args.gamma, 
        epsilon=args.epsilon,
        target_update=args.target_update,
        device=args.device,
        batch_size=args.batch_size
    )
        
    return_list = []

    #draw process
    for i in range(10):
        with tqdm(total=int(args.num_episodes / 10),desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(args.num_episodes / 10)):
                #init
                episode_return = 0
                state = env.reset()
                state = state[0]
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ , _ = env.step(action)
                    replay_buffer.add(state,action,reward,next_state,done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > args.minimal_size:
                        b_s,b_a,b_r,b_ns,b_d = replay_buffer.sample(args.batch_size)
                        transition_dict = {
                            'states':b_s,
                            'actions':b_a,
                            'rewards':b_r,
                            'next_states':b_ns,
                            'dones':b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                writer.add_scalar("Return",episode_return,args.num_episodes/10 * i + i_episode + 1)
                #draw
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':'%d' % (args.num_episodes/10 * i + i_episode + 1),
                        'return':'%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    writer.close()
    env.close()

if __name__ == "__main__":
    train()

