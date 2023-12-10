源码地址:https://github.com/EaAzin/DQN

---

# 1.环境安装说明

几个关键的包

- gymnasium:前身是gym，gym作为openai开发的平台，为强化学习算法测试提供了便捷条件，在版本不断更新的过程中，gym从0.21开始有了较大的改动，使得先前很多没有更新代码的项目使用出现bug，从2021年开始，gym已经迁移到了gymnasium
- tqdm:tqdm可用于在控制台绘制进度条跟踪，清楚展示运行过程
- torch:本文使用pytorch实现Q网络，注意的是请检查安装的torch是否支持cuda

# 2.Q-learning与DQN的原理讲解

下面将Q-learning和DQN对比进行介绍，以便更好的了解他们之间的联系和区别。

## 共同点一：两者都基于价值，处理离散**动作**空间

强化学习的目标是使得累计价值最大，而作为value-based经典方法的Q-learning和DQN，实现目标的方法是通过学习一个能使得动作价值函数Q(s,a)最大化的最优策略，表示在长期价值上回报的期望最佳。由于两者都基于贝尔曼最优方程，该方程描述了最优策略下动作值的更新关系，公式如下:


$$
Q^∗(s,a)=R(s,a)+γmax_{a′}Q^∗(s′,a′)
$$


为了表示使得动作价值函数Q（s,a）最大化的最优策略，可通过贪心策略对最优动作的选择进行实现，公式如下:
$$
π(s)=argmax_a​Q(s,a)
$$


对于该公式有如下两个解读

- 处理离散情景:该函数将对每一个可能动作分配一个值，因此仅适用于动作空间连续的情况，所以Q-learning和DQN是处理离散动作空间的方法
- 基于价值：Q-learng和DQN是以贝尔曼最优方程为依据更新策略的value-based算法，该方程从贪心的角度说，智能体会始终倾向于选择当前最优动作。然而，如果智能体始终选择当前最优动作，很可能过早得出一个局部最优策略。因此，为了使智能体在探索和利用之间取得平衡，可以采用epsilon-greedy策略，即——以 \epsilon的概率随机选择动作(小概率)，以 1-epsilon的概率选择当前最优动作(大概率)

## 不同点一：Q-learning要求动作和状态空间均离散，DQN可以还可处理连续空间，离散动作的场景

Q-learning以时序差分（Temporal Difference,TD）的思路利用Bellman最优方程更新价值函数，更新公式如下:
$$
Q(s,a)←Q(s,a)+α[R+γmax_{a′}Q(s′,a′)−Q(s,a)]
$$
Q-learning以动作和状态为自变量能计算出精确的Q值，但这也限制了它只能适用于动作和状态空间都是离散的场景，DQN则运用神经网络强大的表示能力代替了Bellman最优方程的计算，该拟合Bellman最优方程的网络称之为Q网络

## 共同点二：Q-learing和DQN都是Off-policy 算法

下面是SARSA算法和Q-learing算法的公式对比:
$$
SARSA:Q(s,a)←Q(s,a)+α[R+γQ(s′,a′)−Q(s,a)]
$$

$$
Q-learning:Q(s,a)←Q(s,a)+α[R+γmax_{a′}Q(s′,a′)−Q(s,a)]
$$

sarsa和q-learning均采用时序差分(Temporal Difference,TD)单步更新Q值，两者都采用epsilon-greedy选取下一步动作a′。区别在于，sarsa直接将选取的动作进行Q值更新。而Q-learning则需要多选择一些动作，计算多个Q值，并选择出最佳的动作。因此sarsa是on-policy算法，其行为策略epsilon-greedy和目标策略（更新Q值）的动作选取是一致的。Q-learning为off-policy算法，行为策略epsilon-greedy选取的动作将挑选出最佳动作再进行目标策略的更新。

DQN基于off-policy的思路在Q-learning的基础上做了如下改进：

- 经验回放：原本Q-learning只是记录该状态下的Q值，选出Q值最佳的动作后数据就弃用了。然而DQN将这些数据保存起来，存储与经验回放缓存区中，每一个训练步骤，智能体从这个缓存区中随机抽取一批样本用于学习。数据的充分利用不仅提高了利用率，也打破了样本间的时序相关性，使得经验回放的样本是独立的。（联想：作为Online learning方法，保持了样本独立的特点）

>### 为什么打破样本的时序相关性，保证Online learning场景下的DQN其样本独立有助于学习的稳定性？
>
>学习的稳定性可以从方差的角度进行说明
>
>- 独立，意味着独立随机变量之间协方差为零，有效减小方差大小
>- 在线学习与环境交互，复杂环境不一定是一尘不变的，面对MDP场景下状态转移函数可能改变的情况，如果样本更多表示自身特点而非时序相关性，在轻微改变的环境里表现可能依旧很好，提升泛化能力

>### 为什么使用先前的数据不破坏马尔可夫决策过程的假设？
>
>马尔可夫决策过程（MDP）指的不是当前状态之和上一个状态有关与再之前无关，而是指上一个状态概括了历史的全部信息，因此先前的数据也作为描述环境的内容。而且DQN作为online learning，如果过去的经验不能很好的描述环境，也可通过当前与环境交互去反馈得到这样的信息。而离线学习由于不和环境交互，使用先前的数据可能就会导致动作价值函数估计的偏差（离线学习的关键问题之一），因此经验回放既可以保证样本的独立，又可以用DQN在线学习的优势弥补其缺点。采用经验回放并不破坏MDP

- 目标网络：在Q-learning的基础上，同时训练Q-net和target-Q-net，DQN采用深度学习架构在每次利用 Q-net 计算多组动作对应 Q 值之后，将主网络 Q-net 的最优结果参数复制给目标网络 target-Q-net , 保证了训练的稳定性

>### DQN的损失函数是如何设计的，为什么这样设计可以使得网络能近似计算出Q值？
>
>TQ（target_Q_net）是为了使得训练过程更为稳定，抛开TQ不谈在训练过程中的作用不谈，DQN的损失函数设计如下：
>$$
>Loss=E[((Q(s,a;θ)−(r+γmax_{a′}Q(s′,a′;θ^-)))^2)]
>$$
>其中：
>
>- $Q(s,a;θ)$是当前 Q 网络对于状态 s和动作 a 的估计 Q 值，θ 是网络的参数
>- r 是在状态 s执行动作 a 后获得的即时奖励
>- γ 是折扣因子，表示未来奖励的衰减
>- $max⁡_{a′}Q(s′,a′ ;θ^-)$ 是目标 Q 网络对于下一个状态 s′ 选择的动作的 Q 值的最大估计，其中 θ 是目标网络的参数
>
>DQN利用均方误差对Q-learning的更新公式进行改写，使用梯度下降算法对Q进行反向传播，此时，可将Q网络的训练看作一个**监督学习任务**,该网络的目标是逼近Bellman方程的目标值
>
>1.前向传播
>
>- 输入：当前状态 s 和动作 a
>- 输出：网络产生的Q值估计
>- 目的：通过网络前向传播获得对当前状态和动作Q值的估计
>
>2.计算损失：
>
>- 目标值 Y(s,a)的计算：通过Bellman最优方程计算Q值，其中$ Y(s,a)=r+γ\space max_{⁡a′}Q(s′,a′)$
>- 损失函数：采用均方误差（MSE）作为损失函数，即 $MSE=\frac12(Y(s,a)−Q(s,a))2$
>
>虽然我们不知道真实的Bellman最优方程是什么样的，但是如果我们的网络估值是正确的，那么本轮训练就能获得更多数据点Q（s,a）(任务能继续进行下去)，在这个角度上来说，获得累计期望汇报较高的轮次，它训练出了更好的策略，说明其Bellman最佳方程拟合效果更好。

# 3.代码实现与解读

下面我们对代码关键部分实行实现和解读，DQN的实现思路借鉴了【1】动手学强化学习 中的代码，将关键部分分为 replybuffer（经验回放），DQN（torch框架实现），Q-net（torch框架实现）

```python
class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
```

- init:此处使用双向队列deque作为buffer的容器,其最大的有点在于，deque在限定长度后继续使用append()，头部的元素会自动弹出，不会发生报错，这是deque容器特有的
- size:训练时我们需要让经验池现有一定数量的样本，之后才开始启用，因此我们需要知道当前deque的长度
- **sample**:transitions是buffer中随机抽取的样本，单个元素为五元组的形式，可采用zip*(transitions)对每个列进行提取

```python
class Qnet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Qnet,self).__init__()
        self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

一个最简单的Q-net，由单层神经网络组成

```python
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
       #update将在讲解了训练过程之后再进行讲解，以便理解
```

- init: 对DQN进行解耦，说明了哪些模块需要哪些参数，count与target_update用于off-policy的更新，count作为计数器，达到target_update时说明该更新目标网络参数了
- take_action:用随机数实现epsilon-greedy策略，self.q_net(state).argmax().item()返回了结果中近似估计最大 Q 值的索引，索引对应不同action

```python
def train(args=get_args()):
    '''
    Replaybuffer:
        minimal_size
    '''
	#省略环境env的创建，智能体的示例化的过程
    return_list = []
    #draw process
    for i in range(10):
        with tqdm(total=int(args.num_episodes / 10),desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(args.num_episodes / 10)):
                #init
                episode_return = 0
                state = env.reset()
                state = state[0]
                done = False #布尔值，标识是否为本轮学习的最后一次
                while not done:
                    action = agent.take_action(state) #epsilon-greedy策略选择动作
                    next_state, reward, done, _ , _ = env.step(action) #根据该动作，获取当前的reward,以及next_state,用于计算Q值
                    replay_buffer.add(state,action,reward,next_state,done)#将单步记录至replaybuffer中
                    state = next_state
                    episode_return += reward
                    #replaybuffer没有达到minimal_size时，不更新网络，只收集数据
                    #达到后，将replay_buffer中的值进行采样并用于更新agent
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
                #draw
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':'%d' % (args.num_episodes/10 * i + i_episode + 1),
                        'return':'%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    env.close()
```

下面详细讲解update()函数:

```python
 class DQN：
    def update(self,transition_dict):
        states= torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)
			
        #q_values获取actions下的估计Q值
        q_values = self.q_net(states).gather(1,actions)
        #max_next_q_values获取经验回放池中采样的其中最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1,1)
		#利用Bellman最优公式得出当前actions下的最优Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        #mse-loss:使得q_net下的估计Q值与q_target_net最优Q值相近
        dqn_loss = torch.mean(F.mse_loss(q_values,q_targets))

        self.optimizer.zero_grad() #显式禁止存储梯度，防止显存爆炸
        dqn_loss.backward()
        self.optimizer.step()
		#当更新一定量q_net后将其参数复制到target_q_net
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict()
            )
        self.count += 1
```

# 4.结果展示

环境使用gym下的CartPole-v0，效果如下：

<video src="D:\codePrograms\ReinforcementLearning\DQN\video\exp-2\rl-video-episode-200.mp4"></video>

思考：既然有的轮次训练的累计期望奖励更大，其策略更优，估计Q值的表示能力更精确。是否能在该网络参数的基础上重新训练整个过程？
