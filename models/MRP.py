import numpy as np
np.random.seed(0)
#状态转移概率矩阵P
P = [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
P = np.array(P)

rewards = [-1,-2,-2,10,1,0] #奖励函数定义
gamma = 0.5 #折扣因子

def compute_return(start_index,chain,gamma):
    G = 0
    for i in reversed(range(start_index,len(chain))):
        G = gamma * G + rewards[chain[i] - 1]
    return G
def compute(P,rewards,gamma,states_num):
    '''利用贝尔曼方程矩阵形式计算解析解，states_nums是MRP的状态数'''
    rewards = np.array(rewards).reshape((-1,1))
    value = np.dot(np.linalg.inv(np.eye(states_num,states_num) - gamma * P),rewards)
    return value
V = compute(P,rewards,gamma,6)
print("MRP中每个状态价值分别为\n", V)
#一个状态序列
chain = [1, 2, 3, 6]
start_index = 0
G = compute_return(start_index,chain,gamma)
print("根据序列计算得到的回报为:%s" % G)
