import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    """"K表示拉杆个数，多臂老虎机"""
    def __init__(self,K):
        self.probs = np.random.uniform(size=K) #K个拉杆获奖
        self.best_idx = np.argmax(self.probs) #最大获奖概率
        self.best_prob = self.probs[self.best_idx]

        self.K = K

    def step(self,k):
        #当玩家选择第k个拉杆后，返回概率
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0

class Solver:
    "多臂老虎机基本框架"
    def __init__(self,bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)
        self.regret = 0 #当前的累计悔意
        self.actions = []
        self.regrets = []

    def update_regret(self,k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        #搜索算法的选择
        raise NotImplementedError

    def run(self,num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

class EpslionGreedy(Solver):
    def __init__(self,bandit,epsilon = 0.01, init_prob = 1.0):
        super(EpslionGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        #初始化所有拉动拉杆的奖励初值
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0,self.bandit.K)
        else:
            k = np.argmax(self.estimates) #选择期望最大的估值
        r = self.bandit.step(k)
        self.estimates[k] += 1./ (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class DecayingEpsilonGreedy(Solver):
    def __init__(self,bandit,epsilon = 0.01, init_prob = 1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        #初始化所有拉动拉杆的奖励初值
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1/self.total_count:
            k = np.random.randint(0,self.bandit.K)
        else:
            k = np.argmax(self.estimates) #选择期望最大的估值
        r = self.bandit.step(k)
        self.estimates[k] += 1./ (self.counts[k] + 1) * (r - self.estimates[k])
        return k

def plot_results(solvers,solver_names):
    for idx,solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list,solver.regrets,label = solver_names[idx])
    plt.xlabel("time steps")
    plt.ylabel('cumulative regrets')
    plt.title("%d-armed bandit" % solvers[0].bandit.K)
    plt.legend()
    plt.show()
#上界信界
class UCB(Solver):
    def __init__(self,bandit,coef,init_prob=1.0):
        super(UCB,self).__init__(bandit)
        self.total_count = 0
        self.estimate = np.array([init_prob] * self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimate + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimate[k] += 1./ (self.counts[k] + 1) * (r - self.estimate[k])
        return k
#汤普森
class ThompsonSampling(Solver):
    def __init__(self,bandit):
        super(ThompsonSampling,self).__init__(bandit)
        self._a = np.ones(self.bandit.K)
        self._b = np.ones(self.bandit.K)

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)
        k = np.argmax(samples)
        r = self.bandit.step(k)

        self._a[k] += r
        self._b[k] += (1 - r)
        return k

np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)
#e-g
epsilons = [1e-4,0.01,0.1,0.25,0.5]
epsilon_greedy_solver_list = [
    EpslionGreedy(bandit_10_arm,e) for e in epsilons
]
epsilon_greedy_solver_names = ["epsilon = {}".format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)
plot_results(epsilon_greedy_solver_list,epsilon_greedy_solver_names)
#de-g
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
print('decaying-epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])
#ucb
coef = 1
UCB_solver = UCB(bandit_10_arm,coef)
UCB_solver.run(5000)
print('上置信界算法的累积懊悔为：', UCB_solver.regret)
plot_results([UCB_solver], ["UCB"])

thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver], ["ThompsonSampling"])