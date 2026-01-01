"""
强化学习的主要形式及其关系

本文件详细介绍强化学习的主要分类方法，并通过代码演示不同形式的RL算法。
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simple_environment import GridWorld


# ============================================================================
# 强化学习分类体系
# ============================================================================

def print_rl_taxonomy():
    """打印强化学习的分类体系"""
    print("\n" + "="*80)
    print("强化学习的主要分类")
    print("="*80)

    taxonomy = """
强化学习算法可以从多个维度进行分类：

1️⃣  按是否建立环境模型分类：
   ├─ Model-Free（无模型）：不学习环境动态模型
   │  └─ 优势：简单、通用
   │  └─ 劣势：样本效率低
   │  └─ 代表：Q-Learning, SARSA, Policy Gradient
   │
   └─ Model-Based（有模型）：学习环境的状态转移和奖励函数
      └─ 优势：样本效率高、可规划
      └─ 劣势：模型误差会累积
      └─ 代表：Dyna-Q, MCTS, MuZero

2️⃣  按策略更新方式分类：
   ├─ On-Policy（同策略）：学习和行动使用同一个策略
   │  └─ 优势：稳定性好
   │  └─ 劣势：样本利用率低
   │  └─ 代表：SARSA, A2C
   │
   └─ Off-Policy（异策略）：学习的策略和行动的策略不同
      └─ 优势：样本利用率高、可以学习专家数据
      └─ 劣势：可能不稳定
      └─ 代表：Q-Learning, DQN, DDPG

3️⃣  按学习内容分类：
   ├─ Value-Based（基于价值）：学习状态或动作的价值函数
   │  └─ 学习：V(s) 或 Q(s,a)
   │  └─ 策略：隐式（从价值函数导出）
   │  └─ 代表：Q-Learning, DQN, SARSA
   │
   ├─ Policy-Based（基于策略）：直接学习策略函数
   │  └─ 学习：π(a|s)
   │  └─ 优势：可以处理连续动作、随机策略
   │  └─ 代表：REINFORCE, PPO, TRPO
   │
   └─ Actor-Critic（演员-评论家）：同时学习策略和价值函数
      └─ Actor：策略网络（选择动作）
      └─ Critic：价值网络（评估动作）
      └─ 代表：A3C, SAC, TD3

4️⃣  按更新方式分类：
   ├─ Monte Carlo（蒙特卡洛）：使用完整回合的回报
   │  └─ 优势：无偏估计
   │  └─ 劣势：高方差、需要完整回合
   │
   ├─ Temporal Difference（时间差分）：使用单步或多步回报
   │  └─ 优势：低方差、在线学习
   │  └─ 劣势：有偏估计
   │
   └─ TD(λ)：结合MC和TD
      └─ 使用资格迹（Eligibility Trace）
"""
    print(taxonomy)

    print("\n" + "="*80)
    print("主要算法关系图")
    print("="*80)
    relationship = """
                    强化学习算法
                         |
        ┌────────────────┼────────────────┐
        │                │                │
    Value-Based     Policy-Based    Actor-Critic
        │                │                │
    ┌───┴───┐        ┌───┴───┐       ┌───┴───┐
    │       │        │       │       │       │
  Q-Learning  SARSA  REINFORCE PPO   A3C    SAC
    │       │        │       │       │       │
 Off-Policy On-Policy                │       │
                              Actor + Critic

特点对比：
╔═══════════════╦═══════════════╦═══════════════╦═══════════════╗
║   算法类型    ║  学习目标     ║  策略类型     ║  更新方式     ║
╠═══════════════╬═══════════════╬═══════════════╬═══════════════╣
║ Q-Learning    ║ Q(s,a)        ║ Off-Policy    ║ TD            ║
║ SARSA         ║ Q(s,a)        ║ On-Policy     ║ TD            ║
║ REINFORCE     ║ π(a|s)        ║ On-Policy     ║ Monte Carlo   ║
║ Actor-Critic  ║ π(a|s) + V(s) ║ On-Policy     ║ TD            ║
╚═══════════════╩═══════════════╩═══════════════╩═══════════════╝
"""
    print(relationship)


# ============================================================================
# 1. Model-Free: Q-Learning（无模型、异策略、基于价值）
# ============================================================================

class QLearning:
    """
    Q-Learning算法

    分类：
    - Model-Free：不需要环境模型
    - Off-Policy：学习最优策略，但用ε-贪心探索
    - Value-Based：学习Q值函数
    - TD Learning：使用时间差分更新

    更新公式：
    Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
                                  ↑
                        使用最优动作（贪心）
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha      # 学习率
        self.gamma = gamma      # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.n_actions = n_actions

    def choose_action(self, state):
        """ε-贪心策略选择动作"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # 探索
        return np.argmax(self.q_table[state])          # 利用

    def update(self, state, action, reward, next_state, done):
        """
        Q-Learning更新（Off-Policy）

        关键：使用max Q(s',a')，即下一状态的最优动作
        这与实际执行的动作可能不同（Off-Policy）
        """
        if done:
            target = reward
        else:
            # 使用贪心策略选择下一动作（即使实际可能用ε-贪心）
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # TD更新
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error


# ============================================================================
# 2. Model-Free: SARSA（无模型、同策略、基于价值）
# ============================================================================

class SARSA:
    """
    SARSA算法（State-Action-Reward-State-Action）

    分类：
    - Model-Free：不需要环境模型
    - On-Policy：学习当前执行的策略
    - Value-Based：学习Q值函数
    - TD Learning：使用时间差分更新

    更新公式：
    Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
                                  ↑
                      使用实际执行的动作（可能是探索的）
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions

    def choose_action(self, state):
        """ε-贪心策略"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA更新（On-Policy）

        关键：使用Q(s',a')，即实际将要执行的下一动作
        这与选择动作的策略一致（On-Policy）
        """
        if done:
            target = reward
        else:
            # 使用实际选择的下一动作
            target = reward + self.gamma * self.q_table[next_state, next_action]

        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error


# ============================================================================
# 3. Model-Based: Dyna-Q（有模型、异策略、基于价值）
# ============================================================================

class DynaQ:
    """
    Dyna-Q算法

    分类：
    - Model-Based：学习环境模型
    - Off-Policy：基于Q-Learning
    - Value-Based：学习Q值函数
    - Planning + Learning：结合真实经验和模拟经验

    核心思想：
    1. 从真实经验中学习（Direct RL）
    2. 学习环境模型（Model Learning）
    3. 使用模型生成模拟经验（Planning）
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9,
                 epsilon=0.1, planning_steps=5):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.planning_steps = planning_steps

        # 环境模型
        self.model = {}  # {(s,a): (r, s')}
        self.visited_states = set()

    def choose_action(self, state):
        """ε-贪心策略"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """
        Dyna-Q更新

        包含三步：
        1. Direct RL：从真实经验更新Q值
        2. Model Learning：更新环境模型
        3. Planning：使用模型进行规划
        """
        # 1. Direct RL（与Q-Learning相同）
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

        # 2. Model Learning：记录环境动态
        self.model[(state, action)] = (reward, next_state, done)
        self.visited_states.add(state)

        # 3. Planning：使用模型进行规划
        for _ in range(self.planning_steps):
            # 随机选择之前访问过的状态和动作
            if not self.visited_states:
                break

            s = np.random.choice(list(self.visited_states))
            # 随机选择一个在模型中的动作
            available_actions = [a for a in range(self.n_actions)
                               if (s, a) in self.model]
            if not available_actions:
                continue

            a = np.random.choice(available_actions)
            r, s_next, d = self.model[(s, a)]

            # 使用模拟经验更新Q值
            if d:
                simulated_target = r
            else:
                simulated_target = r + self.gamma * np.max(self.q_table[s_next])
            self.q_table[s, a] += self.alpha * (simulated_target - self.q_table[s, a])


# ============================================================================
# 4. Policy-Based: REINFORCE（基于策略、同策略）
# ============================================================================

class REINFORCE:
    """
    REINFORCE算法（Monte Carlo Policy Gradient）

    分类：
    - Model-Free：不需要环境模型
    - On-Policy：学习当前策略
    - Policy-Based：直接学习策略
    - Monte Carlo：使用完整回合的回报

    核心思想：
    - 直接参数化策略 π(a|s; θ)
    - 使用策略梯度更新参数
    - 好的动作（高回报）增加概率，坏的动作减少概率
    """

    def __init__(self, n_states, n_actions, alpha=0.01, gamma=0.9):
        # 策略参数：对每个状态-动作对有一个偏好值
        self.preferences = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.n_actions = n_actions

    def get_policy(self, state):
        """
        计算策略概率分布（Softmax）

        π(a|s) = exp(h(s,a)) / Σ exp(h(s,a'))
        """
        exp_pref = np.exp(self.preferences[state] - np.max(self.preferences[state]))
        return exp_pref / np.sum(exp_pref)

    def choose_action(self, state):
        """根据策略概率采样动作"""
        probs = self.get_policy(state)
        return np.random.choice(self.n_actions, p=probs)

    def update(self, episode):
        """
        REINFORCE更新（需要完整回合）

        episode: [(state, action, reward), ...]

        策略梯度：
        ∇J(θ) = E[G_t · ∇log π(a_t|s_t; θ)]
        """
        T = len(episode)

        # 计算每个时间步的回报
        G = np.zeros(T)
        G[-1] = episode[-1][2]  # 最后一步的回报就是奖励
        for t in range(T-2, -1, -1):
            G[t] = episode[t][2] + self.gamma * G[t+1]

        # 策略梯度更新
        for t in range(T):
            state, action, reward = episode[t]

            # 计算梯度
            probs = self.get_policy(state)
            # ∇log π(a|s) = 1(a=a_selected) - π(a|s)
            grad = -probs
            grad[action] += 1

            # 更新参数
            self.preferences[state] += self.alpha * G[t] * grad


# ============================================================================
# 训练和对比
# ============================================================================

def train_agent(agent, env, n_episodes=500, agent_type='q-learning'):
    """训练智能体"""
    episode_rewards = []
    episode_steps = []

    for episode in range(n_episodes):
        state = env.reset()
        state_idx = env.get_state_index(state)
        total_reward = 0
        steps = 0

        if agent_type == 'reinforce':
            # REINFORCE需要收集完整回合
            episode_data = []
        elif agent_type == 'sarsa':
            # SARSA需要先选择第一个动作
            action = agent.choose_action(state_idx)

        for step in range(100):
            if agent_type != 'sarsa':
                action = agent.choose_action(state_idx)

            next_state, reward, done, _ = env.step(action)
            next_state_idx = env.get_state_index(next_state)

            if agent_type == 'reinforce':
                episode_data.append((state_idx, action, reward))
            elif agent_type == 'sarsa':
                next_action = agent.choose_action(next_state_idx)
                agent.update(state_idx, action, reward, next_state_idx,
                           next_action, done)
                action = next_action
            else:
                agent.update(state_idx, action, reward, next_state_idx, done)

            total_reward += reward
            steps += 1
            state_idx = next_state_idx

            if done:
                break

        if agent_type == 'reinforce' and episode_data:
            agent.update(episode_data)

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

    return episode_rewards, episode_steps


def compare_algorithms():
    """对比不同RL算法"""
    print("\n" + "="*80)
    print("对比不同强化学习算法")
    print("="*80)

    env = GridWorld(size=4)
    n_states = env.state_space_size
    n_actions = env.action_space_size
    n_episodes = 300

    # 创建不同算法
    algorithms = {
        'Q-Learning\n(Off-Policy)': (QLearning(n_states, n_actions), 'q-learning'),
        'SARSA\n(On-Policy)': (SARSA(n_states, n_actions), 'sarsa'),
        'Dyna-Q\n(Model-Based)': (DynaQ(n_states, n_actions, planning_steps=5), 'q-learning'),
        'REINFORCE\n(Policy-Based)': (REINFORCE(n_states, n_actions, alpha=0.001), 'reinforce')
    }

    results = {}

    # 训练每个算法
    for name, (agent, agent_type) in algorithms.items():
        print(f"\n训练 {name.replace(chr(10), ' ')}...")
        rewards, steps = train_agent(agent, env, n_episodes, agent_type)
        results[name] = (rewards, steps)

        # 打印最后100回合的平均性能
        avg_reward = np.mean(rewards[-100:])
        avg_steps = np.mean(steps[-100:])
        print(f"  最后100回合平均奖励: {avg_reward:.2f}")
        print(f"  最后100回合平均步数: {avg_steps:.2f}")

    # 可视化对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for name, (rewards, steps) in results.items():
        # 计算移动平均
        window = 20
        rewards_ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        steps_ma = np.convolve(steps, np.ones(window)/window, mode='valid')

        ax1.plot(range(window-1, len(rewards)), rewards_ma, label=name, linewidth=2)
        ax2.plot(range(window-1, len(steps)), steps_ma, label=name, linewidth=2)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Learning Curve: Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Steps')
    ax2.set_title('Learning Curve: Steps to Goal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rl_algorithms_comparison.png', dpi=150, bbox_inches='tight')
    print("\n算法对比图已保存为 'rl_algorithms_comparison.png'")


def demonstrate_on_off_policy():
    """演示On-Policy vs Off-Policy的区别"""
    print("\n" + "="*80)
    print("On-Policy vs Off-Policy 详解")
    print("="*80)

    explanation = """
关键区别：学习的策略和行动的策略是否相同

1. On-Policy（同策略）- SARSA
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   更新公式：Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
                                        ↑
                                    实际执行的动作

   特点：
   - 学习当前使用的策略（ε-贪心）
   - 更保守，考虑了探索的风险
   - 适合在线学习

   例子：学开车
   - 你在学习如何安全驾驶（包括保守的探索行为）
   - 学到的策略就是你实际使用的策略

   优势：✓ 更稳定  ✓ 保守安全
   劣势：✗ 样本利用率低  ✗ 学习较慢

2. Off-Policy（异策略）- Q-Learning
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   更新公式：Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
                                        ↑
                                    最优动作（可能不执行）

   特点：
   - 学习最优策略，但用ε-贪心探索
   - 更激进，直接学习最优行为
   - 可以从其他策略的数据中学习

   例子：学开车
   - 你在观察别人开车（包括新手的探索）
   - 但只学习最优的驾驶行为

   优势：✓ 样本利用率高  ✓ 可以学习专家数据  ✓ 学习较快
   劣势：✗ 可能不稳定  ✗ 可能过于激进

实际影响：
在悬崖行走问题中：
- SARSA：学会远离悬崖的安全路径（考虑探索的风险）
- Q-Learning：学会沿着悬崖的最优路径（不考虑探索的风险）
    """
    print(explanation)


def main():
    """主函数"""
    print("="*80)
    print("强化学习的主要形式及其关系")
    print("="*80)

    # 1. 打印分类体系
    print_rl_taxonomy()

    # 2. 演示On-Policy vs Off-Policy
    demonstrate_on_off_policy()

    # 3. 对比不同算法
    compare_algorithms()

    # 4. 总结
    print("\n" + "="*80)
    print("关键要点总结")
    print("="*80)
    print("""
1. Model-Free vs Model-Based
   - Model-Free：简单但样本效率低（Q-Learning, SARSA）
   - Model-Based：样本效率高但需要准确的模型（Dyna-Q）

2. On-Policy vs Off-Policy
   - On-Policy：稳定但样本利用率低（SARSA）
   - Off-Policy：样本利用率高但可能不稳定（Q-Learning）

3. Value-Based vs Policy-Based
   - Value-Based：适合离散动作（Q-Learning, DQN）
   - Policy-Based：适合连续动作和随机策略（REINFORCE, PPO）
   - Actor-Critic：结合两者优势（A3C, SAC）

4. 算法选择指南
   - 离散动作 + 表格状态 → Q-Learning/SARSA
   - 连续动作 → Policy Gradient/Actor-Critic
   - 需要样本效率 → Model-Based
   - 需要稳定性 → On-Policy
   - 有专家数据 → Off-Policy

5. 现代深度强化学习
   - DQN：Deep Q-Network（Value-Based + Off-Policy）
   - PPO：Proximal Policy Optimization（Policy-Based + On-Policy）
   - SAC：Soft Actor-Critic（Actor-Critic + Off-Policy）
    """)
    print("="*80)


if __name__ == "__main__":
    main()
