"""
简单的Q-learning智能体
一个最基础的强化学习智能体示例
"""

import numpy as np
import random


class QLearningAgent:
    """
    Q-learning 智能体

    Q-learning 是一种无模型的强化学习算法：
    - 通过学习Q值（状态-动作值函数）来决策
    - Q值表示在某个状态下执行某个动作的期望累积奖励
    - 使用时间差分（TD）方法更新Q值

    Q-learning 更新公式：
    Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]

    其中：
    - s: 当前状态
    - a: 当前动作
    - r: 获得的奖励
    - s': 下一个状态
    - α: 学习率
    - γ: 折扣因子
    """

    def __init__(self, state_space_size, action_space_size,
                 learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        初始化Q-learning智能体

        Args:
            state_space_size: 状态空间大小
            action_space_size: 动作空间大小
            learning_rate: 学习率 α，控制新信息的接受程度
            discount_factor: 折扣因子 γ，控制未来奖励的重要性
            epsilon: 探索率，控制随机探索的概率
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        # 初始化Q表为全0
        # Q表维度：[状态数 x 动作数]
        self.q_table = np.zeros((state_space_size, action_space_size))

    def choose_action(self, state_index, explore=True):
        """
        选择动作（ε-贪心策略）

        以 ε 的概率随机选择动作（探索）
        以 1-ε 的概率选择Q值最大的动作（利用）

        Args:
            state_index: 当前状态索引
            explore: 是否启用探索

        Returns:
            选择的动作
        """
        if explore and random.random() < self.epsilon:
            # 探索：随机选择动作
            return random.randint(0, self.action_space_size - 1)
        else:
            # 利用：选择Q值最大的动作
            return np.argmax(self.q_table[state_index])

    def learn(self, state_index, action, reward, next_state_index, done):
        """
        使用Q-learning算法更新Q值

        Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]

        Args:
            state_index: 当前状态索引
            action: 执行的动作
            reward: 获得的奖励
            next_state_index: 下一个状态索引
            done: 是否到达终止状态
        """
        # 当前Q值
        current_q = self.q_table[state_index, action]

        # 如果到达终止状态，下一个状态的Q值为0
        if done:
            next_max_q = 0
        else:
            # 下一个状态的最大Q值
            next_max_q = np.max(self.q_table[next_state_index])

        # TD目标（时间差分目标）
        td_target = reward + self.discount_factor * next_max_q

        # TD误差
        td_error = td_target - current_q

        # 更新Q值
        self.q_table[state_index, action] = current_q + self.learning_rate * td_error

    def get_q_value(self, state_index, action):
        """
        获取指定状态-动作对的Q值

        Args:
            state_index: 状态索引
            action: 动作

        Returns:
            Q值
        """
        return self.q_table[state_index, action]

    def get_best_action(self, state_index):
        """
        获取指定状态下的最佳动作

        Args:
            state_index: 状态索引

        Returns:
            最佳动作
        """
        return np.argmax(self.q_table[state_index])

    def get_state_value(self, state_index):
        """
        获取状态价值（该状态下的最大Q值）

        Args:
            state_index: 状态索引

        Returns:
            状态价值
        """
        return np.max(self.q_table[state_index])


class RandomAgent:
    """
    随机智能体（作为基准对比）

    总是随机选择动作，不进行学习
    """

    def __init__(self, action_space_size):
        """
        初始化随机智能体

        Args:
            action_space_size: 动作空间大小
        """
        self.action_space_size = action_space_size

    def choose_action(self, state_index, explore=True):
        """
        随机选择动作

        Args:
            state_index: 当前状态索引（未使用）
            explore: 是否探索（未使用）

        Returns:
            随机动作
        """
        return random.randint(0, self.action_space_size - 1)

    def learn(self, state_index, action, reward, next_state_index, done):
        """
        随机智能体不学习

        Args:
            state_index: 当前状态索引（未使用）
            action: 动作（未使用）
            reward: 奖励（未使用）
            next_state_index: 下一个状态索引（未使用）
            done: 是否完成（未使用）
        """
        pass


if __name__ == "__main__":
    # 测试智能体
    print("测试Q-learning智能体")
    print("=" * 40)

    # 创建一个简单的智能体
    state_space_size = 16  # 4x4网格
    action_space_size = 4  # 上下左右

    agent = QLearningAgent(
        state_space_size=state_space_size,
        action_space_size=action_space_size,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.1
    )

    print(f"状态空间大小: {state_space_size}")
    print(f"动作空间大小: {action_space_size}")
    print(f"学习率: {agent.learning_rate}")
    print(f"折扣因子: {agent.discount_factor}")
    print(f"探索率: {agent.epsilon}")

    print("\n初始Q表:")
    print(agent.q_table)

    # 模拟一些学习步骤
    print("\n模拟学习过程:")
    state_index = 0
    for i in range(5):
        action = agent.choose_action(state_index)
        reward = random.uniform(-1, 1)
        next_state_index = random.randint(0, state_space_size - 1)
        done = (i == 4)

        print(f"\n步骤 {i + 1}:")
        print(f"  状态: {state_index}, 动作: {action}, 奖励: {reward:.2f}")

        agent.learn(state_index, action, reward, next_state_index, done)

        print(f"  更新后的Q值: {agent.get_q_value(state_index, action):.4f}")

        state_index = next_state_index

    print("\n更新后的Q表:")
    print(agent.q_table)
