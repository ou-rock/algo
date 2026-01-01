"""
强化学习演示程序
演示Q-learning智能体在网格世界中的学习过程
"""

import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_environment import GridWorld
from simple_agent import QLearningAgent, RandomAgent
import matplotlib.pyplot as plt
import numpy as np


def train_agent(env, agent, num_episodes=500, max_steps=100, verbose=False):
    """
    训练智能体

    Args:
        env: 环境
        agent: 智能体
        num_episodes: 训练回合数
        max_steps: 每回合最大步数
        verbose: 是否打印详细信息

    Returns:
        episode_rewards: 每回合的总奖励
        episode_steps: 每回合的步数
    """
    episode_rewards = []
    episode_steps = []

    for episode in range(num_episodes):
        state = env.reset()
        state_index = env.get_state_index(state)
        total_reward = 0
        steps = 0

        for step in range(max_steps):
            # 选择动作
            action = agent.choose_action(state_index, explore=True)

            # 执行动作
            next_state, reward, done, info = env.step(action)
            next_state_index = env.get_state_index(next_state)

            # 学习
            agent.learn(state_index, action, reward, next_state_index, done)

            total_reward += reward
            steps += 1
            state_index = next_state_index

            if done:
                break

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        # 打印训练进度
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_steps[-100:])
            print(f"回合 {episode + 1}/{num_episodes}, "
                  f"平均奖励: {avg_reward:.2f}, "
                  f"平均步数: {avg_steps:.2f}")

    return episode_rewards, episode_steps


def test_agent(env, agent, num_episodes=10, max_steps=100, render=False):
    """
    测试智能体

    Args:
        env: 环境
        agent: 智能体
        num_episodes: 测试回合数
        max_steps: 每回合最大步数
        render: 是否渲染环境

    Returns:
        success_rate: 成功率
        avg_steps: 平均步数
    """
    success_count = 0
    total_steps = []

    for episode in range(num_episodes):
        state = env.reset()
        state_index = env.get_state_index(state)

        if render:
            print(f"\n===== 测试回合 {episode + 1} =====")
            env.render()

        for step in range(max_steps):
            # 选择动作（不探索，总是选择最佳动作）
            action = agent.choose_action(state_index, explore=False)

            # 执行动作
            next_state, reward, done, info = env.step(action)
            next_state_index = env.get_state_index(next_state)

            if render:
                print(f"\n步骤 {step + 1}: 动作={env.get_action_name(action)}")
                env.render()

            state_index = next_state_index

            if done:
                success_count += 1
                total_steps.append(step + 1)
                if render:
                    print(f"成功！用了 {step + 1} 步")
                break

        if not done and render:
            print(f"失败：未在 {max_steps} 步内到达目标")

    success_rate = success_count / num_episodes
    avg_steps = np.mean(total_steps) if total_steps else max_steps

    return success_rate, avg_steps


def visualize_q_table(env, agent):
    """
    可视化Q表（显示每个状态的最佳动作）

    Args:
        env: 环境
        agent: 智能体
    """
    print("\n策略可视化（每个状态的最佳动作）:")
    print("=" * (env.size * 4 + 1))

    action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    for i in range(env.size):
        print("|", end="")
        for j in range(env.size):
            state_index = env.get_state_index((i, j))

            if (i, j) == env.goal_state:
                print(" G ", end="|")  # 目标
            else:
                best_action = agent.get_best_action(state_index)
                symbol = action_symbols[best_action]
                print(f" {symbol} ", end="|")
        print()
        print("=" * (env.size * 4 + 1))


def plot_training_results(rewards, steps, window=50):
    """
    绘制训练结果

    Args:
        rewards: 每回合的奖励
        steps: 每回合的步数
        window: 移动平均窗口大小
    """
    # 计算移动平均
    def moving_average(data, window_size):
        if len(data) < window_size:
            window_size = len(data)
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    rewards_ma = moving_average(rewards, window)
    steps_ma = moving_average(steps, window)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 绘制奖励
    ax1.plot(rewards, alpha=0.3, label='原始奖励')
    ax1.plot(range(window - 1, len(rewards)), rewards_ma, label=f'{window}回合移动平均')
    ax1.set_xlabel('回合')
    ax1.set_ylabel('总奖励')
    ax1.set_title('训练过程 - 奖励')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 绘制步数
    ax2.plot(steps, alpha=0.3, label='原始步数')
    ax2.plot(range(window - 1, len(steps)), steps_ma, label=f'{window}回合移动平均')
    ax2.set_xlabel('回合')
    ax2.set_ylabel('步数')
    ax2.set_title('训练过程 - 步数')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rl_training_results.png', dpi=150, bbox_inches='tight')
    print("\n训练结果图已保存为 'rl_training_results.png'")


def main():
    """
    主函数：演示强化学习的完整流程
    """
    print("=" * 60)
    print("强化学习演示：Q-learning 在网格世界中学习")
    print("=" * 60)

    # 1. 创建环境
    print("\n1. 创建环境")
    print("-" * 60)
    grid_size = 4
    env = GridWorld(size=grid_size)
    print(f"环境：{grid_size}x{grid_size} 网格世界")
    print(f"起点：{env.start_state}")
    print(f"终点：{env.goal_state}")
    print(f"动作空间：{env.action_space_size} (上、下、左、右)")
    print(f"状态空间：{env.state_space_size}")

    # 2. 创建智能体
    print("\n2. 创建智能体")
    print("-" * 60)
    agent = QLearningAgent(
        state_space_size=env.state_space_size,
        action_space_size=env.action_space_size,
        learning_rate=0.1,    # 学习率
        discount_factor=0.9,  # 折扣因子
        epsilon=0.1           # 探索率
    )
    print(f"智能体：Q-learning")
    print(f"学习率 α = {agent.learning_rate}")
    print(f"折扣因子 γ = {agent.discount_factor}")
    print(f"探索率 ε = {agent.epsilon}")

    # 3. 训练前测试
    print("\n3. 训练前测试（随机策略）")
    print("-" * 60)
    success_rate, avg_steps = test_agent(env, agent, num_episodes=100)
    print(f"成功率: {success_rate * 100:.1f}%")
    print(f"平均步数: {avg_steps:.1f}")

    # 4. 训练智能体
    print("\n4. 训练智能体")
    print("-" * 60)
    num_episodes = 500
    print(f"训练回合数: {num_episodes}")

    rewards, steps = train_agent(
        env, agent,
        num_episodes=num_episodes,
        max_steps=100,
        verbose=True
    )

    # 5. 训练后测试
    print("\n5. 训练后测试")
    print("-" * 60)
    success_rate, avg_steps = test_agent(env, agent, num_episodes=100)
    print(f"成功率: {success_rate * 100:.1f}%")
    print(f"平均步数: {avg_steps:.1f}")

    # 6. 可视化学到的策略
    print("\n6. 学到的策略")
    print("-" * 60)
    visualize_q_table(env, agent)

    # 7. 演示一个完整回合
    print("\n7. 演示一个完整回合")
    print("-" * 60)
    test_agent(env, agent, num_episodes=1, render=True)

    # 8. 绘制训练曲线
    print("\n8. 绘制训练曲线")
    print("-" * 60)
    try:
        plot_training_results(rewards, steps)
    except Exception as e:
        print(f"无法绘制图表（可能没有安装matplotlib）: {e}")

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
