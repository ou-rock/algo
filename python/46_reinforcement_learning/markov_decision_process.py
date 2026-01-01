"""
马尔可夫决策过程（Markov Decision Process, MDP）详解

MDP是强化学习的数学基础，提供了形式化的框架来描述序贯决策问题。
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ============================================================================
# 理论讲解
# ============================================================================

def print_mdp_theory():
    """打印MDP理论讲解"""
    print("\n" + "="*80)
    print("马尔可夫决策过程（MDP）理论")
    print("="*80)

    theory = """
1. 马尔可夫性质（Markov Property）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

定义：未来状态只依赖于当前状态，与过去无关

数学表述：
P(S_{t+1} | S_t, S_{t-1}, ..., S_0) = P(S_{t+1} | S_t)

直观理解：
- "当前状态包含了所有相关的历史信息"
- 就像下棋，只需要看当前棋局，不需要知道之前的每一步

例子：
✓ 国际象棋：当前棋盘状态包含了所有信息
✓ 天气预测：今天的天气主要由昨天决定
✗ 股票价格：可能需要更长的历史信息（非马尔可夫）


2. MDP的定义
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MDP是一个五元组：(S, A, P, R, γ)

📌 S：状态空间（State Space）
   - 所有可能状态的集合
   - 例：网格世界中的所有格子位置

📌 A：动作空间（Action Space）
   - 所有可能动作的集合
   - 可以是离散的（上下左右）或连续的（角度、力度）

📌 P：状态转移概率（Transition Probability）
   - P(s'|s,a)：在状态s执行动作a后转移到状态s'的概率
   - 描述环境的动态特性
   - 随机性环境：P(s'|s,a) < 1
   - 确定性环境：P(s'|s,a) = 1（某个s'）

📌 R：奖励函数（Reward Function）
   - R(s,a,s')：在状态s执行动作a转移到s'获得的即时奖励
   - 有时简化为R(s)或R(s,a)

📌 γ：折扣因子（Discount Factor）
   - γ ∈ [0, 1]
   - 用于权衡即时奖励和未来奖励
   - γ=0：只考虑即时奖励（短视）
   - γ→1：重视长期奖励（远见）


3. 策略（Policy）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

策略π：从状态到动作的映射

确定性策略：π(s) = a
随机性策略：π(a|s) = P(A_t=a | S_t=s)

目标：找到最优策略π*，使累积奖励最大


4. 回报（Return）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

从时间步t开始的累积折扣奖励：

G_t = R_{t+1} + γR_{t+2} + γ²R_{t+3} + ...
    = Σ_{k=0}^∞ γ^k R_{t+k+1}

为什么需要折扣因子γ？
1. 数学上：保证无限序列收敛
2. 不确定性：未来的不确定性更大
3. 即时性偏好：更偏好早获得的奖励
4. 有限视野：避免无限期等待


5. 价值函数（Value Function）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

状态价值函数V^π(s)：
- 从状态s开始，遵循策略π的期望回报
- V^π(s) = E_π[G_t | S_t=s]
- 回答："在这个状态有多好？"

动作价值函数Q^π(s,a)：
- 在状态s执行动作a，然后遵循策略π的期望回报
- Q^π(s,a) = E_π[G_t | S_t=s, A_t=a]
- 回答："在这个状态执行这个动作有多好？"

关系：
V^π(s) = Σ_a π(a|s) Q^π(s,a)


6. 贝尔曼方程（Bellman Equation）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

贝尔曼期望方程（Bellman Expectation Equation）：

V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γV^π(s')]
         ↑           ↑                  ↑            ↑
      策略概率    转移概率           即时奖励      未来价值

Q^π(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γΣ_{a'} π(a'|s') Q^π(s',a')]

直观理解：
当前状态的价值 = 即时奖励 + 未来状态的折扣价值


贝尔曼最优方程（Bellman Optimality Equation）：

V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γV*(s')]
        ↑
      选择最优动作

Q*(s,a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ max_{a'} Q*(s',a')]

最优策略：
π*(s) = argmax_a Q*(s,a)


7. 求解MDP的方法
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

动态规划（需要完整的环境模型）：
├─ 策略评估（Policy Evaluation）：计算给定策略的价值函数
├─ 策略改进（Policy Improvement）：根据价值函数改进策略
├─ 策略迭代（Policy Iteration）：交替进行评估和改进
└─ 价值迭代（Value Iteration）：直接迭代最优价值函数

蒙特卡洛方法（不需要模型）：
└─ 通过完整回合的采样估计价值函数

时间差分学习（不需要模型）：
└─ Q-Learning, SARSA等（结合了动态规划和蒙特卡洛）
    """
    print(theory)


# ============================================================================
# 简单的MDP示例：学生学习问题
# ============================================================================

class StudentMDP:
    """
    学生学习MDP示例

    状态空间S：
    - 'sleeping': 睡觉
    - 'studying': 学习
    - 'playing': 玩耍
    - 'passed': 通过考试（终止状态）
    - 'failed': 考试失败（终止状态）

    动作空间A：
    - 'study': 学习
    - 'play': 玩耍
    - 'sleep': 睡觉
    - 'quit': 放弃

    奖励：
    - 学习：-2（努力的痛苦）
    - 玩耍：+1（快乐）
    - 睡觉：0（休息）
    - 通过考试：+10
    - 失败：-10
    """

    def __init__(self, gamma=0.9):
        self.gamma = gamma

        # 状态空间
        self.states = ['sleeping', 'studying', 'playing', 'passed', 'failed']
        self.terminal_states = ['passed', 'failed']

        # 动作空间
        self.actions = ['study', 'play', 'sleep', 'quit']

        # 定义状态转移和奖励
        self._define_transitions()

    def _define_transitions(self):
        """
        定义状态转移概率和奖励

        self.transitions[state][action] = [(prob, next_state, reward), ...]
        """
        self.transitions = {
            'sleeping': {
                'study': [(1.0, 'studying', -2)],
                'play': [(1.0, 'playing', 1)],
                'sleep': [(1.0, 'sleeping', 0)],
                'quit': [(1.0, 'failed', -10)]
            },
            'studying': {
                'study': [
                    (0.8, 'studying', -2),   # 80%继续学习
                    (0.2, 'passed', 10)      # 20%通过考试
                ],
                'play': [(1.0, 'playing', 1)],
                'sleep': [(1.0, 'sleeping', 0)],
                'quit': [(1.0, 'failed', -10)]
            },
            'playing': {
                'study': [(1.0, 'studying', -2)],
                'play': [
                    (0.9, 'playing', 1),     # 90%继续玩
                    (0.1, 'failed', -10)     # 10%因为玩太多而失败
                ],
                'sleep': [(1.0, 'sleeping', 0)],
                'quit': [(1.0, 'failed', -10)]
            },
            'passed': {},  # 终止状态
            'failed': {}   # 终止状态
        }

    def get_transitions(self, state, action):
        """获取状态转移"""
        if state in self.terminal_states:
            return []
        return self.transitions.get(state, {}).get(action, [])

    def is_terminal(self, state):
        """判断是否为终止状态"""
        return state in self.terminal_states


# ============================================================================
# 策略评估（Policy Evaluation）
# ============================================================================

def policy_evaluation(mdp, policy, theta=0.0001, max_iterations=1000):
    """
    策略评估：计算给定策略的状态价值函数

    使用迭代法求解贝尔曼期望方程：
    V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R + γV^π(s')]

    参数：
        mdp: MDP环境
        policy: 策略字典 {state: action}
        theta: 收敛阈值
        max_iterations: 最大迭代次数

    返回：
        V: 状态价值函数
    """
    # 初始化价值函数
    V = {state: 0.0 for state in mdp.states}

    for iteration in range(max_iterations):
        delta = 0

        # 对每个状态进行更新
        for state in mdp.states:
            if mdp.is_terminal(state):
                continue

            v = V[state]
            action = policy.get(state)

            if action is None:
                continue

            # 计算新的价值
            new_value = 0
            transitions = mdp.get_transitions(state, action)

            for prob, next_state, reward in transitions:
                new_value += prob * (reward + mdp.gamma * V[next_state])

            V[state] = new_value
            delta = max(delta, abs(v - V[state]))

        # 检查收敛
        if delta < theta:
            print(f"  策略评估在第 {iteration+1} 次迭代后收敛")
            break

    return V


# ============================================================================
# 策略改进（Policy Improvement）
# ============================================================================

def policy_improvement(mdp, V):
    """
    策略改进：根据价值函数生成更好的策略

    使用贪心策略：
    π'(s) = argmax_a Σ_{s'} P(s'|s,a) [R + γV(s')]

    参数：
        mdp: MDP环境
        V: 状态价值函数

    返回：
        new_policy: 改进后的策略
        policy_stable: 策略是否稳定
    """
    new_policy = {}
    policy_stable = True

    for state in mdp.states:
        if mdp.is_terminal(state):
            continue

        # 计算每个动作的价值
        action_values = {}
        for action in mdp.actions:
            transitions = mdp.get_transitions(state, action)
            if not transitions:
                continue

            value = 0
            for prob, next_state, reward in transitions:
                value += prob * (reward + mdp.gamma * V[next_state])

            action_values[action] = value

        # 选择最佳动作
        if action_values:
            best_action = max(action_values.items(), key=lambda x: x[1])[0]
            new_policy[state] = best_action

    return new_policy


# ============================================================================
# 策略迭代（Policy Iteration）
# ============================================================================

def policy_iteration(mdp, initial_policy=None):
    """
    策略迭代：交替进行策略评估和策略改进

    算法流程：
    1. 初始化一个任意策略π
    2. 策略评估：计算V^π
    3. 策略改进：根据V^π生成π'
    4. 如果π' = π，停止；否则π = π'，返回步骤2

    参数：
        mdp: MDP环境
        initial_policy: 初始策略

    返回：
        policy: 最优策略
        V: 最优价值函数
    """
    # 初始化策略
    if initial_policy is None:
        policy = {state: mdp.actions[0] for state in mdp.states
                 if not mdp.is_terminal(state)}
    else:
        policy = initial_policy.copy()

    iteration = 0
    print("\n开始策略迭代...")

    while True:
        iteration += 1
        print(f"\n--- 迭代 {iteration} ---")

        # 策略评估
        print("  执行策略评估...")
        V = policy_evaluation(mdp, policy)

        # 策略改进
        print("  执行策略改进...")
        new_policy = policy_improvement(mdp, V)

        # 检查策略是否稳定
        if new_policy == policy:
            print(f"\n策略在第 {iteration} 次迭代后收敛！")
            break

        policy = new_policy

    return policy, V


# ============================================================================
# 价值迭代（Value Iteration）
# ============================================================================

def value_iteration(mdp, theta=0.0001, max_iterations=1000):
    """
    价值迭代：直接迭代最优价值函数

    使用贝尔曼最优方程：
    V*(s) = max_a Σ_{s'} P(s'|s,a) [R + γV*(s')]

    算法流程：
    1. 初始化V(s) = 0
    2. 对每个状态，更新V(s)为所有动作中的最大值
    3. 重复直到收敛
    4. 提取最优策略

    参数：
        mdp: MDP环境
        theta: 收敛阈值
        max_iterations: 最大迭代次数

    返回：
        policy: 最优策略
        V: 最优价值函数
    """
    # 初始化价值函数
    V = {state: 0.0 for state in mdp.states}

    print("\n开始价值迭代...")

    for iteration in range(max_iterations):
        delta = 0

        # 对每个状态进行更新
        for state in mdp.states:
            if mdp.is_terminal(state):
                continue

            v = V[state]

            # 计算所有动作的价值，取最大值
            max_value = float('-inf')
            for action in mdp.actions:
                transitions = mdp.get_transitions(state, action)
                if not transitions:
                    continue

                action_value = 0
                for prob, next_state, reward in transitions:
                    action_value += prob * (reward + mdp.gamma * V[next_state])

                max_value = max(max_value, action_value)

            if max_value > float('-inf'):
                V[state] = max_value

            delta = max(delta, abs(v - V[state]))

        # 检查收敛
        if delta < theta:
            print(f"  价值迭代在第 {iteration+1} 次迭代后收敛")
            break

    # 提取最优策略
    policy = policy_improvement(mdp, V)

    return policy, V


# ============================================================================
# 可视化和演示
# ============================================================================

def print_policy_and_values(mdp, policy, V):
    """打印策略和价值函数"""
    print("\n" + "="*60)
    print("最优策略和状态价值")
    print("="*60)

    for state in mdp.states:
        if mdp.is_terminal(state):
            print(f"{state:12s}: 终止状态, V = {V[state]:6.2f}")
        else:
            action = policy.get(state, 'N/A')
            print(f"{state:12s}: {action:8s}, V = {V[state]:6.2f}")


def demonstrate_policy_evaluation():
    """演示策略评估"""
    print("\n" + "="*80)
    print("演示1：策略评估（Policy Evaluation）")
    print("="*80)

    mdp = StudentMDP(gamma=0.9)

    # 定义一个简单策略：总是学习
    simple_policy = {
        'sleeping': 'study',
        'studying': 'study',
        'playing': 'study'
    }

    print("\n评估策略：在所有状态都选择'study'")
    print("这个策略好吗？让我们计算它的价值函数...\n")

    V = policy_evaluation(mdp, simple_policy)

    print("\n策略评估结果：")
    for state in mdp.states:
        if not mdp.is_terminal(state):
            print(f"  {state:12s}: V = {V[state]:6.2f}")


def demonstrate_policy_iteration():
    """演示策略迭代"""
    print("\n" + "="*80)
    print("演示2：策略迭代（Policy Iteration）")
    print("="*80)

    mdp = StudentMDP(gamma=0.9)

    # 初始策略：随机策略
    initial_policy = {
        'sleeping': 'sleep',
        'studying': 'play',
        'playing': 'sleep'
    }

    print("\n初始策略（随机）：")
    for state, action in initial_policy.items():
        print(f"  {state}: {action}")

    policy, V = policy_iteration(mdp, initial_policy)

    print_policy_and_values(mdp, policy, V)


def demonstrate_value_iteration():
    """演示价值迭代"""
    print("\n" + "="*80)
    print("演示3：价值迭代（Value Iteration）")
    print("="*80)

    mdp = StudentMDP(gamma=0.9)

    policy, V = value_iteration(mdp)

    print_policy_and_values(mdp, policy, V)


def compare_methods():
    """对比不同方法"""
    print("\n" + "="*80)
    print("对比：策略迭代 vs 价值迭代")
    print("="*80)

    mdp = StudentMDP(gamma=0.9)

    print("\n" + "-"*60)
    print("策略迭代：")
    print("-"*60)
    policy_pi, V_pi = policy_iteration(mdp)

    print("\n" + "-"*60)
    print("价值迭代：")
    print("-"*60)
    policy_vi, V_vi = value_iteration(mdp)

    print("\n" + "-"*60)
    print("结果对比：")
    print("-"*60)

    print("\n策略是否相同？", policy_pi == policy_vi)
    print("\n价值函数差异：")
    for state in mdp.states:
        diff = abs(V_pi[state] - V_vi[state])
        print(f"  {state:12s}: {diff:.6f}")


def visualize_mdp():
    """可视化MDP"""
    mdp = StudentMDP(gamma=0.9)
    policy, V = value_iteration(mdp)

    # 准备数据
    non_terminal_states = [s for s in mdp.states if not mdp.is_terminal(s)]
    values = [V[s] for s in non_terminal_states]
    actions = [policy[s] for s in non_terminal_states]

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 状态价值函数
    colors = ['green' if v > 0 else 'red' for v in values]
    ax1.barh(non_terminal_states, values, color=colors, alpha=0.7)
    ax1.set_xlabel('State Value')
    ax1.set_title('Optimal State Value Function')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    # 最优策略
    action_mapping = {action: i for i, action in enumerate(mdp.actions)}
    action_values = [action_mapping[policy[s]] for s in non_terminal_states]

    ax2.barh(non_terminal_states, action_values, alpha=0.7)
    ax2.set_xlabel('Action')
    ax2.set_xticks(range(len(mdp.actions)))
    ax2.set_xticklabels(mdp.actions)
    ax2.set_title('Optimal Policy')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mdp_solution.png', dpi=150, bbox_inches='tight')
    print("\nMDP求解结果图已保存为 'mdp_solution.png'")


def main():
    """主函数"""
    print("="*80)
    print("马尔可夫决策过程（MDP）详解")
    print("="*80)

    # 1. 理论讲解
    print_mdp_theory()

    # 2. 演示策略评估
    demonstrate_policy_evaluation()

    # 3. 演示策略迭代
    demonstrate_policy_iteration()

    # 4. 演示价值迭代
    demonstrate_value_iteration()

    # 5. 对比方法
    compare_methods()

    # 6. 可视化
    print("\n" + "="*80)
    print("生成可视化图...")
    print("="*80)
    visualize_mdp()

    # 7. 总结
    print("\n" + "="*80)
    print("关键要点总结")
    print("="*80)
    print("""
1. 马尔可夫性质
   - 未来只依赖于现在，与过去无关
   - 是MDP的基础假设

2. MDP的组成
   - 状态空间S、动作空间A
   - 转移概率P、奖励函数R
   - 折扣因子γ

3. 价值函数
   - V(s)：状态价值函数，衡量状态的好坏
   - Q(s,a)：动作价值函数，衡量在状态下执行动作的好坏

4. 贝尔曼方程
   - 期望方程：评估给定策略
   - 最优方程：找到最优策略

5. 求解方法对比
   ┌────────────────┬──────────────┬──────────────┬──────────────┐
   │    方法        │  需要模型    │  计算复杂度  │  收敛速度    │
   ├────────────────┼──────────────┼──────────────┼──────────────┤
   │ 策略迭代       │  是          │  高          │  快（步数少）│
   │ 价值迭代       │  是          │  中          │  中          │
   │ Q-Learning     │  否          │  低          │  慢          │
   │ Monte Carlo    │  否          │  低          │  很慢        │
   └────────────────┴──────────────┴──────────────┴──────────────┘

6. 实际应用
   - 动态规划：适用于小规模、已知模型的问题
   - Q-Learning等：适用于大规模、未知模型的问题
   - 深度RL：使用神经网络近似价值函数/策略

7. MDP的局限性
   - 假设马尔可夫性（实际可能不满足）
   - 需要明确定义状态空间（可能很大）
   - 动态规划需要完整的环境模型（通常没有）

   解决方案：
   - 部分可观测MDP（POMDP）
   - 函数近似（Deep RL）
   - Model-Free方法
    """)
    print("="*80)


if __name__ == "__main__":
    main()
