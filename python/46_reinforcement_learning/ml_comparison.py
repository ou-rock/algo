"""
机器学习方法对比：强化学习 vs 监督学习 vs 非监督学习

本文件详细对比三种主要的机器学习范式，并通过实际代码演示它们的区别。
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ============================================================================
# 1. 监督学习示例：根据温度和湿度预测是否适合户外活动
# ============================================================================

class SupervisedLearningExample:
    """
    监督学习示例：分类问题

    特点：
    - 有标注的训练数据（输入-输出对）
    - 学习输入到输出的映射函数
    - 目标是最小化预测误差
    - 学习过程：从大量标注样本中学习模式
    """

    def __init__(self):
        self.weights = None
        self.bias = None

    def generate_labeled_data(self, n_samples=100):
        """
        生成带标签的训练数据

        规则：温度高（>25度）且湿度低（<60%）适合户外活动

        Returns:
            X: 特征矩阵 [温度, 湿度]
            y: 标签 (1=适合, 0=不适合)
        """
        np.random.seed(42)

        # 生成随机特征
        temperature = np.random.uniform(10, 40, n_samples)  # 温度10-40度
        humidity = np.random.uniform(30, 90, n_samples)      # 湿度30-90%

        X = np.column_stack([temperature, humidity])

        # 根据规则生成标签（带一些噪声）
        y = np.zeros(n_samples)
        for i in range(n_samples):
            if temperature[i] > 25 and humidity[i] < 60:
                y[i] = 1 if np.random.random() > 0.1 else 0  # 90%正确率
            else:
                y[i] = 0 if np.random.random() > 0.1 else 1

        return X, y

    def train(self, X, y, learning_rate=0.01, epochs=1000):
        """
        训练简单的逻辑回归模型

        监督学习的典型训练过程：
        1. 初始化参数
        2. 前向传播：计算预测值
        3. 计算损失
        4. 反向传播：更新参数
        5. 重复直到收敛
        """
        n_samples, n_features = X.shape

        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 梯度下降
        for epoch in range(epochs):
            # 前向传播
            linear_output = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_output)

            # 计算梯度
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            # 更新参数
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

            if epoch % 200 == 0:
                loss = -np.mean(y * np.log(predictions + 1e-8) +
                              (1-y) * np.log(1 - predictions + 1e-8))
                print(f"  Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        """预测新样本"""
        linear_output = np.dot(X, self.weights) + self.bias
        return (self._sigmoid(linear_output) > 0.5).astype(int)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# ============================================================================
# 2. 非监督学习示例：将天气数据聚类
# ============================================================================

class UnsupervisedLearningExample:
    """
    非监督学习示例：K-Means聚类

    特点：
    - 没有标注的数据（只有输入，没有输出）
    - 学习数据的内在结构和模式
    - 目标是发现数据中的隐藏模式
    - 学习过程：自动发现数据的分组或特征
    """

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.centroids = None

    def generate_unlabeled_data(self, n_samples=150):
        """
        生成无标签的数据

        注意：这里我们生成数据时知道真实的类别，但算法训练时不使用这些标签

        Returns:
            X: 特征矩阵 [温度, 湿度]
        """
        np.random.seed(42)

        # 生成三个簇的数据（但不告诉算法标签）
        cluster1 = np.random.randn(50, 2) * 3 + [15, 70]  # 冷且湿
        cluster2 = np.random.randn(50, 2) * 3 + [30, 40]  # 热且干
        cluster3 = np.random.randn(50, 2) * 3 + [20, 55]  # 温和

        X = np.vstack([cluster1, cluster2, cluster3])
        return X

    def fit(self, X, max_iters=100):
        """
        K-Means聚类算法

        非监督学习的典型过程：
        1. 随机初始化聚类中心
        2. 分配：每个点分配到最近的中心
        3. 更新：重新计算每个簇的中心
        4. 重复2-3直到收敛

        关键：整个过程不需要标签！
        """
        n_samples = X.shape[0]

        # 随机初始化聚类中心
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for iteration in range(max_iters):
            # 分配每个点到最近的聚类中心
            labels = self._assign_clusters(X)

            # 保存旧的中心点
            old_centroids = self.centroids.copy()

            # 更新聚类中心
            for k in range(self.n_clusters):
                points_in_cluster = X[labels == k]
                if len(points_in_cluster) > 0:
                    self.centroids[k] = points_in_cluster.mean(axis=0)

            # 检查是否收敛
            if np.allclose(old_centroids, self.centroids):
                print(f"  聚类在第 {iteration} 次迭代后收敛")
                break

        return labels

    def predict(self, X):
        """预测新样本属于哪个簇"""
        return self._assign_clusters(X)

    def _assign_clusters(self, X):
        """分配每个点到最近的聚类中心"""
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.linalg.norm(X - self.centroids[k], axis=1)
        return np.argmin(distances, axis=1)


# ============================================================================
# 3. 强化学习示例：学习在不同天气下选择最佳活动
# ============================================================================

class ReinforcementLearningExample:
    """
    强化学习示例：多臂老虎机（Multi-Armed Bandit）

    特点：
    - 通过与环境交互学习
    - 通过奖励信号学习，而不是正确答案
    - 需要平衡探索和利用
    - 目标是最大化累积奖励
    - 学习过程：试错学习（trial-and-error）
    """

    def __init__(self, n_actions=4):
        """
        n_actions: 动作数量（例如：室内运动、户外运动、读书、睡觉）
        """
        self.n_actions = n_actions
        self.q_values = np.zeros(n_actions)  # 估计的动作价值
        self.action_counts = np.zeros(n_actions)  # 每个动作被选择的次数

    def select_action(self, epsilon=0.1):
        """
        ε-贪心策略选择动作

        强化学习的关键：探索vs利用的权衡
        - 以ε的概率随机探索
        - 以1-ε的概率选择当前最优动作
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.n_actions)  # 探索
        else:
            return np.argmax(self.q_values)  # 利用

    def update(self, action, reward):
        """
        更新动作价值估计

        使用增量式平均更新：
        Q(a) ← Q(a) + α[R - Q(a)]

        这是强化学习的核心：从奖励中学习
        """
        self.action_counts[action] += 1
        # 增量式更新
        alpha = 1.0 / self.action_counts[action]  # 学习率
        self.q_values[action] += alpha * (reward - self.q_values[action])

    def train(self, environment, n_steps=1000, epsilon=0.1):
        """
        训练智能体

        强化学习的典型训练循环：
        1. 观察当前状态
        2. 选择动作
        3. 执行动作，获得奖励
        4. 更新策略
        5. 重复
        """
        rewards_history = []

        for step in range(n_steps):
            # 选择动作
            action = self.select_action(epsilon)

            # 执行动作，获得奖励
            reward = environment.get_reward(action)

            # 更新Q值
            self.update(action, reward)

            # 记录奖励
            rewards_history.append(reward)

            if step % 200 == 0:
                avg_reward = np.mean(rewards_history[-100:]) if step > 0 else 0
                print(f"  步骤 {step}, 平均奖励: {avg_reward:.3f}")

        return rewards_history


class BanditEnvironment:
    """
    强化学习的环境：多臂老虎机

    环境定义了不同动作的真实奖励分布
    智能体需要通过交互来发现最优动作
    """

    def __init__(self):
        # 每个动作的真实平均奖励（智能体不知道）
        self.true_rewards = [0.1, 0.5, 0.3, 0.8]  # 动作3是最优的
        self.action_names = ["室内运动", "户外运动", "读书", "睡觉"]

    def get_reward(self, action):
        """
        执行动作，返回随机奖励

        注意：奖励是随机的，智能体需要多次尝试才能估计出真实价值
        """
        mean_reward = self.true_rewards[action]
        # 添加噪声
        reward = mean_reward + np.random.randn() * 0.1
        return reward


# ============================================================================
# 对比演示
# ============================================================================

def print_comparison_table():
    """打印三种学习方法的对比表"""
    print("\n" + "="*80)
    print("机器学习三大范式对比")
    print("="*80)

    comparison = """
╔════════════════╦══════════════════════╦══════════════════════╦══════════════════════╗
║    特性        ║    监督学习          ║    非监督学习        ║    强化学习          ║
╠════════════════╬══════════════════════╬══════════════════════╬══════════════════════╣
║ 训练数据       ║ 标注数据(输入+输出)  ║ 无标注数据(仅输入)   ║ 交互数据(状态+奖励)  ║
╠════════════════╬══════════════════════╬══════════════════════╬══════════════════════╣
║ 学习目标       ║ 学习输入到输出的映射 ║ 发现数据内在结构     ║ 学习最优决策策略     ║
╠════════════════╬══════════════════════╬══════════════════════╬══════════════════════╣
║ 反馈类型       ║ 直接反馈(正确答案)   ║ 无反馈               ║ 延迟反馈(奖励信号)   ║
╠════════════════╬══════════════════════╬══════════════════════╬══════════════════════╣
║ 学习方式       ║ 从示例中学习         ║ 从数据分布中学习     ║ 从试错中学习         ║
╠════════════════╬══════════════════════╬══════════════════════╬══════════════════════╣
║ 典型算法       ║ 线性回归、SVM、神经网络║ K-Means、PCA、自编码器║ Q-learning、策略梯度  ║
╠════════════════╬══════════════════════╬══════════════════════╬══════════════════════╣
║ 应用场景       ║ 分类、回归、预测     ║ 聚类、降维、异常检测 ║ 游戏、机器人、推荐   ║
╠════════════════╬══════════════════════╬══════════════════════╬══════════════════════╣
║ 关键挑战       ║ 需要大量标注数据     ║ 评估结果质量困难     ║ 探索-利用权衡        ║
╚════════════════╩══════════════════════╩══════════════════════╩══════════════════════╝
"""
    print(comparison)

    print("\n关键区别总结：")
    print("-" * 80)
    print("1. 监督学习：老师教学")
    print("   - 就像有老师告诉你每道题的正确答案")
    print("   - 例：根据历史数据预测房价")
    print()
    print("2. 非监督学习：自主学习")
    print("   - 就像自己观察数据找出规律，没有标准答案")
    print("   - 例：将客户分组，但不知道应该分成哪些组")
    print()
    print("3. 强化学习：实践学习")
    print("   - 就像学骑自行车，通过尝试和反馈来学习")
    print("   - 例：AlphaGo学习下围棋，通过胜负来改进策略")
    print("=" * 80)


def demonstrate_supervised_learning():
    """演示监督学习"""
    print("\n" + "="*80)
    print("1. 监督学习演示：天气活动预测")
    print("="*80)

    print("\n问题：根据温度和湿度预测是否适合户外活动")
    print("特点：我们有大量历史数据，每条数据都标注了是否适合户外活动\n")

    model = SupervisedLearningExample()

    # 生成训练数据
    X_train, y_train = model.generate_labeled_data(100)
    print(f"训练数据：{len(X_train)} 个样本，每个样本都有标签")
    print(f"前5个样本：")
    for i in range(5):
        print(f"  温度={X_train[i,0]:.1f}°C, 湿度={X_train[i,1]:.1f}%, "
              f"标签={'适合' if y_train[i]==1 else '不适合'}")

    # 训练模型
    print("\n开始训练（从标注数据中学习模式）...")
    model.train(X_train, y_train, epochs=1000)

    # 测试
    X_test = np.array([[28, 50], [20, 80], [35, 40]])
    predictions = model.predict(X_test)

    print("\n测试预测：")
    for i, (x, pred) in enumerate(zip(X_test, predictions)):
        print(f"  温度={x[0]}°C, 湿度={x[1]}%, 预测={'适合' if pred==1 else '不适合'}")

    print("\n监督学习总结：")
    print("- 需要：标注的训练数据")
    print("- 学习：输入特征到输出标签的映射")
    print("- 优势：当有足够标注数据时，预测准确")
    print("- 劣势：获取标注数据成本高")


def demonstrate_unsupervised_learning():
    """演示非监督学习"""
    print("\n" + "="*80)
    print("2. 非监督学习演示：天气模式聚类")
    print("="*80)

    print("\n问题：将天气数据自动分组，发现天气模式")
    print("特点：我们只有温度和湿度数据，没有任何标签\n")

    model = UnsupervisedLearningExample(n_clusters=3)

    # 生成无标签数据
    X = model.generate_unlabeled_data(150)
    print(f"无标签数据：{len(X)} 个样本，没有告诉算法应该如何分组")
    print(f"前5个样本（注意：没有标签！）：")
    for i in range(5):
        print(f"  温度={X[i,0]:.1f}°C, 湿度={X[i,1]:.1f}%")

    # 训练（聚类）
    print("\n开始聚类（自动发现数据分组）...")
    labels = model.fit(X)

    # 显示结果
    print(f"\n聚类结果（自动发现了 {model.n_clusters} 个天气模式）：")
    for k in range(model.n_clusters):
        cluster_points = X[labels == k]
        avg_temp = cluster_points[:, 0].mean()
        avg_humidity = cluster_points[:, 1].mean()
        print(f"  簇 {k}: {len(cluster_points)} 个样本, "
              f"平均温度={avg_temp:.1f}°C, 平均湿度={avg_humidity:.1f}%")

    print("\n非监督学习总结：")
    print("- 需要：无标签数据")
    print("- 学习：数据的内在结构和模式")
    print("- 优势：不需要昂贵的标注过程")
    print("- 劣势：难以评估结果质量，需要领域知识解释")


def demonstrate_reinforcement_learning():
    """演示强化学习"""
    print("\n" + "="*80)
    print("3. 强化学习演示：学习最佳活动选择")
    print("="*80)

    print("\n问题：学习在不同时间选择最佳活动（最大化快乐值）")
    print("特点：没有人告诉我们哪个活动最好，只能通过尝试获得反馈\n")

    env = BanditEnvironment()
    agent = ReinforcementLearningExample(n_actions=4)

    print(f"可选活动：{env.action_names}")
    print(f"真实奖励（智能体不知道！）：{env.true_rewards}")
    print("\n智能体需要通过试错来发现最佳活动")

    # 训练
    print("\n开始学习（通过试错和奖励反馈）...")
    rewards = agent.train(env, n_steps=1000, epsilon=0.1)

    # 显示学习结果
    print(f"\n学习到的动作价值（Q值）：")
    for i, (name, q_value, true_reward) in enumerate(
        zip(env.action_names, agent.q_values, env.true_rewards)):
        print(f"  {name}: Q值={q_value:.3f}, 真实奖励={true_reward:.3f}, "
              f"被选择{int(agent.action_counts[i])}次")

    best_action = np.argmax(agent.q_values)
    print(f"\n学到的最佳活动：{env.action_names[best_action]}")
    print(f"真实最佳活动：{env.action_names[np.argmax(env.true_rewards)]}")

    print("\n强化学习总结：")
    print("- 需要：环境交互和奖励信号")
    print("- 学习：通过试错找到最优策略")
    print("- 优势：适合序贯决策问题，不需要标注数据")
    print("- 劣势：需要大量交互，探索-利用权衡困难")


def visualize_all_methods():
    """可视化三种方法"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. 监督学习可视化
    ax1 = axes[0]
    model_sl = SupervisedLearningExample()
    X, y = model_sl.generate_labeled_data(100)
    colors = ['red' if label == 0 else 'blue' for label in y]
    ax1.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6)
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Humidity (%)')
    ax1.set_title('Supervised Learning\n(Labeled Data: Red=Not Suitable, Blue=Suitable)')
    ax1.grid(True, alpha=0.3)

    # 2. 非监督学习可视化
    ax2 = axes[1]
    model_ul = UnsupervisedLearningExample(n_clusters=3)
    X_unlabeled = model_ul.generate_unlabeled_data(150)
    labels = model_ul.fit(X_unlabeled)
    scatter = ax2.scatter(X_unlabeled[:, 0], X_unlabeled[:, 1],
                         c=labels, cmap='viridis', alpha=0.6)
    ax2.scatter(model_ul.centroids[:, 0], model_ul.centroids[:, 1],
               c='red', marker='X', s=200, edgecolors='black', label='Centroids')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Humidity (%)')
    ax2.set_title('Unsupervised Learning\n(Discovered Clusters)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 强化学习可视化
    ax3 = axes[2]
    env = BanditEnvironment()
    agent = ReinforcementLearningExample(n_actions=4)
    rewards = agent.train(env, n_steps=1000, epsilon=0.1)

    # 计算移动平均
    window = 50
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax3.plot(moving_avg)
    ax3.axhline(y=max(env.true_rewards), color='r', linestyle='--',
                label='Optimal Reward')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Average Reward')
    ax3.set_title('Reinforcement Learning\n(Learning Curve)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ml_comparison.png', dpi=150, bbox_inches='tight')
    print("\n可视化图已保存为 'ml_comparison.png'")


def main():
    """主函数"""
    print("="*80)
    print("机器学习三大范式详解与对比")
    print("="*80)

    # 打印对比表
    print_comparison_table()

    # 演示三种方法
    demonstrate_supervised_learning()
    demonstrate_unsupervised_learning()
    demonstrate_reinforcement_learning()

    # 可视化
    print("\n" + "="*80)
    print("生成可视化对比图...")
    print("="*80)
    visualize_all_methods()

    # 总结
    print("\n" + "="*80)
    print("关键要点总结")
    print("="*80)
    print("""
1. 强化学习 vs 监督学习：
   - 监督学习：有明确的"正确答案"（标签）
   - 强化学习：只有"好坏"的反馈（奖励），需要自己探索

2. 强化学习 vs 非监督学习：
   - 非监督学习：完全没有反馈，纯粹发现模式
   - 强化学习：有奖励反馈，但反馈可能延迟且稀疏

3. 强化学习的独特之处：
   - 交互性：需要与环境持续交互
   - 延迟反馈：当前动作的效果可能在未来才显现
   - 探索-利用权衡：需要平衡尝试新动作和使用已知好动作
   - 序贯决策：决策之间相互影响

4. 何时使用强化学习：
   - 问题涉及序贯决策（一系列相关的决策）
   - 难以获得标注数据，但可以定义奖励
   - 需要学习长期最优策略
   - 环境可以交互（游戏、机器人、推荐系统等）
    """)
    print("="*80)


if __name__ == "__main__":
    main()
