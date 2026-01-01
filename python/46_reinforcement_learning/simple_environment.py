"""
简单的网格世界环境
一个最基础的强化学习环境示例
"""


class GridWorld:
    """
    简单的网格世界环境

    环境描述：
    - 智能体在一个 n x n 的网格中移动
    - 起点在左上角 (0, 0)
    - 终点在右下角 (n-1, n-1)
    - 智能体可以执行4个动作：上、下、左、右
    - 到达终点获得 +10 奖励，每移动一步获得 -1 奖励
    """

    def __init__(self, size=4):
        """
        初始化网格世界

        Args:
            size: 网格大小 (size x size)
        """
        self.size = size
        self.start_state = (0, 0)
        self.goal_state = (size - 1, size - 1)
        self.current_state = self.start_state

        # 定义动作空间：0-上, 1-下, 2-左, 3-右
        self.actions = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        self.action_space_size = len(self.actions)
        self.state_space_size = size * size

    def reset(self):
        """
        重置环境到初始状态

        Returns:
            初始状态 (x, y)
        """
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        """
        执行一个动作，返回下一个状态、奖励、是否结束

        Args:
            action: 动作编号 (0-3)

        Returns:
            next_state: 下一个状态
            reward: 获得的奖励
            done: 是否到达终点
            info: 额外信息
        """
        # 获取当前位置
        x, y = self.current_state

        # 根据动作计算新位置
        dx, dy = self.actions[action]
        new_x, new_y = x + dx, y + dy

        # 检查是否超出边界
        if 0 <= new_x < self.size and 0 <= new_y < self.size:
            self.current_state = (new_x, new_y)
        # 如果超出边界，保持在原位置

        # 计算奖励
        if self.current_state == self.goal_state:
            reward = 10.0  # 到达终点
            done = True
        else:
            reward = -1.0  # 每步惩罚
            done = False

        info = {}
        return self.current_state, reward, done, info

    def get_state_index(self, state):
        """
        将二维状态转换为一维索引

        Args:
            state: (x, y) 状态

        Returns:
            状态索引
        """
        x, y = state
        return x * self.size + y

    def render(self):
        """
        打印当前环境状态
        """
        print("\n" + "=" * (self.size * 4 + 1))
        for i in range(self.size):
            print("|", end="")
            for j in range(self.size):
                if (i, j) == self.current_state:
                    print(" A ", end="|")  # A 表示智能体
                elif (i, j) == self.goal_state:
                    print(" G ", end="|")  # G 表示目标
                else:
                    print("   ", end="|")
            print()
            print("=" * (self.size * 4 + 1))

    def get_action_name(self, action):
        """
        获取动作名称

        Args:
            action: 动作编号

        Returns:
            动作名称
        """
        action_names = {0: "上", 1: "下", 2: "左", 3: "右"}
        return action_names.get(action, "未知")


if __name__ == "__main__":
    # 测试环境
    print("测试网格世界环境")
    print("=" * 40)

    env = GridWorld(size=4)
    state = env.reset()

    print(f"初始状态: {state}")
    print(f"目标状态: {env.goal_state}")
    print(f"动作空间大小: {env.action_space_size}")
    print(f"状态空间大小: {env.state_space_size}")

    print("\n初始环境:")
    env.render()

    # 执行几个随机动作
    import random
    print("\n执行随机动作:")
    for step in range(5):
        action = random.randint(0, 3)
        next_state, reward, done, info = env.step(action)
        print(f"\n步骤 {step + 1}: 动作={env.get_action_name(action)}, "
              f"状态={next_state}, 奖励={reward}, 完成={done}")
        env.render()

        if done:
            print("\n到达目标！")
            break
