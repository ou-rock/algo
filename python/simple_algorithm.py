# 最简单的算法实现
# 包含：求和、求最大值、求最小值、求平均值
#
# Author: Claude

class SimpleAlgorithm:
    """简单算法类，实现一些基础的数组算法"""

    def __init__(self, data=None):
        """初始化方法

        参数：
            data: 整数数组，默认为空列表
        """
        self.data = data if data is not None else []

    def sum(self):
        """求和算法

        返回：
            数组所有元素的和
        """
        if not self.data:
            return 0

        total = 0
        for num in self.data:
            total += num
        return total

    def find_max(self):
        """查找最大值算法

        返回：
            数组中的最大值，如果数组为空则返回 None
        """
        if not self.data:
            return None

        max_value = self.data[0]
        for num in self.data:
            if num > max_value:
                max_value = num
        return max_value

    def find_min(self):
        """查找最小值算法

        返回：
            数组中的最小值，如果数组为空则返回 None
        """
        if not self.data:
            return None

        min_value = self.data[0]
        for num in self.data:
            if num < min_value:
                min_value = num
        return min_value

    def average(self):
        """求平均值算法

        返回：
            数组所有元素的平均值，如果数组为空则返回 0
        """
        if not self.data:
            return 0

        return self.sum() / len(self.data)

    def print_all(self):
        """打印当前数组所有数据"""
        print(f"数组内容: {self.data}")


# 测试代码
if __name__ == "__main__":
    # 创建算法对象
    algo = SimpleAlgorithm([5, 2, 8, 1, 9, 3])

    # 打印数组
    algo.print_all()

    # 测试各种算法
    print(f"求和结果: {algo.sum()}")
    print(f"最大值: {algo.find_max()}")
    print(f"最小值: {algo.find_min()}")
    print(f"平均值: {algo.average()}")

    # 空数组测试
    empty_algo = SimpleAlgorithm()
    print("\n空数组测试:")
    print(f"求和结果: {empty_algo.sum()}")
    print(f"最大值: {empty_algo.find_max()}")
    print(f"最小值: {empty_algo.find_min()}")
    print(f"平均值: {empty_algo.average()}")
