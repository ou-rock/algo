"""
算法可视化工具
支持多种算法的可视化演示，包括排序算法、搜索算法等

Author: Claude
Date: 2025-12-19
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
from typing import List, Tuple, Callable
import time


class AlgorithmVisualizer:
    """算法可视化器基类"""

    def __init__(self, data: List[int] = None):
        """初始化

        Args:
            data: 要可视化的数据列表
        """
        self.data = data if data is not None else []
        self.fig, self.ax = None, None
        self.steps = []  # 存储算法执行的每一步

    def setup_plot(self, title: str):
        """设置绘图环境

        Args:
            title: 图表标题
        """
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()


class SortingVisualizer(AlgorithmVisualizer):
    """排序算法可视化器"""

    def __init__(self, data: List[int] = None):
        super().__init__(data)

    def bubble_sort_visualize(self):
        """冒泡排序可视化"""
        self.steps = []
        arr = self.data.copy()
        n = len(arr)

        # 记录初始状态
        self.steps.append({
            'array': arr.copy(),
            'comparing': [],
            'swapped': [],
            'sorted': [],
            'description': '初始状态'
        })

        for i in range(n):
            swapped = False
            for j in range(n - i - 1):
                # 记录比较状态
                self.steps.append({
                    'array': arr.copy(),
                    'comparing': [j, j + 1],
                    'swapped': [],
                    'sorted': list(range(n - i, n)),
                    'description': f'比较 {arr[j]} 和 {arr[j+1]}'
                })

                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    swapped = True
                    # 记录交换状态
                    self.steps.append({
                        'array': arr.copy(),
                        'comparing': [],
                        'swapped': [j, j + 1],
                        'sorted': list(range(n - i, n)),
                        'description': f'交换 {arr[j+1]} 和 {arr[j]}'
                    })

            if not swapped:
                break

        # 记录最终状态
        self.steps.append({
            'array': arr.copy(),
            'comparing': [],
            'swapped': [],
            'sorted': list(range(n)),
            'description': '排序完成！'
        })

        return self._create_animation('冒泡排序 (Bubble Sort)')

    def insertion_sort_visualize(self):
        """插入排序可视化"""
        self.steps = []
        arr = self.data.copy()
        n = len(arr)

        # 记录初始状态
        self.steps.append({
            'array': arr.copy(),
            'comparing': [],
            'inserting': -1,
            'sorted': [0],
            'description': '初始状态'
        })

        for i in range(1, n):
            key = arr[i]
            j = i - 1

            # 记录当前要插入的元素
            self.steps.append({
                'array': arr.copy(),
                'comparing': [],
                'inserting': i,
                'sorted': list(range(i)),
                'description': f'选择元素 {key} 准备插入'
            })

            while j >= 0 and arr[j] > key:
                # 记录比较和移动
                self.steps.append({
                    'array': arr.copy(),
                    'comparing': [j, j + 1],
                    'inserting': i,
                    'sorted': list(range(i)),
                    'description': f'比较 {arr[j]} > {key}, 向右移动'
                })

                arr[j + 1] = arr[j]
                j -= 1

            arr[j + 1] = key

            # 记录插入后的状态
            self.steps.append({
                'array': arr.copy(),
                'comparing': [],
                'inserting': -1,
                'sorted': list(range(i + 1)),
                'description': f'将 {key} 插入到位置 {j + 1}'
            })

        # 记录最终状态
        self.steps.append({
            'array': arr.copy(),
            'comparing': [],
            'inserting': -1,
            'sorted': list(range(n)),
            'description': '排序完成！'
        })

        return self._create_animation('插入排序 (Insertion Sort)')

    def selection_sort_visualize(self):
        """选择排序可视化"""
        self.steps = []
        arr = self.data.copy()
        n = len(arr)

        # 记录初始状态
        self.steps.append({
            'array': arr.copy(),
            'comparing': [],
            'min_index': -1,
            'sorted': [],
            'description': '初始状态'
        })

        for i in range(n):
            min_idx = i

            # 记录开始寻找最小值
            self.steps.append({
                'array': arr.copy(),
                'comparing': [i],
                'min_index': min_idx,
                'sorted': list(range(i)),
                'description': f'在位置 {i} 到 {n-1} 中寻找最小值'
            })

            for j in range(i + 1, n):
                # 记录比较过程
                self.steps.append({
                    'array': arr.copy(),
                    'comparing': [j, min_idx],
                    'min_index': min_idx,
                    'sorted': list(range(i)),
                    'description': f'比较 {arr[j]} 和当前最小值 {arr[min_idx]}'
                })

                if arr[j] < arr[min_idx]:
                    min_idx = j

            # 交换
            if min_idx != i:
                self.steps.append({
                    'array': arr.copy(),
                    'comparing': [i, min_idx],
                    'min_index': min_idx,
                    'sorted': list(range(i)),
                    'description': f'交换 {arr[i]} 和 {arr[min_idx]}'
                })

                arr[i], arr[min_idx] = arr[min_idx], arr[i]

            # 记录交换后的状态
            self.steps.append({
                'array': arr.copy(),
                'comparing': [],
                'min_index': -1,
                'sorted': list(range(i + 1)),
                'description': f'位置 {i} 已排序'
            })

        # 记录最终状态
        self.steps.append({
            'array': arr.copy(),
            'comparing': [],
            'min_index': -1,
            'sorted': list(range(n)),
            'description': '排序完成！'
        })

        return self._create_animation('选择排序 (Selection Sort)')

    def _create_animation(self, title: str):
        """创建动画

        Args:
            title: 动画标题
        """
        self.setup_plot(title)

        def update(frame):
            self.ax.clear()
            step = self.steps[frame]
            arr = step['array']
            n = len(arr)

            # 设置颜色
            colors = ['lightblue'] * n

            # 根据不同的状态设置不同的颜色
            if 'sorted' in step:
                for idx in step['sorted']:
                    colors[idx] = 'lightgreen'

            if 'comparing' in step:
                for idx in step['comparing']:
                    if idx < n:
                        colors[idx] = 'orange'

            if 'swapped' in step:
                for idx in step['swapped']:
                    if idx < n:
                        colors[idx] = 'red'

            if 'inserting' in step and step['inserting'] >= 0:
                colors[step['inserting']] = 'purple'

            if 'min_index' in step and step['min_index'] >= 0:
                colors[step['min_index']] = 'yellow'

            # 绘制柱状图
            bars = self.ax.bar(range(n), arr, color=colors, edgecolor='black', linewidth=1.5)

            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, arr)):
                height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

            # 设置标题和描述
            self.ax.set_title(f'{title}\n{step["description"]}',
                            fontsize=14, fontweight='bold')
            self.ax.set_xlabel('索引', fontsize=12)
            self.ax.set_ylabel('值', fontsize=12)
            self.ax.set_ylim(0, max(self.data) * 1.2)

            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightblue', label='未排序'),
                Patch(facecolor='orange', label='正在比较'),
                Patch(facecolor='red', label='已交换'),
                Patch(facecolor='lightgreen', label='已排序')
            ]
            self.ax.legend(handles=legend_elements, loc='upper right')

            return bars

        anim = animation.FuncAnimation(
            self.fig, update, frames=len(self.steps),
            interval=800, repeat=True, blit=False
        )

        return anim


class SearchVisualizer(AlgorithmVisualizer):
    """搜索算法可视化器"""

    def binary_search_visualize(self, target: int):
        """二分查找可视化

        Args:
            target: 要查找的目标值
        """
        self.steps = []
        arr = sorted(self.data.copy())  # 二分查找需要排序数组

        low, high = 0, len(arr) - 1
        found = False
        result_index = -1

        # 记录初始状态
        self.steps.append({
            'array': arr,
            'low': low,
            'high': high,
            'mid': -1,
            'target': target,
            'found': False,
            'description': f'在排序数组中查找 {target}'
        })

        while low <= high:
            mid = low + (high - low) // 2

            # 记录当前中间位置
            self.steps.append({
                'array': arr,
                'low': low,
                'high': high,
                'mid': mid,
                'target': target,
                'found': False,
                'description': f'检查中间位置 {mid}, 值为 {arr[mid]}'
            })

            if arr[mid] == target:
                found = True
                result_index = mid
                self.steps.append({
                    'array': arr,
                    'low': low,
                    'high': high,
                    'mid': mid,
                    'target': target,
                    'found': True,
                    'description': f'找到目标值 {target} 在位置 {mid}!'
                })
                break
            elif arr[mid] < target:
                self.steps.append({
                    'array': arr,
                    'low': low,
                    'high': high,
                    'mid': mid,
                    'target': target,
                    'found': False,
                    'description': f'{arr[mid]} < {target}, 在右半部分查找'
                })
                low = mid + 1
            else:
                self.steps.append({
                    'array': arr,
                    'low': low,
                    'high': high,
                    'mid': mid,
                    'target': target,
                    'found': False,
                    'description': f'{arr[mid]} > {target}, 在左半部分查找'
                })
                high = mid - 1

        if not found:
            self.steps.append({
                'array': arr,
                'low': low,
                'high': high,
                'mid': -1,
                'target': target,
                'found': False,
                'description': f'未找到目标值 {target}'
            })

        return self._create_search_animation('二分查找 (Binary Search)', target)

    def _create_search_animation(self, title: str, target: int):
        """创建搜索动画

        Args:
            title: 动画标题
            target: 目标值
        """
        self.setup_plot(title)

        def update(frame):
            self.ax.clear()
            step = self.steps[frame]
            arr = step['array']
            n = len(arr)

            # 设置颜色
            colors = ['lightgray'] * n

            # 标记搜索范围
            for i in range(step['low'], step['high'] + 1):
                colors[i] = 'lightblue'

            # 标记中间位置
            if step['mid'] >= 0:
                if step['found']:
                    colors[step['mid']] = 'lightgreen'
                else:
                    colors[step['mid']] = 'orange'

            # 绘制柱状图
            bars = self.ax.bar(range(n), arr, color=colors, edgecolor='black', linewidth=1.5)

            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, arr)):
                height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

            # 绘制目标线
            self.ax.axhline(y=target, color='red', linestyle='--', linewidth=2,
                          label=f'目标值: {target}')

            # 设置标题和描述
            self.ax.set_title(f'{title}\n{step["description"]}',
                            fontsize=14, fontweight='bold')
            self.ax.set_xlabel('索引', fontsize=12)
            self.ax.set_ylabel('值', fontsize=12)
            self.ax.set_ylim(0, max(arr) * 1.2)
            self.ax.legend(loc='upper right')

            return bars

        anim = animation.FuncAnimation(
            self.fig, update, frames=len(self.steps),
            interval=1000, repeat=True, blit=False
        )

        return anim


class SimpleAlgorithmVisualizer(AlgorithmVisualizer):
    """简单算法可视化器（求和、最大值、最小值）"""

    def find_max_visualize(self):
        """查找最大值可视化"""
        self.steps = []
        arr = self.data.copy()

        if not arr:
            return None

        max_val = arr[0]
        max_idx = 0

        # 记录初始状态
        self.steps.append({
            'array': arr,
            'current': 0,
            'max_index': 0,
            'max_value': max_val,
            'description': f'初始最大值: {max_val} (位置 0)'
        })

        for i in range(1, len(arr)):
            # 记录比较过程
            self.steps.append({
                'array': arr,
                'current': i,
                'max_index': max_idx,
                'max_value': max_val,
                'description': f'比较 {arr[i]} 和当前最大值 {max_val}'
            })

            if arr[i] > max_val:
                max_val = arr[i]
                max_idx = i
                # 记录更新最大值
                self.steps.append({
                    'array': arr,
                    'current': i,
                    'max_index': max_idx,
                    'max_value': max_val,
                    'description': f'更新最大值: {max_val} (位置 {i})'
                })

        # 记录最终状态
        self.steps.append({
            'array': arr,
            'current': len(arr) - 1,
            'max_index': max_idx,
            'max_value': max_val,
            'description': f'找到最大值: {max_val} (位置 {max_idx})'
        })

        return self._create_simple_animation('查找最大值 (Find Maximum)')

    def sum_visualize(self):
        """求和算法可视化"""
        self.steps = []
        arr = self.data.copy()

        total = 0

        # 记录初始状态
        self.steps.append({
            'array': arr,
            'current': -1,
            'sum': 0,
            'description': '开始求和, 初始和为 0'
        })

        for i in range(len(arr)):
            # 记录加法过程
            self.steps.append({
                'array': arr,
                'current': i,
                'sum': total,
                'description': f'当前和: {total}, 加上 {arr[i]}'
            })

            total += arr[i]

            # 记录累加后的结果
            self.steps.append({
                'array': arr,
                'current': i,
                'sum': total,
                'description': f'累加后的和: {total}'
            })

        # 记录最终状态
        self.steps.append({
            'array': arr,
            'current': len(arr) - 1,
            'sum': total,
            'description': f'最终和: {total}'
        })

        return self._create_simple_animation('数组求和 (Array Sum)')

    def _create_simple_animation(self, title: str):
        """创建简单算法动画

        Args:
            title: 动画标题
        """
        self.setup_plot(title)

        def update(frame):
            self.ax.clear()
            step = self.steps[frame]
            arr = step['array']
            n = len(arr)

            # 设置颜色
            colors = ['lightblue'] * n

            # 标记当前处理的元素
            if step['current'] >= 0:
                colors[step['current']] = 'orange'

            # 标记最大值位置（如果有）
            if 'max_index' in step:
                colors[step['max_index']] = 'lightgreen'

            # 绘制柱状图
            bars = self.ax.bar(range(n), arr, color=colors, edgecolor='black', linewidth=1.5)

            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, arr)):
                height = bar.get_height()
                self.ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val}',
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

            # 显示结果
            result_text = ''
            if 'max_value' in step:
                result_text = f'当前最大值: {step["max_value"]}'
            elif 'sum' in step:
                result_text = f'当前和: {step["sum"]}'

            # 设置标题和描述
            title_text = f'{title}\n{step["description"]}'
            if result_text:
                title_text += f'\n{result_text}'

            self.ax.set_title(title_text, fontsize=14, fontweight='bold')
            self.ax.set_xlabel('索引', fontsize=12)
            self.ax.set_ylabel('值', fontsize=12)

            if arr:
                self.ax.set_ylim(0, max(arr) * 1.2)

            return bars

        anim = animation.FuncAnimation(
            self.fig, update, frames=len(self.steps),
            interval=800, repeat=True, blit=False
        )

        return anim


def demo_sorting_algorithms():
    """演示排序算法"""
    print("=" * 60)
    print("排序算法可视化演示")
    print("=" * 60)

    # 测试数据
    data = [64, 34, 25, 12, 22, 11, 90, 88, 45, 50]
    print(f"原始数据: {data}\n")

    # 1. 冒泡排序
    print("1. 冒泡排序 (Bubble Sort)")
    visualizer = SortingVisualizer(data)
    anim = visualizer.bubble_sort_visualize()
    plt.show()

    # 2. 插入排序
    print("\n2. 插入排序 (Insertion Sort)")
    visualizer = SortingVisualizer(data)
    anim = visualizer.insertion_sort_visualize()
    plt.show()

    # 3. 选择排序
    print("\n3. 选择排序 (Selection Sort)")
    visualizer = SortingVisualizer(data)
    anim = visualizer.selection_sort_visualize()
    plt.show()


def demo_search_algorithms():
    """演示搜索算法"""
    print("\n" + "=" * 60)
    print("搜索算法可视化演示")
    print("=" * 60)

    # 测试数据
    data = [3, 7, 12, 18, 25, 34, 45, 56, 67, 78, 89, 95]
    target = 45
    print(f"排序数据: {data}")
    print(f"查找目标: {target}\n")

    visualizer = SearchVisualizer(data)
    anim = visualizer.binary_search_visualize(target)
    plt.show()


def demo_simple_algorithms():
    """演示简单算法"""
    print("\n" + "=" * 60)
    print("简单算法可视化演示")
    print("=" * 60)

    # 测试数据
    data = [5, 2, 8, 1, 9, 3, 7, 4, 6]
    print(f"数据: {data}\n")

    # 1. 查找最大值
    print("1. 查找最大值 (Find Maximum)")
    visualizer = SimpleAlgorithmVisualizer(data)
    anim = visualizer.find_max_visualize()
    plt.show()

    # 2. 数组求和
    print("\n2. 数组求和 (Array Sum)")
    visualizer = SimpleAlgorithmVisualizer(data)
    anim = visualizer.sum_visualize()
    plt.show()


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║           算法可视化工具 v1.0                             ║
    ║      Algorithm Visualization Tool                         ║
    ╚══════════════════════════════════════════════════════════╝

    本工具提供以下算法的可视化演示:

    1. 排序算法:
       - 冒泡排序 (Bubble Sort)
       - 插入排序 (Insertion Sort)
       - 选择排序 (Selection Sort)

    2. 搜索算法:
       - 二分查找 (Binary Search)

    3. 简单算法:
       - 查找最大值 (Find Maximum)
       - 数组求和 (Array Sum)

    使用方法:
    - 运行后会依次展示各个算法的动画演示
    - 每个动画会自动循环播放
    - 关闭窗口后会显示下一个算法
    """)

    try:
        # 依次演示各类算法
        demo_sorting_algorithms()
        demo_search_algorithms()
        demo_simple_algorithms()

        print("\n" + "=" * 60)
        print("所有演示完成!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n演示已中断")
    except Exception as e:
        print(f"\n错误: {e}")
