"""
高级算法的实际应用 - 最小化代码实现复杂应用
Advanced Algorithm Applications - Minimal Code, Maximum Impact

本文件展示了数据结构和算法在实际场景中的应用
每个应用都用最少的代码实现最复杂的功能

Author: Claude
Date: 2025-12-19
"""

from typing import List, Dict, Optional, Tuple, Set
from collections import OrderedDict, defaultdict
import heapq
from dataclasses import dataclass


# ============================================================================
# 1. 动态规划应用：投资组合优化器 (背包问题的实际应用)
# ============================================================================

class PortfolioOptimizer:
    """投资组合优化器 - 在有限预算内选择最优投资组合"""

    @staticmethod
    def optimize(projects: List[Tuple[str, int, int]], budget: int) -> Tuple[int, List[str]]:
        """
        选择项目以最大化收益

        Args:
            projects: [(名称, 成本, 收益), ...]
            budget: 总预算

        Returns:
            (最大收益, 选中的项目列表)

        示例：
            projects = [("AI项目", 50, 100), ("云计算", 30, 80), ("区块链", 40, 90)]
            budget = 80
            返回: (180, ["AI项目", "云计算"])
        """
        n = len(projects)
        # dp[i][j] = 前i个项目，预算j时的最大收益
        dp = [[0] * (budget + 1) for _ in range(n + 1)]

        # 动态规划填表
        for i in range(1, n + 1):
            name, cost, profit = projects[i-1]
            for j in range(budget + 1):
                # 不选第i个项目
                dp[i][j] = dp[i-1][j]
                # 选第i个项目
                if j >= cost:
                    dp[i][j] = max(dp[i][j], dp[i-1][j-cost] + profit)

        # 回溯找出选中的项目
        selected = []
        j = budget
        for i in range(n, 0, -1):
            if dp[i][j] != dp[i-1][j]:
                selected.append(projects[i-1][0])
                j -= projects[i-1][1]

        return dp[n][budget], selected[::-1]


# ============================================================================
# 2. LRU缓存应用：智能Web缓存系统
# ============================================================================

class SmartWebCache:
    """智能Web缓存 - 自动缓存最常访问的网页内容"""

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def get(self, url: str) -> Optional[str]:
        """获取缓存的网页内容"""
        if url in self.cache:
            self.hits += 1
            self.cache.move_to_end(url)  # 移到最近使用
            return self.cache[url]
        self.misses += 1
        return None

    def put(self, url: str, content: str) -> None:
        """缓存网页内容"""
        if url in self.cache:
            self.cache.move_to_end(url)
        self.cache[url] = content
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # 删除最久未使用

    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        return {
            "命中": self.hits,
            "未命中": self.misses,
            "命中率": f"{hit_rate:.2f}%",
            "当前缓存": list(self.cache.keys())
        }


# ============================================================================
# 3. Trie树应用：搜索引擎自动补全
# ============================================================================

class AutoComplete:
    """搜索自动补全系统 - 像Google搜索一样的智能提示"""

    class TrieNode:
        def __init__(self):
            self.children = {}
            self.is_word = False
            self.frequency = 0  # 搜索频率

    def __init__(self):
        self.root = self.TrieNode()

    def add_word(self, word: str, frequency: int = 1) -> None:
        """添加单词到字典（模拟用户搜索历史）"""
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = self.TrieNode()
            node = node.children[char]
        node.is_word = True
        node.frequency += frequency

    def autocomplete(self, prefix: str, limit: int = 5) -> List[Tuple[str, int]]:
        """
        返回以prefix开头的最热门搜索词

        Returns:
            [(词, 频率), ...] 按频率降序排列
        """
        node = self.root
        # 找到前缀节点
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]

        # DFS收集所有候选词
        results = []
        self._dfs_collect(node, prefix.lower(), results)

        # 按频率排序，返回top N
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def _dfs_collect(self, node, current_word, results):
        """深度优先搜索收集所有单词"""
        if node.is_word:
            results.append((current_word, node.frequency))

        for char, child_node in node.children.items():
            self._dfs_collect(child_node, current_word + char, results)


# ============================================================================
# 4. Dijkstra应用：智能导航系统
# ============================================================================

class SmartNavigation:
    """智能导航系统 - 计算两地之间的最短路径"""

    def __init__(self):
        self.graph = defaultdict(list)  # {节点: [(邻居, 距离), ...]}
        self.locations = {}  # 位置名称映射

    def add_road(self, from_loc: str, to_loc: str, distance: int, bidirectional: bool = True):
        """添加道路"""
        self.graph[from_loc].append((to_loc, distance))
        if bidirectional:
            self.graph[to_loc].append((from_loc, distance))

    def find_shortest_path(self, start: str, end: str) -> Tuple[int, List[str]]:
        """
        找到最短路径

        Returns:
            (总距离, 路径列表)
        """
        # 优先队列: (距离, 当前节点, 路径)
        pq = [(0, start, [start])]
        visited = set()

        while pq:
            dist, node, path = heapq.heappop(pq)

            if node in visited:
                continue

            if node == end:
                return dist, path

            visited.add(node)

            for neighbor, d in self.graph[node]:
                if neighbor not in visited:
                    heapq.heappush(pq, (dist + d, neighbor, path + [neighbor]))

        return float('inf'), []


# ============================================================================
# 5. KMP应用：DNA序列匹配器
# ============================================================================

class DNAMatcher:
    """DNA序列匹配器 - 在基因序列中快速查找特定模式"""

    @staticmethod
    def _build_lps(pattern: str) -> List[int]:
        """构建最长公共前后缀数组 (KMP核心)"""
        lps = [0] * len(pattern)
        length = 0
        i = 1

        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    def find_all_matches(self, dna: str, pattern: str) -> List[int]:
        """
        找到所有匹配位置

        Args:
            dna: DNA序列 (如 "ATCGATCGATCG")
            pattern: 要查找的模式 (如 "GATC")

        Returns:
            所有匹配位置的索引列表
        """
        if not pattern or not dna:
            return []

        lps = self._build_lps(pattern)
        matches = []

        i = j = 0
        while i < len(dna):
            if dna[i] == pattern[j]:
                i += 1
                j += 1

            if j == len(pattern):
                matches.append(i - j)
                j = lps[j - 1]
            elif i < len(dna) and dna[i] != pattern[j]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1

        return matches


# ============================================================================
# 6. 回溯应用：数独求解器
# ============================================================================

class SudokuSolver:
    """数独求解器 - 自动解决任何数独难题"""

    def solve(self, board: List[List[int]]) -> bool:
        """
        解决数独

        Args:
            board: 9x9的数独板，0表示空格

        Returns:
            是否成功求解（board会被修改）
        """
        empty = self._find_empty(board)
        if not empty:
            return True  # 已完成

        row, col = empty

        for num in range(1, 10):
            if self._is_valid(board, num, row, col):
                board[row][col] = num

                if self.solve(board):
                    return True

                board[row][col] = 0  # 回溯

        return False

    def _find_empty(self, board: List[List[int]]) -> Optional[Tuple[int, int]]:
        """找到第一个空格"""
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def _is_valid(self, board: List[List[int]], num: int, row: int, col: int) -> bool:
        """检查数字是否可以放置"""
        # 检查行
        if num in board[row]:
            return False

        # 检查列
        if num in [board[i][col] for i in range(9)]:
            return False

        # 检查3x3方块
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False

        return True


# ============================================================================
# 7. 贪心算法应用：会议室调度系统
# ============================================================================

@dataclass
class Meeting:
    """会议信息"""
    name: str
    start: int
    end: int
    priority: int = 0

class MeetingScheduler:
    """会议室调度系统 - 最大化会议室利用率"""

    @staticmethod
    def schedule_max_meetings(meetings: List[Meeting]) -> List[Meeting]:
        """
        调度最多的会议（活动选择问题）

        策略：按结束时间排序，贪心选择
        """
        # 按结束时间排序
        sorted_meetings = sorted(meetings, key=lambda m: m.end)

        scheduled = []
        last_end = 0

        for meeting in sorted_meetings:
            if meeting.start >= last_end:
                scheduled.append(meeting)
                last_end = meeting.end

        return scheduled


# ============================================================================
# 8. 堆应用：实时任务调度器
# ============================================================================

@dataclass
class Task:
    """任务信息"""
    name: str
    priority: int
    deadline: int

    def __lt__(self, other):
        # 优先级高的先执行（数字小表示优先级高）
        return self.priority < other.priority

class TaskScheduler:
    """实时任务调度器 - 基于优先级的任务管理"""

    def __init__(self):
        self.task_queue = []
        self.completed = []

    def add_task(self, task: Task) -> None:
        """添加任务到队列"""
        heapq.heappush(self.task_queue, task)

    def execute_next(self) -> Optional[Task]:
        """执行下一个最高优先级任务"""
        if not self.task_queue:
            return None

        task = heapq.heappop(self.task_queue)
        self.completed.append(task)
        return task

    def peek_next(self) -> Optional[Task]:
        """查看下一个待执行任务"""
        return self.task_queue[0] if self.task_queue else None


# ============================================================================
# 9. 并查集应用：社交网络分析
# ============================================================================

class SocialNetwork:
    """社交网络分析 - 查找朋友圈和社群"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.groups = n  # 独立社群数量

    def find(self, x: int) -> int:
        """查找根节点（路径压缩）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """连接两个用户（按秩合并）"""
        root_x, root_y = self.find(x), self.find(y)

        if root_x == root_y:
            return False  # 已经是朋友

        # 按秩合并
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.groups -= 1
        return True

    def are_friends(self, x: int, y: int) -> bool:
        """检查两个用户是否在同一个社群"""
        return self.find(x) == self.find(y)

    def get_group_count(self) -> int:
        """获取独立社群数量"""
        return self.groups


# ============================================================================
# 10. 滑动窗口应用：股票交易分析器
# ============================================================================

class StockAnalyzer:
    """股票交易分析器 - 实时分析股票趋势"""

    @staticmethod
    def max_profit_one_transaction(prices: List[int]) -> Tuple[int, int, int]:
        """
        找到最佳买卖时机（只交易一次）

        Returns:
            (最大利润, 买入日, 卖出日)
        """
        if not prices:
            return 0, -1, -1

        min_price = prices[0]
        max_profit = 0
        buy_day = sell_day = 0
        temp_buy_day = 0

        for i in range(1, len(prices)):
            if prices[i] < min_price:
                min_price = prices[i]
                temp_buy_day = i

            profit = prices[i] - min_price
            if profit > max_profit:
                max_profit = profit
                buy_day = temp_buy_day
                sell_day = i

        return max_profit, buy_day, sell_day

    @staticmethod
    def moving_average(prices: List[int], window: int) -> List[float]:
        """计算移动平均线"""
        if len(prices) < window:
            return []

        result = []
        window_sum = sum(prices[:window])
        result.append(window_sum / window)

        for i in range(window, len(prices)):
            window_sum = window_sum - prices[i-window] + prices[i]
            result.append(window_sum / window)

        return result


# ============================================================================
# 演示所有应用
# ============================================================================

def demo_all_applications():
    """演示所有算法应用"""

    print("=" * 80)
    print("高级算法实际应用演示")
    print("=" * 80)

    # 1. 投资组合优化
    print("\n【1. 投资组合优化器】")
    print("-" * 40)
    projects = [
        ("AI研发", 50, 120),
        ("云计算", 30, 80),
        ("区块链", 40, 100),
        ("物联网", 35, 70),
        ("5G通信", 45, 110)
    ]
    optimizer = PortfolioOptimizer()
    max_profit, selected = optimizer.optimize(projects, 100)
    print(f"总预算: 100万")
    print(f"最大收益: {max_profit}万")
    print(f"选择项目: {', '.join(selected)}")

    # 2. Web缓存系统
    print("\n【2. 智能Web缓存系统】")
    print("-" * 40)
    cache = SmartWebCache(capacity=3)
    urls = [
        "index.html", "about.html", "contact.html",
        "index.html", "products.html", "index.html"
    ]
    for url in urls:
        if cache.get(url):
            print(f"✓ 缓存命中: {url}")
        else:
            print(f"✗ 缓存未命中: {url}")
            cache.put(url, f"<content of {url}>")

    stats = cache.get_stats()
    print(f"\n缓存统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 3. 搜索自动补全
    print("\n【3. 搜索引擎自动补全】")
    print("-" * 40)
    autocomplete = AutoComplete()
    # 模拟用户搜索历史
    searches = [
        ("python", 100), ("python tutorial", 50), ("python programming", 30),
        ("java", 80), ("javascript", 90), ("java spring", 40),
        ("programming", 60), ("programmer", 45)
    ]
    for word, freq in searches:
        autocomplete.add_word(word, freq)

    prefix = "py"
    suggestions = autocomplete.autocomplete(prefix)
    print(f"输入: '{prefix}'")
    print(f"建议:")
    for word, freq in suggestions:
        print(f"  {word} (搜索 {freq} 次)")

    # 4. 智能导航
    print("\n【4. 智能导航系统】")
    print("-" * 40)
    nav = SmartNavigation()
    nav.add_road("家", "超市", 2)
    nav.add_road("家", "学校", 5)
    nav.add_road("超市", "学校", 1)
    nav.add_road("超市", "公园", 3)
    nav.add_road("学校", "公司", 4)
    nav.add_road("公园", "公司", 2)

    distance, path = nav.find_shortest_path("家", "公司")
    print(f"从'家'到'公司'的最短路径:")
    print(f"  路径: {' → '.join(path)}")
    print(f"  总距离: {distance}km")

    # 5. DNA序列匹配
    print("\n【5. DNA序列匹配器】")
    print("-" * 40)
    matcher = DNAMatcher()
    dna = "ATCGATCGATCGTAGCTAGCTAGC"
    pattern = "GATC"
    matches = matcher.find_all_matches(dna, pattern)
    print(f"DNA序列: {dna}")
    print(f"查找模式: {pattern}")
    print(f"找到 {len(matches)} 个匹配:")
    print(f"  位置: {matches}")

    # 6. 数独求解
    print("\n【6. 数独求解器】")
    print("-" * 40)
    board = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    solver = SudokuSolver()
    print("原始数独:")
    for row in board[:3]:
        print(f"  {row[:3]}")
    print("  ...")

    if solver.solve(board):
        print("\n✓ 求解成功!")
        print("解决方案:")
        for row in board[:3]:
            print(f"  {row[:3]}")
        print("  ...")

    # 7. 会议室调度
    print("\n【7. 会议室调度系统】")
    print("-" * 40)
    meetings = [
        Meeting("晨会", 9, 10),
        Meeting("项目评审", 9, 11),
        Meeting("技术分享", 10, 12),
        Meeting("午餐会", 12, 13),
        Meeting("客户会议", 11, 14),
        Meeting("总结会", 14, 15)
    ]

    scheduler = MeetingScheduler()
    scheduled = scheduler.schedule_max_meetings(meetings)
    print(f"提交的会议: {len(meetings)} 个")
    print(f"可调度的会议: {len(scheduled)} 个")
    print(f"调度结果:")
    for m in scheduled:
        print(f"  {m.start}:00-{m.end}:00 {m.name}")

    # 8. 任务调度
    print("\n【8. 实时任务调度器】")
    print("-" * 40)
    task_scheduler = TaskScheduler()
    tasks = [
        Task("处理订单", 1, 10),  # 高优先级
        Task("发送邮件", 3, 20),
        Task("备份数据", 5, 30),
        Task("处理退款", 1, 15),  # 高优先级
        Task("生成报表", 4, 25)
    ]

    for task in tasks:
        task_scheduler.add_task(task)

    print("任务执行顺序（按优先级）:")
    while task_scheduler.task_queue:
        task = task_scheduler.execute_next()
        print(f"  {task.name} (优先级: {task.priority})")

    # 9. 社交网络分析
    print("\n【9. 社交网络分析】")
    print("-" * 40)
    # 假设有6个用户
    network = SocialNetwork(6)
    friendships = [(0, 1), (1, 2), (3, 4)]  # 朋友关系

    for u1, u2 in friendships:
        network.union(u1, u2)
        print(f"  用户{u1} 和 用户{u2} 成为朋友")

    print(f"\n社群数量: {network.get_group_count()}")
    print(f"用户0 和 用户2 是朋友: {network.are_friends(0, 2)}")
    print(f"用户0 和 用户3 是朋友: {network.are_friends(0, 3)}")

    # 10. 股票交易分析
    print("\n【10. 股票交易分析器】")
    print("-" * 40)
    prices = [100, 80, 60, 70, 90, 110, 95, 105, 120]
    analyzer = StockAnalyzer()

    profit, buy_day, sell_day = analyzer.max_profit_one_transaction(prices)
    print(f"股票价格: {prices}")
    print(f"最佳买入日: 第{buy_day}天 (价格: {prices[buy_day]})")
    print(f"最佳卖出日: 第{sell_day}天 (价格: {prices[sell_day]})")
    print(f"最大利润: {profit}")

    ma5 = analyzer.moving_average(prices, 5)
    print(f"\n5日移动平均线: {[f'{x:.2f}' for x in ma5[:3]]}...")

    print("\n" + "=" * 80)
    print("演示完成！以上展示了算法在实际场景中的强大应用")
    print("=" * 80)


if __name__ == "__main__":
    demo_all_applications()
