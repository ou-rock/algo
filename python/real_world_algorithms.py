"""
真实世界的算法应用 - 工程实践中的算法
Real World Algorithms - Algorithms in Production Systems

本文件展示了在实际工程中广泛使用的算法和数据结构
涵盖分布式系统、网络服务、数据处理等场景

Author: Claude
Date: 2025-12-19
"""

from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict, deque
import time
import hashlib
import random


# ============================================================================
# 1. 限流器：令牌桶算法 (Token Bucket)
# ============================================================================

class RateLimiter:
    """API限流器 - 防止系统过载"""

    def __init__(self, rate: int, capacity: int):
        """
        Args:
            rate: 每秒生成令牌数
            capacity: 桶容量
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_time = time.time()

    def allow_request(self) -> bool:
        """判断请求是否允许通过"""
        now = time.time()
        # 计算这段时间内生成的令牌
        elapsed = now - self.last_time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_time = now

        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False

    def get_status(self) -> Dict:
        """获取限流器状态"""
        return {
            "当前令牌数": f"{self.tokens:.2f}",
            "桶容量": self.capacity,
            "生成速率": f"{self.rate}/秒"
        }


# ============================================================================
# 2. 一致性哈希：分布式缓存
# ============================================================================

class ConsistentHash:
    """一致性哈希 - 分布式缓存的负载均衡"""

    def __init__(self, nodes: List[str], virtual_nodes: int = 150):
        """
        Args:
            nodes: 服务器列表
            virtual_nodes: 每个节点的虚拟节点数
        """
        self.virtual_nodes = virtual_nodes
        self.ring = {}  # 哈希环
        self.sorted_keys = []

        for node in nodes:
            self.add_node(node)

    def _hash(self, key: str) -> int:
        """计算哈希值"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node: str):
        """添加节点"""
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}#{i}"
            hash_val = self._hash(virtual_key)
            self.ring[hash_val] = node

        self.sorted_keys = sorted(self.ring.keys())

    def remove_node(self, node: str):
        """移除节点"""
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}#{i}"
            hash_val = self._hash(virtual_key)
            if hash_val in self.ring:
                del self.ring[hash_val]

        self.sorted_keys = sorted(self.ring.keys())

    def get_node(self, key: str) -> str:
        """获取key应该存储的节点"""
        if not self.ring:
            return None

        hash_val = self._hash(key)

        # 二分查找找到第一个大于等于hash_val的位置
        idx = self._binary_search(hash_val)
        return self.ring[self.sorted_keys[idx]]

    def _binary_search(self, hash_val: int) -> int:
        """二分查找"""
        left, right = 0, len(self.sorted_keys) - 1

        if hash_val > self.sorted_keys[right]:
            return 0  # 环形，回到开头

        while left < right:
            mid = (left + right) // 2
            if self.sorted_keys[mid] < hash_val:
                left = mid + 1
            else:
                right = mid

        return left


# ============================================================================
# 3. 布隆过滤器：快速判断元素是否存在
# ============================================================================

class BloomFilter:
    """布隆过滤器 - 快速判断元素是否可能存在"""

    def __init__(self, size: int = 1000, hash_count: int = 3):
        """
        Args:
            size: 位数组大小
            hash_count: 哈希函数数量
        """
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [False] * size

    def _hashes(self, item: str) -> List[int]:
        """生成多个哈希值"""
        hashes = []
        for i in range(self.hash_count):
            hash_val = int(hashlib.md5(f"{item}{i}".encode()).hexdigest(), 16)
            hashes.append(hash_val % self.size)
        return hashes

    def add(self, item: str):
        """添加元素"""
        for hash_val in self._hashes(item):
            self.bit_array[hash_val] = True

    def contains(self, item: str) -> bool:
        """检查元素是否可能存在"""
        return all(self.bit_array[h] for h in self._hashes(item))


# ============================================================================
# 4. 拓扑排序：任务依赖调度
# ============================================================================

class TaskDependencyScheduler:
    """任务依赖调度器 - 自动解决任务依赖关系"""

    def __init__(self):
        self.graph = defaultdict(list)
        self.in_degree = defaultdict(int)

    def add_dependency(self, task: str, depends_on: str):
        """添加依赖关系：task依赖于depends_on"""
        self.graph[depends_on].append(task)
        self.in_degree[task] += 1
        if depends_on not in self.in_degree:
            self.in_degree[depends_on] = 0

    def schedule(self) -> List[str]:
        """
        返回任务执行顺序（拓扑排序）

        Returns:
            任务列表，如果有环则返回空列表
        """
        queue = deque([task for task, degree in self.in_degree.items() if degree == 0])
        result = []

        while queue:
            task = queue.popleft()
            result.append(task)

            for next_task in self.graph[task]:
                self.in_degree[next_task] -= 1
                if self.in_degree[next_task] == 0:
                    queue.append(next_task)

        # 检查是否所有任务都被调度（无环）
        if len(result) != len(self.in_degree):
            return []  # 存在环

        return result


# ============================================================================
# 5. 最小生成树：网络布线优化 (Prim算法)
# ============================================================================

class NetworkOptimizer:
    """网络布线优化器 - 使用最少线缆连接所有节点"""

    @staticmethod
    def minimum_spanning_tree(n: int, edges: List[Tuple[int, int, int]]) -> Tuple[int, List]:
        """
        计算最小生成树

        Args:
            n: 节点数量
            edges: [(节点1, 节点2, 成本), ...]

        Returns:
            (总成本, 选中的边列表)
        """
        # 构建邻接表
        graph = defaultdict(list)
        for u, v, cost in edges:
            graph[u].append((v, cost))
            graph[v].append((u, cost))

        visited = [False] * n
        min_heap = [(0, 0, -1)]  # (成本, 节点, 父节点)
        total_cost = 0
        selected_edges = []

        while min_heap:
            cost, node, parent = min(min_heap, key=lambda x: x[0])
            min_heap.remove((cost, node, parent))

            if visited[node]:
                continue

            visited[node] = True
            total_cost += cost

            if parent != -1:
                selected_edges.append((parent, node, cost))

            for neighbor, edge_cost in graph[node]:
                if not visited[neighbor]:
                    min_heap.append((edge_cost, neighbor, node))

        return total_cost, selected_edges


# ============================================================================
# 6. 编辑距离：拼写检查和智能纠错
# ============================================================================

class SpellChecker:
    """拼写检查器 - 基于编辑距离的智能纠错"""

    def __init__(self, dictionary: List[str]):
        self.dictionary = dictionary

    def edit_distance(self, word1: str, word2: str) -> int:
        """
        计算编辑距离（Levenshtein距离）

        需要的最少操作次数：插入、删除、替换
        """
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # 初始化
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # 动态规划
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,    # 删除
                        dp[i][j-1] + 1,    # 插入
                        dp[i-1][j-1] + 1   # 替换
                    )

        return dp[m][n]

    def suggest(self, word: str, max_suggestions: int = 3) -> List[Tuple[str, int]]:
        """
        返回拼写建议

        Returns:
            [(建议词, 编辑距离), ...] 按距离排序
        """
        suggestions = []

        for dict_word in self.dictionary:
            distance = self.edit_distance(word, dict_word)
            suggestions.append((dict_word, distance))

        suggestions.sort(key=lambda x: x[1])
        return suggestions[:max_suggestions]


# ============================================================================
# 7. 最长公共子序列：文件差异对比 (diff工具)
# ============================================================================

class FileDiff:
    """文件差异对比工具 - 类似git diff"""

    @staticmethod
    def lcs(text1: List[str], text2: List[str]) -> List[str]:
        """计算最长公共子序列"""
        m, n = len(text1), len(text2)
        dp = [[""] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + text1[i-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1], key=len)

        return list(dp[m][n])

    @staticmethod
    def diff(file1_lines: List[str], file2_lines: List[str]) -> List[str]:
        """
        生成文件差异报告

        Returns:
            差异行列表，用符号标识：
            + 表示新增
            - 表示删除
              表示相同
        """
        lcs = FileDiff.lcs(file1_lines, file2_lines)
        lcs_set = set(lcs)

        result = []
        i = j = 0

        while i < len(file1_lines) or j < len(file2_lines):
            if i < len(file1_lines) and file1_lines[i] not in lcs_set:
                result.append(f"- {file1_lines[i]}")
                i += 1
            elif j < len(file2_lines) and file2_lines[j] not in lcs_set:
                result.append(f"+ {file2_lines[j]}")
                j += 1
            else:
                if i < len(file1_lines):
                    result.append(f"  {file1_lines[i]}")
                    i += 1
                j += 1

        return result


# ============================================================================
# 8. 跳表：高效的有序集合 (Redis的有序集合实现)
# ============================================================================

class SkipListNode:
    """跳表节点"""
    def __init__(self, value, level):
        self.value = value
        self.forward = [None] * (level + 1)

class SkipList:
    """跳表 - Redis ZSET的底层实现"""

    def __init__(self, max_level: int = 16):
        self.max_level = max_level
        self.head = SkipListNode(float('-inf'), max_level)
        self.level = 0

    def _random_level(self) -> int:
        """随机生成层数"""
        level = 0
        while random.random() < 0.5 and level < self.max_level:
            level += 1
        return level

    def insert(self, value: int):
        """插入元素"""
        update = [None] * (self.max_level + 1)
        current = self.head

        # 从最高层向下找到插入位置
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].value < value:
                current = current.forward[i]
            update[i] = current

        level = self._random_level()

        if level > self.level:
            for i in range(self.level + 1, level + 1):
                update[i] = self.head
            self.level = level

        new_node = SkipListNode(value, level)

        for i in range(level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

    def search(self, value: int) -> bool:
        """查找元素"""
        current = self.head

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].value < value:
                current = current.forward[i]

        current = current.forward[0]
        return current and current.value == value

    def to_list(self) -> List[int]:
        """转换为列表"""
        result = []
        current = self.head.forward[0]
        while current:
            result.append(current.value)
            current = current.forward[0]
        return result


# ============================================================================
# 9. 时间轮：高效定时器 (Netty/Kafka的定时器实现)
# ============================================================================

class TimeWheel:
    """时间轮定时器 - 高效处理大量定时任务"""

    def __init__(self, tick_duration: int, wheel_size: int):
        """
        Args:
            tick_duration: 每格时间（毫秒）
            wheel_size: 轮子大小
        """
        self.tick_duration = tick_duration
        self.wheel_size = wheel_size
        self.wheel = [[] for _ in range(wheel_size)]
        self.current_tick = 0

    def add_task(self, task_name: str, delay_ms: int):
        """
        添加定时任务

        Args:
            task_name: 任务名称
            delay_ms: 延迟时间（毫秒）
        """
        ticks = delay_ms // self.tick_duration
        slot = (self.current_tick + ticks) % self.wheel_size
        self.wheel[slot].append((task_name, self.current_tick + ticks))

    def tick(self) -> List[str]:
        """
        时钟滴答，返回当前应执行的任务

        Returns:
            应执行的任务列表
        """
        slot = self.current_tick % self.wheel_size
        ready_tasks = []

        # 检查当前槽位的任务
        remaining_tasks = []
        for task_name, execute_tick in self.wheel[slot]:
            if execute_tick == self.current_tick:
                ready_tasks.append(task_name)
            else:
                remaining_tasks.append((task_name, execute_tick))

        self.wheel[slot] = remaining_tasks
        self.current_tick += 1

        return ready_tasks


# ============================================================================
# 10. 雪花算法：分布式ID生成器
# ============================================================================

class SnowflakeIDGenerator:
    """雪花算法ID生成器 - Twitter的分布式ID方案"""

    def __init__(self, datacenter_id: int, worker_id: int):
        """
        Args:
            datacenter_id: 数据中心ID (0-31)
            worker_id: 工作节点ID (0-31)
        """
        self.datacenter_id = datacenter_id & 0x1F
        self.worker_id = worker_id & 0x1F
        self.sequence = 0
        self.last_timestamp = -1

        # 起始时间戳 (2020-01-01)
        self.epoch = 1577836800000

    def _current_millis(self) -> int:
        """获取当前毫秒时间戳"""
        return int(time.time() * 1000)

    def generate_id(self) -> int:
        """
        生成唯一ID

        ID结构（64位）:
        - 1位：符号位，始终为0
        - 41位：时间戳（毫秒）
        - 5位：数据中心ID
        - 5位：工作节点ID
        - 12位：序列号
        """
        timestamp = self._current_millis()

        # 时钟回拨
        if timestamp < self.last_timestamp:
            raise Exception("时钟回拨，拒绝生成ID")

        # 同一毫秒内
        if timestamp == self.last_timestamp:
            self.sequence = (self.sequence + 1) & 0xFFF
            if self.sequence == 0:
                # 序列号用完，等待下一毫秒
                while timestamp <= self.last_timestamp:
                    timestamp = self._current_millis()
        else:
            self.sequence = 0

        self.last_timestamp = timestamp

        # 组装ID
        id_value = ((timestamp - self.epoch) << 22) | \
                   (self.datacenter_id << 17) | \
                   (self.worker_id << 12) | \
                   self.sequence

        return id_value


# ============================================================================
# 演示所有应用
# ============================================================================

def demo_all_real_world_applications():
    """演示所有真实世界的算法应用"""

    print("=" * 80)
    print("真实世界的算法应用演示")
    print("=" * 80)

    # 1. 限流器
    print("\n【1. API限流器】")
    print("-" * 40)
    limiter = RateLimiter(rate=2, capacity=5)
    print("限流配置: 2请求/秒, 桶容量5")

    for i in range(8):
        allowed = limiter.allow_request()
        status = "✓ 允许" if allowed else "✗ 拒绝"
        print(f"请求{i+1}: {status}")
        if i == 3:
            print("  (等待0.5秒...)")
            time.sleep(0.5)

    # 2. 一致性哈希
    print("\n【2. 一致性哈希 - 分布式缓存】")
    print("-" * 40)
    nodes = ["server1", "server2", "server3"]
    ch = ConsistentHash(nodes)

    keys = ["user:1001", "user:1002", "user:1003", "user:1004", "user:1005"]
    print("缓存键分配:")
    distribution = defaultdict(list)
    for key in keys:
        node = ch.get_node(key)
        distribution[node].append(key)

    for node, assigned_keys in distribution.items():
        print(f"  {node}: {len(assigned_keys)}个键")

    # 3. 布隆过滤器
    print("\n【3. 布隆过滤器 - 快速查重】")
    print("-" * 40)
    bf = BloomFilter(size=100, hash_count=3)

    existing_users = ["alice", "bob", "charlie", "david"]
    for user in existing_users:
        bf.add(user)

    test_users = ["alice", "eve", "bob", "frank"]
    print("查询结果:")
    for user in test_users:
        exists = bf.contains(user)
        actual = user in existing_users
        status = "✓" if exists == actual else "✗"
        print(f"  {status} {user}: {'存在' if exists else '不存在'}")

    # 4. 拓扑排序
    print("\n【4. 任务依赖调度】")
    print("-" * 40)
    scheduler = TaskDependencyScheduler()

    # 构建编译依赖: main -> utils -> config
    dependencies = [
        ("config.py", None),
        ("utils.py", "config.py"),
        ("database.py", "config.py"),
        ("models.py", "database.py"),
        ("main.py", "utils.py"),
        ("main.py", "models.py")
    ]

    print("依赖关系:")
    for task, dep in dependencies:
        if dep:
            scheduler.add_dependency(task, dep)
            print(f"  {task} 依赖于 {dep}")

    order = scheduler.schedule()
    print(f"\n执行顺序: {' → '.join(order)}")

    # 5. 最小生成树
    print("\n【5. 网络布线优化】")
    print("-" * 40)
    edges = [
        (0, 1, 4), (0, 2, 3), (1, 2, 1),
        (1, 3, 2), (2, 3, 4), (3, 4, 2), (2, 4, 5)
    ]
    print("网络节点: 0, 1, 2, 3, 4")
    print(f"可用线路: {len(edges)} 条")

    total_cost, selected = NetworkOptimizer.minimum_spanning_tree(5, edges)
    print(f"\n最优方案:")
    print(f"  总成本: {total_cost}")
    print(f"  使用线路: {len(selected)} 条")
    for u, v, cost in selected:
        print(f"    节点{u} ↔ 节点{v} (成本: {cost})")

    # 6. 拼写检查
    print("\n【6. 智能拼写检查】")
    print("-" * 40)
    dictionary = ["python", "java", "javascript", "golang", "rust", "swift"]
    checker = SpellChecker(dictionary)

    misspelled = "javasript"
    suggestions = checker.suggest(misspelled)

    print(f"输入: '{misspelled}'")
    print("建议:")
    for word, distance in suggestions:
        print(f"  {word} (编辑距离: {distance})")

    # 7. 文件差异对比
    print("\n【7. 文件差异对比 (diff)】")
    print("-" * 40)
    file1 = ["def hello():", "    print('Hello')", "    return True"]
    file2 = ["def hello():", "    print('Hi')", "    print('World')", "    return True"]

    print("文件对比:")
    diff_result = FileDiff.diff(file1, file2)
    for line in diff_result:
        print(f"  {line}")

    # 8. 跳表
    print("\n【8. 跳表 - 有序集合】")
    print("-" * 40)
    skiplist = SkipList()
    values = [3, 1, 4, 1, 5, 9, 2, 6]

    print(f"插入元素: {values}")
    for val in values:
        skiplist.insert(val)

    print(f"有序结果: {skiplist.to_list()}")
    print(f"查找5: {skiplist.search(5)}")
    print(f"查找7: {skiplist.search(7)}")

    # 9. 时间轮
    print("\n【9. 时间轮定时器】")
    print("-" * 40)
    time_wheel = TimeWheel(tick_duration=100, wheel_size=10)

    # 添加任务
    time_wheel.add_task("发送邮件", 300)
    time_wheel.add_task("清理缓存", 500)
    time_wheel.add_task("备份数据", 700)

    print("模拟时钟运行:")
    for i in range(10):
        ready = time_wheel.tick()
        if ready:
            print(f"  滴答{i}: 执行任务 {ready}")
        else:
            print(f"  滴答{i}: 无任务")

    # 10. 雪花算法
    print("\n【10. 分布式ID生成器】")
    print("-" * 40)
    generator = SnowflakeIDGenerator(datacenter_id=1, worker_id=1)

    print("生成的唯一ID:")
    for i in range(5):
        id_val = generator.generate_id()
        print(f"  ID{i+1}: {id_val}")

    print("\n" + "=" * 80)
    print("演示完成！这些算法正在支撑着互联网的各种服务")
    print("=" * 80)


if __name__ == "__main__":
    demo_all_real_world_applications()
