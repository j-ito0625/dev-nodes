# 問題固有の工夫点とアルゴリズム最適化

## 目次
1. [本問題の特殊性と課題](#本問題の特殊性と課題)
2. [染色体表現の工夫](#染色体表現の工夫)
3. [制約処理の特殊化](#制約処理の特殊化)
4. [初期個体群生成の最適化](#初期個体群生成の最適化)
5. [適応度関数の調整](#適応度関数の調整)
6. [遺伝的操作の問題特化](#遺伝的操作の問題特化)
7. [性能最適化技法](#性能最適化技法)

## 本問題の特殊性と課題

### 一般的なジョブショップ問題との違い

#### 1. **複雑な時間制約**

**標準的なJSP**:
```
タスクA → タスクB (単純な先行関係)
```

**本問題**:
```python
# 複雑な時間差制約
{
    '先行作業ID': 'task-0001',
    '後作業ID': 'task-0008', 
    '先行作業原点': '終了',      # または '開始'
    '後作業原点': '開始',        # または '終了'  
    '時間差下限': -180          # 負値も許可（先行実行可能）
}
```

**実装での対応**:
- 4パターンの時間基準点の組み合わせに対応
- 負の時間差制約（オーバーラップ実行）の正確な処理
- 複数制約の同時充足アルゴリズム

#### 2. **柔軟なリソース割り当て制約**

**標準的なJSP**:
```
タスク → 固定の機械（1対1対応）
```

**本問題**:
```python
# 1つのタスクに複数の実行可能パターン
'task-0001' -> [
    {'worker': 'rsrc-0002', 'equipment': 'rsrc-0010'},
    {'worker': 'rsrc-0004', 'equipment': 'rsrc-0008'},
    {'worker': 'rsrc-0006', 'equipment': 'rsrc-0012'}
]
```

**実装での対応**:
- 動的なリソース選択メカニズム
- 制約充足と最適化の両立
- 割り当てパターンの効率的な管理

#### 3. **2次元リソース制約**

**標準的なJSP**: 機械のみ
**本問題**: 作業者 × 設備（組み合わせ制約）

```python
# 両方のリソースを同時に確保する必要
worker_availability[worker] = end_time
equipment_availability[equipment] = end_time
```

## 染色体表現の工夫

### 1. ハイブリッド染色体設計

#### 設計思想
```python
# 単純な順序表現（一般的なJSP）
chromosome = [task1, task3, task2, task4, ...]

# 本実装のハイブリッド表現
chromosome = [
    {'task': task1, 'worker': w1, 'equipment': e1},
    {'task': task3, 'worker': w2, 'equipment': e2},
    {'task': task2, 'worker': w1, 'equipment': e3},
    ...
]
```

#### 利点
1. **順序情報**: 染色体の配列順序がタスクの実行順序
2. **割り当て情報**: 各遺伝子にリソース情報を内包
3. **制約保持**: 有効な割り当てパターン内でのみ変更

#### 実装での詳細処理

```python
def create_individual(self):
    """
    制約充足を保証する初期個体生成
    """
    # Step 1: 順序制約を満たすタスク順序の生成
    task_order = self.topological_sort_with_randomness()
    
    # Step 2: 各タスクに有効なリソースを割り当て
    chromosome = []
    for task_id in task_order:
        pattern = self.tasks[task_id]['pattern']
        
        # 有効な割り当てパターンからランダム選択
        if pattern in self.allocation_patterns:
            valid_allocations = self.allocation_patterns[pattern]
            allocation = random.choice(valid_allocations)
            
            chromosome.append({
                'task': task_id,
                'worker': allocation['worker'],
                'equipment': allocation['equipment']
            })
        else:
            # フォールバック: ランダム割り当て
            chromosome.append({
                'task': task_id,
                'worker': random.choice(self.workers[:-1]),
                'equipment': random.choice(self.equipments[:-1])
            })
    
    return chromosome
```

### 2. トポロジカルソート＋ランダマイゼーション

#### 問題の課題
- 厳密なトポロジカルソートでは多様性が不足
- 完全ランダムでは制約違反が多発

#### 解決策: 制約付きランダマイゼーション

```python
def topological_sort_with_randomness(self):
    """
    順序制約を満たしつつランダム性を導入
    """
    # 依存関係グラフの構築
    in_degree = {task: 0 for task in self.task_ids}
    graph = {task: [] for task in self.task_ids}
    
    for constraint in self.order_constraints:
        pred = constraint['predecessor']
        succ = constraint['successor'] 
        if pred in graph and succ in in_degree:
            graph[pred].append(succ)
            in_degree[succ] += 1
    
    # 制約付きランダム選択
    result = []
    available = [task for task in self.task_ids if in_degree[task] == 0]
    
    while available:
        # ★ キーポイント: ランダム選択で多様性確保
        random.shuffle(available)
        task = available.pop(0)
        result.append(task)
        
        # 後続タスクの更新
        for successor in graph[task]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                available.append(successor)
    
    # 孤立タスクの追加
    for task in self.task_ids:
        if task not in result:
            result.append(task)
    
    return result
```

## 制約処理の特殊化

### 1. 複雑な時間差制約の処理

#### 4パターンの時間基準点

```python
def calculate_time_constraint(self, schedule, constraint):
    """
    複雑な時間差制約の計算
    
    パターン:
    1. 終了 → 開始: 通常の順序制約
    2. 終了 → 終了: 並行実行制約  
    3. 開始 → 開始: 同期開始制約
    4. 開始 → 終了: 逆順制約
    """
    pred_id = constraint['predecessor']
    succ_id = constraint['successor']
    
    # 基準時刻の取得
    if constraint['pred_origin'] == '終了':
        pred_time = schedule[pred_id]['end_time']
    else:  # '開始'
        pred_time = schedule[pred_id]['start_time']
    
    if constraint['succ_origin'] == '終了':  
        succ_time = schedule[succ_id]['end_time']
    else:  # '開始'
        succ_time = schedule[succ_id]['start_time']
    
    return succ_time - pred_time
```

#### 負の時間差制約への対応

```python
# 例: task-0001の終了180分前からtask-0008が実行可能
{
    'predecessor': 'task-0001',
    'successor': 'task-0008',
    'pred_origin': '終了',
    'succ_origin': '開始', 
    'time_diff_min': -180  # 負値
}

# デコーディング時の処理
if constraint['succ_origin'] == '開始':
    # 後続タスクの最早開始時刻
    min_start = pred_time + constraint['time_diff_min']  # -180
    earliest_start = max(earliest_start, min_start)
else:
    # 後続タスクの最早終了時刻から逆算
    min_end = pred_time + constraint['time_diff_min']
    min_start = min_end - duration
    earliest_start = max(earliest_start, min_start)
```

### 2. 段階的制約充足メカニズム

```python
def decode_chromosome(self, chromosome):
    """
    複数制約を段階的に満たすスケジュール生成
    """
    schedule = {}
    worker_availability = {worker: 0 for worker in self.workers}
    equipment_availability = {equipment: 0 for equipment in self.equipments}
    
    for gene in chromosome:
        task_id = gene['task']
        duration = self.tasks[task_id]['duration']
        
        # Phase 1: リソース制約による最早開始時刻
        earliest_start = max(
            worker_availability.get(gene['worker'], 0),
            equipment_availability.get(gene['equipment'], 0)
        )
        
        # Phase 2: 順序制約による追加制限
        for constraint in self.order_constraints:
            if constraint['successor'] == task_id and constraint['predecessor'] in schedule:
                constraint_start = self.calculate_constraint_start_time(
                    schedule, constraint, duration
                )
                earliest_start = max(earliest_start, constraint_start)
        
        # Phase 3: スケジュール確定
        start_time = earliest_start
        end_time = start_time + duration
        
        schedule[task_id] = {
            'start_time': start_time,
            'end_time': end_time,
            'worker': gene['worker'],
            'equipment': gene['equipment']
        }
        
        # Phase 4: リソース利用可能時刻の更新
        worker_availability[gene['worker']] = end_time
        equipment_availability[gene['equipment']] = end_time
    
    return schedule
```

## 初期個体群生成の最適化

### 1. 多様性を考慮した初期化戦略

```python
def initialize_population_with_diversity(self):
    """
    多様な初期個体群の生成
    
    戦略:
    1. ランダム生成 (40%)
    2. 貪欲法ベース (30%)  
    3. 制約重視 (20%)
    4. リソース均等化 (10%)
    """
    population = []
    
    # Strategy 1: 完全ランダム個体
    for _ in range(int(self.population_size * 0.4)):
        population.append(self.create_random_individual())
    
    # Strategy 2: 貪欲法ベース個体
    for _ in range(int(self.population_size * 0.3)):
        population.append(self.create_greedy_individual())
    
    # Strategy 3: 制約充足重視個体
    for _ in range(int(self.population_size * 0.2)):
        population.append(self.create_constraint_focused_individual())
    
    # Strategy 4: リソース負荷均等化個体
    for _ in range(int(self.population_size * 0.1)):
        population.append(self.create_load_balanced_individual())
    
    return population

def create_greedy_individual(self):
    """
    貪欲法による個体生成（短い処理時間のタスクを優先）
    """
    # タスクを処理時間順にソート
    tasks_by_duration = sorted(
        self.task_ids, 
        key=lambda t: self.tasks[t]['duration']
    )
    
    # トポロジカル制約内で短いタスクを優先配置
    return self.create_individual_with_preference(tasks_by_duration)

def create_load_balanced_individual(self):
    """
    リソース負荷を均等化する個体生成
    """
    # 各リソースの使用頻度をカウント
    worker_load = {worker: 0 for worker in self.workers}
    equipment_load = {equipment: 0 for equipment in self.equipments}
    
    chromosome = []
    task_order = self.topological_sort_with_randomness()
    
    for task_id in task_order:
        pattern = self.tasks[task_id]['pattern']
        valid_allocations = self.allocation_patterns.get(pattern, [])
        
        if valid_allocations:
            # 最も負荷の少ないリソース組み合わせを選択
            best_allocation = min(valid_allocations, 
                key=lambda a: worker_load[a['worker']] + equipment_load[a['equipment']])
            
            chromosome.append({
                'task': task_id,
                'worker': best_allocation['worker'],
                'equipment': best_allocation['equipment']
            })
            
            # 負荷を更新
            duration = self.tasks[task_id]['duration']
            worker_load[best_allocation['worker']] += duration
            equipment_load[best_allocation['equipment']] += duration
    
    return chromosome
```

## 適応度関数の調整

### 1. 動的ペナルティ係数

```python
def calculate_dynamic_penalty_coefficient(self, generation, total_generations):
    """
    世代に応じてペナルティ係数を動的調整
    
    序盤: 低いペナルティで探索を重視
    終盤: 高いペナルティで制約充足を重視
    """
    progress = generation / total_generations
    
    # 序盤: 係数5, 終盤: 係数20
    base_penalty = 5
    max_penalty = 20
    
    return base_penalty + (max_penalty - base_penalty) * progress

def calculate_fitness_with_adaptive_penalty(self, chromosome, generation):
    """
    適応的ペナルティを使用した適応度計算
    """
    schedule = self.decode_chromosome(chromosome)
    if schedule is None:
        return float('inf')
    
    makespan = max([task['end_time'] for task in schedule.values()])
    
    # 動的ペナルティ係数
    penalty_coeff = self.calculate_dynamic_penalty_coefficient(
        generation, self.num_generations
    )
    
    penalty = 0
    for constraint in self.order_constraints:
        violation = self.calculate_constraint_violation(schedule, constraint)
        if violation > 0:
            penalty += violation * penalty_coeff
    
    return makespan + penalty
```

### 2. 多目的最適化への拡張

```python
def calculate_multi_objective_fitness(self, chromosome):
    """
    多目的適応度: メイクスパン + リソース利用効率
    """
    schedule = self.decode_chromosome(chromosome)
    
    # 目的1: メイクスパン最小化
    makespan = max([task['end_time'] for task in schedule.values()])
    
    # 目的2: リソース利用効率最大化
    resource_utilization = self.calculate_resource_utilization(schedule)
    
    # 目的3: 負荷均等化
    load_balance = self.calculate_load_balance(schedule)
    
    # 重み付き結合
    return (0.7 * makespan + 
            0.2 * (1000 - resource_utilization) +  # 利用効率を逆転
            0.1 * load_balance)

def calculate_resource_utilization(self, schedule):
    """
    リソース利用効率の計算
    """
    makespan = max([task['end_time'] for task in schedule.values()])
    
    # 各リソースの稼働時間
    worker_utilization = {}
    equipment_utilization = {}
    
    for task_id, info in schedule.items():
        duration = info['end_time'] - info['start_time']
        
        worker = info['worker']
        equipment = info['equipment']
        
        worker_utilization[worker] = worker_utilization.get(worker, 0) + duration
        equipment_utilization[equipment] = equipment_utilization.get(equipment, 0) + duration
    
    # 平均利用率の計算
    total_worker_time = sum(worker_utilization.values())
    total_equipment_time = sum(equipment_utilization.values())
    
    max_possible_time = makespan * (len(self.workers) + len(self.equipments))
    actual_time = total_worker_time + total_equipment_time
    
    return (actual_time / max_possible_time) * 100
```

## 遺伝的操作の問題特化

### 1. 制約保持型交叉

```python
def constraint_preserving_crossover(self, parent1, parent2):
    """
    制約を保持する特殊化交叉
    
    1. 順序制約を満たす部分配列を特定
    2. 有効な割り当てパターンを保持
    3. 制約違反を最小化
    """
    size = len(parent1)
    
    # Critical Path の特定（制約の多いタスクチェーン）
    critical_tasks = self.identify_critical_path(parent1)
    
    # Critical Path は親1から継承
    child = [None] * size
    critical_positions = []
    
    for i, gene in enumerate(parent1):
        if gene['task'] in critical_tasks:
            child[i] = copy.deepcopy(gene)
            critical_positions.append(i)
    
    # 残りのタスクは親2の順序で配置
    remaining_tasks = [gene for gene in parent2 
                      if gene['task'] not in critical_tasks]
    
    j = 0
    for i in range(size):
        if i not in critical_positions and j < len(remaining_tasks):
            child[i] = copy.deepcopy(remaining_tasks[j])
            j += 1
    
    return child

def identify_critical_path(self, chromosome):
    """
    制約チェーンの特定
    """
    schedule = self.decode_chromosome(chromosome)
    
    # 各タスクの制約数をカウント
    constraint_count = {}
    for constraint in self.order_constraints:
        pred = constraint['predecessor']
        succ = constraint['successor']
        
        constraint_count[pred] = constraint_count.get(pred, 0) + 1
        constraint_count[succ] = constraint_count.get(succ, 0) + 1
    
    # 上位25%を重要タスクとして選択
    sorted_tasks = sorted(constraint_count.items(), key=lambda x: x[1], reverse=True)
    critical_count = max(1, len(sorted_tasks) // 4)
    
    return [task for task, _ in sorted_tasks[:critical_count]]
```

### 2. 適応的突然変異

```python
def adaptive_mutation(self, chromosome, generation, diversity_measure):
    """
    多様性に応じた適応的突然変異
    
    多様性が低い → 探索的突然変異（大きな変化）
    多様性が高い → 活用的突然変異（小さな変化）
    """
    if random.random() > self.mutation_rate:
        return chromosome
    
    mutated = copy.deepcopy(chromosome)
    
    # 多様性に基づく突然変異強度の調整
    if diversity_measure < 0.1:  # 多様性が低い
        # 強い突然変異: 大幅なタスク入れ替え
        num_swaps = random.randint(2, 5)
        for _ in range(num_swaps):
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    else:
        # 弱い突然変異: リソースの微調整
        idx = random.randint(0, len(mutated) - 1)
        task_id = mutated[idx]['task']
        pattern = self.tasks[task_id]['pattern']
        
        if pattern in self.allocation_patterns:
            valid_allocations = self.allocation_patterns[pattern]
            current_allocation = {
                'worker': mutated[idx]['worker'],
                'equipment': mutated[idx]['equipment']
            }
            
            # 現在の割り当て以外から選択
            other_allocations = [
                alloc for alloc in valid_allocations 
                if alloc != current_allocation
            ]
            
            if other_allocations:
                new_allocation = random.choice(other_allocations)
                mutated[idx]['worker'] = new_allocation['worker']
                mutated[idx]['equipment'] = new_allocation['equipment']
    
    return mutated
```

## 性能最適化技法

### 1. 適応度計算の高速化

```python
def fast_fitness_calculation(self, chromosome):
    """
    適応度計算の最適化
    
    1. 増分計算による高速化
    2. 制約チェックの早期終了
    3. キャッシュ機能の活用
    """
    # スケジュールのハッシュ値でキャッシュチェック
    chromosome_hash = self.calculate_chromosome_hash(chromosome)
    
    if chromosome_hash in self.fitness_cache:
        return self.fitness_cache[chromosome_hash]
    
    # 増分的スケジュール構築
    schedule = {}
    worker_availability = {worker: 0 for worker in self.workers}
    equipment_availability = {equipment: 0 for equipment in self.equipments}
    
    makespan = 0
    penalty = 0
    
    for gene in chromosome:
        task_info = self.decode_single_task(
            gene, schedule, worker_availability, equipment_availability
        )
        
        schedule[gene['task']] = task_info
        makespan = max(makespan, task_info['end_time'])
        
        # 制約チェック（早期終了）
        task_penalty = self.check_task_constraints(gene['task'], schedule)
        penalty += task_penalty
        
        # 致命的違反の場合は早期終了
        if penalty > 1000:
            fitness = float('inf')
            break
    else:
        fitness = makespan + penalty
    
    # キャッシュに保存
    self.fitness_cache[chromosome_hash] = fitness
    
    return fitness

def calculate_chromosome_hash(self, chromosome):
    """
    染色体のハッシュ値計算（キャッシュ用）
    """
    hash_string = ""
    for gene in chromosome:
        hash_string += f"{gene['task']}-{gene['worker']}-{gene['equipment']}|"
    
    return hash(hash_string)
```

### 2. 並列適応度評価

```python
from multiprocessing import Pool
import multiprocessing as mp

def parallel_fitness_evaluation(self, population):
    """
    適応度計算の並列化
    """
    # CPUコア数に応じてワーカー数を決定
    num_workers = min(mp.cpu_count(), len(population))
    
    with Pool(num_workers) as pool:
        # 各プロセスに個体を分散
        chunk_size = max(1, len(population) // num_workers)
        fitness_scores = pool.map(
            self.calculate_fitness_wrapper,
            population,
            chunksize=chunk_size
        )
    
    return fitness_scores

def calculate_fitness_wrapper(self, chromosome):
    """
    マルチプロセシング用のラッパー関数
    """
    # プロセス間でのデータ共有を考慮した実装
    try:
        return self.calculate_fitness(chromosome)
    except Exception as e:
        # エラー処理
        return float('inf')
```

### 3. メモリ効率化

```python
def memory_efficient_evolution(self):
    """
    メモリ使用量を抑えた進化プロセス
    """
    # 大きな個体群を部分的に処理
    batch_size = 20
    
    for generation in range(self.num_generations):
        # バッチ処理で適応度評価
        for i in range(0, len(self.population), batch_size):
            batch = self.population[i:i+batch_size]
            batch_fitness = [self.calculate_fitness(ind) for ind in batch]
            
            # 適応度の更新
            for j, fitness in enumerate(batch_fitness):
                self.fitness_scores[i+j] = fitness
        
        # 不要なデータの削除
        if generation % 10 == 0:
            self.cleanup_cache()
    
def cleanup_cache(self):
    """
    キャッシュのクリーンアップ
    """
    # 古いキャッシュエントリの削除
    if len(self.fitness_cache) > 1000:
        # 最新の500エントリのみ保持
        cache_items = list(self.fitness_cache.items())
        self.fitness_cache = dict(cache_items[-500:])
```

これらの問題固有の工夫により、複雑な制約を持つジョブショップスケジューリング問題に対して、効率的で実用的なGAソリューションを実現しています。特に、制約処理の特殊化と性能最適化により、大規模問題でも実用的な時間で高品質な解を得ることが可能です。