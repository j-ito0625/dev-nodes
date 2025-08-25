# GA実装詳細ドキュメント

## 目次
1. [クラス構造](#クラス構造)
2. [初期化プロセス](#初期化プロセス)
3. [遺伝的操作の詳細実装](#遺伝的操作の詳細実装)
4. [適応度評価システム](#適応度評価システム)
5. [スケジュールデコーディング](#スケジュールデコーディング)
6. [検証システム](#検証システム)

## クラス構造

### JobShopGAScheduler クラス

```python
class JobShopGAScheduler:
    """
    遺伝的アルゴリズムを用いたジョブショップスケジューラ
    """
    def __init__(self, dataset_path: str)
    def load_data(self, dataset_path: str)  # データ読み込み
    def process_data(self)                  # データ前処理
    def initialize_parameters(self)         # パラメータ初期化
    def create_individual(self)             # 個体生成
    def calculate_fitness(self, chromosome) # 適応度計算
    def decode_chromosome(self, chromosome) # スケジュール生成
    def crossover(self, parent1, parent2)   # 交叉
    def mutate(self, chromosome)            # 突然変異
    def selection(self, population, fitness_scores)  # 選択
    def run(self)                          # GA実行
    def validate_schedule(self, chromosome) # スケジュール検証
    def visualize_schedule(self, chromosome) # 可視化
```

### 主要データ構造

#### 1. 染色体（Chromosome）の構造

```python
chromosome = [
    {
        'task': 'task-0001',      # タスクID
        'worker': 'rsrc-0002',    # 作業者ID
        'equipment': 'rsrc-0010'  # 設備ID
    },
    {
        'task': 'task-0002',
        'worker': 'rsrc-0003',
        'equipment': 'rsrc-0012'
    },
    # ... 全タスク分
]
```

**設計思想**:
- 各遺伝子がタスクの完全な割り当て情報を保持
- 染色体の順序 = タスクの実行順序
- リソース情報も遺伝子に含める（ハイブリッド表現）

#### 2. スケジュール表現

```python
schedule = {
    'task-0001': {
        'start_time': 0.0,
        'end_time': 30.0,
        'worker': 'rsrc-0002',
        'equipment': 'rsrc-0010'
    },
    'task-0002': {
        'start_time': 30.0,
        'end_time': 45.0,
        'worker': 'rsrc-0003',
        'equipment': 'rsrc-0012'
    }
    # ... 全タスク分
}
```

## 初期化プロセス

### 1. データ読み込み（load_data）

```python
def load_data(self, dataset_path: str):
    # 4つのExcelシートを読み込み
    self.resources_df = pd.read_excel(dataset_path, sheet_name='リソース')
    self.tasks_df = pd.read_excel(dataset_path, sheet_name='task') 
    self.allocation_df = pd.read_excel(dataset_path, sheet_name='割り当て情報')
    self.order_constraints_df = pd.read_excel(dataset_path, sheet_name='順序制約')
    
    self.process_data()
```

### 2. データ前処理（process_data）

#### タスク情報の辞書化
```python
self.tasks = {}
for _, row in self.tasks_df.iterrows():
    task_id = row['タスクID']
    self.tasks[task_id] = {
        'duration': row['所要時間'],      # 作業時間（分）
        'pattern': row['割り当て可能パターン'],  # 制約パターンID
        'product': row['品種目']         # 製品種別
    }
```

#### 割り当て制約の構造化
```python
self.allocation_patterns = {}
for _, row in self.allocation_df.iterrows():
    pattern = row['割り当て可能パターン']
    if pattern not in self.allocation_patterns:
        self.allocation_patterns[pattern] = []
    self.allocation_patterns[pattern].append({
        'worker': row['作業者'],
        'equipment': row['設備']
    })

# 例: 'procedure_node_00001' -> [
#   {'worker': 'rsrc-0002', 'equipment': 'rsrc-0010'},
#   {'worker': 'rsrc-0004', 'equipment': 'rsrc-0008'}
# ]
```

#### 順序制約の処理
```python
self.order_constraints = []
for _, row in self.order_constraints_df.iterrows():
    self.order_constraints.append({
        'predecessor': row['先行作業ID'],    # 'task-0001'
        'successor': row['後作業ID'],       # 'task-0005'
        'pred_origin': row['先行作業原点'],   # '終了' or '開始'
        'succ_origin': row['後作業原点'],    # '終了' or '開始'
        'time_diff_min': row['時間差下限']   # -180 ~ 120 (分)
    })
```

### 3. 初期個体生成（create_individual）

```python
def create_individual(self):
    chromosome = []
    
    # 1. トポロジカルソート＋ランダマイゼーション
    task_order = self.topological_sort_with_randomness()
    
    # 2. 各タスクにリソースを割り当て
    for task_id in task_order:
        pattern = self.tasks[task_id]['pattern']
        
        # 有効な割り当てパターンからランダム選択
        if pattern in self.allocation_patterns:
            allocation = random.choice(self.allocation_patterns[pattern])
            chromosome.append({
                'task': task_id,
                'worker': allocation['worker'],
                'equipment': allocation['equipment']
            })
    
    return chromosome
```

#### トポロジカルソートの工夫

```python
def topological_sort_with_randomness(self):
    """
    順序制約を満たしつつランダム性を導入したトポロジカルソート
    """
    # 入次数を計算
    in_degree = {task: 0 for task in self.task_ids}
    graph = {task: [] for task in self.task_ids}
    
    for constraint in self.order_constraints:
        pred = constraint['predecessor']  
        succ = constraint['successor']
        if pred in graph and succ in in_degree:
            graph[pred].append(succ)
            in_degree[succ] += 1
    
    # 入次数0のタスクから開始
    result = []
    available = [task for task in self.task_ids if in_degree[task] == 0]
    
    while available:
        # ★ ランダム選択で多様性を導入
        random.shuffle(available)
        task = available.pop(0)
        result.append(task)
        
        # 後続タスクの入次数を減らす
        for successor in graph[task]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                available.append(successor)
    
    return result
```

## 遺伝的操作の詳細実装

### 1. 選択（Selection） - トーナメント選択

```python
def selection(self, population, fitness_scores):
    """
    トーナメント選択による親の選択
    - 適度な選択圧で多様性を維持
    - エリート戦略との組み合わせで良解を保存
    """
    tournament_size = 3
    selected = []
    
    for _ in range(len(population)):
        # ランダムに3個体を選んでトーナメント
        tournament_idx = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_idx]
        
        # 最良個体を勝者として選択
        winner_idx = tournament_idx[np.argmin(tournament_fitness)]
        selected.append(copy.deepcopy(population[winner_idx]))
    
    return selected
```

**なぜトーナメント選択か？**
- ルーレット選択: 適応度差が小さいと選択圧が不十分
- エリート選択: 多様性を損なう可能性
- **トーナメント選択**: 適度な選択圧で多様性とエリート化のバランス

### 2. 交叉（Crossover） - 順序交叉（OX）

```python
def crossover(self, parent1, parent2):
    """
    順序交叉（Order Crossover: OX）
    
    1. 親1から部分配列をコピー
    2. 親2の順序で残りを埋める
    3. リソース情報も適切に継承
    """
    if random.random() > self.crossover_rate:
        return parent1, parent2
    
    size = len(parent1)
    
    # 交叉点を2つランダムに選択
    start, end = sorted(random.sample(range(size), 2))
    
    # 子個体の初期化
    child1 = [None] * size
    child2 = [None] * size
    
    # 選択された範囲をコピー
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]
    
    # 残りの位置を順序を保って埋める
    def fill_remaining(child, parent):
        # 既に配置されたタスクを特定
        tasks_in_child = {gene['task'] for gene in child if gene is not None}
        
        # 親の順序で残りタスクを取得
        remaining = [gene for gene in parent if gene['task'] not in tasks_in_child]
        
        # 空いている位置に順番に配置
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = remaining[idx]
                idx += 1
        return child
    
    child1 = fill_remaining(child1, parent2)
    child2 = fill_remaining(child2, parent1)
    
    return child1, child2
```

**OXの利点**:
- タスクの順序情報を保持
- 親の良い部分配列を子に継承
- 制約違反を最小限に抑制

### 3. 突然変異（Mutation）

```python
def mutate(self, chromosome):
    """
    2段階突然変異
    1. タスク順序の入れ替え（探索性向上）
    2. リソースの再割り当て（制約内最適化）
    """
    if random.random() > self.mutation_rate:
        return chromosome
    
    mutated = copy.deepcopy(chromosome)
    
    # Phase 1: タスク順序の変更（50%確率）
    if random.random() < 0.5:
        idx1, idx2 = random.sample(range(len(mutated)), 2)
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    
    # Phase 2: リソース再割り当て（50%確率）
    if random.random() < 0.5:
        idx = random.randint(0, len(mutated) - 1)
        task_id = mutated[idx]['task']
        pattern = self.tasks[task_id]['pattern']
        
        # 有効な割り当てパターン内でリソース変更
        if pattern in self.allocation_patterns:
            allocation = random.choice(self.allocation_patterns[pattern])
            mutated[idx]['worker'] = allocation['worker']
            mutated[idx]['equipment'] = allocation['equipment']
    
    return mutated
```

**突然変異の設計思想**:
- **マクロ突然変異**: タスク順序変更で大域的探索
- **ミクロ突然変異**: リソース再割り当てで局所最適化
- **制約保持**: 有効な割り当てパターン内でのみ変更

## 適応度評価システム

### 1. 適応度計算の全体構造

```python
def calculate_fitness(self, chromosome):
    """
    適応度 = メイクスパン + ペナルティ
    
    良い個体ほど小さい値を持つ（最小化問題）
    """
    # スケジュールをデコード
    schedule = self.decode_chromosome(chromosome)
    
    if schedule is None:
        return float('inf')  # デコード失敗（重大な制約違反）
    
    # メイクスパン計算
    makespan = max([task['end_time'] for task in schedule.values()])
    
    # 制約違反ペナルティ
    penalty = self.calculate_constraint_penalty(schedule)
    
    return makespan + penalty
```

### 2. ペナルティ関数の詳細

```python
def calculate_constraint_penalty(self, schedule):
    """
    順序制約違反に対するペナルティ計算
    
    penalty = Σ(違反量 × ペナルティ係数)
    """
    penalty = 0
    
    for constraint in self.order_constraints:
        pred_id = constraint['predecessor']
        succ_id = constraint['successor']
        
        if pred_id in schedule and succ_id in schedule:
            # 基準時刻の取得
            pred_time = (schedule[pred_id]['end_time'] 
                        if constraint['pred_origin'] == '終了' 
                        else schedule[pred_id]['start_time'])
            
            succ_time = (schedule[succ_id]['end_time'] 
                        if constraint['succ_origin'] == '終了'
                        else schedule[succ_id]['start_time'])
            
            # 時間差の計算
            time_diff = succ_time - pred_time
            
            # 制約違反の場合ペナルティ追加
            if time_diff < constraint['time_diff_min']:
                violation = constraint['time_diff_min'] - time_diff
                penalty += violation * 10  # ペナルティ係数
    
    return penalty
```

**ペナルティ係数の設計**:
- `10`: 制約違反の重要度を表現
- メイクスパンと同等の重みで制約充足を促進
- 段階的に制約を満たす解へ収束

## スケジュールデコーディング

### decode_chromosome の詳細実装

```python
def decode_chromosome(self, chromosome):
    """
    染色体を実際のスケジュールに変換
    
    プロセス:
    1. リソースの利用可能時刻を管理
    2. 順序制約を考慮して開始時刻を決定
    3. タスクをスケジュールに配置
    """
    schedule = {}
    
    # リソースの利用可能時刻を初期化
    worker_availability = {worker: 0 for worker in self.workers}
    equipment_availability = {equipment: 0 for equipment in self.equipments}
    
    # 染色体の順序でタスクを処理
    for gene in chromosome:
        task_id = gene['task']
        worker = gene['worker']
        equipment = gene['equipment']
        duration = self.tasks[task_id]['duration']
        
        # 最早開始時刻の計算
        earliest_start = max(
            worker_availability.get(worker, 0),      # 作業者の空き時刻
            equipment_availability.get(equipment, 0)  # 設備の空き時刻
        )
        
        # 順序制約の考慮
        for constraint in self.order_constraints:
            if constraint['successor'] == task_id:
                pred_id = constraint['predecessor']
                
                if pred_id in schedule:
                    # 先行タスクの基準時刻
                    pred_time = (schedule[pred_id]['end_time'] 
                                if constraint['pred_origin'] == '終了'
                                else schedule[pred_id]['start_time'])
                    
                    # 制約に基づく最早開始時刻
                    if constraint['succ_origin'] == '開始':
                        min_start = pred_time + constraint['time_diff_min']
                        earliest_start = max(earliest_start, min_start)
                    else:  # '終了'
                        min_end = pred_time + constraint['time_diff_min']
                        min_start = min_end - duration
                        earliest_start = max(earliest_start, min_start)
        
        # スケジュールエントリの作成
        start_time = earliest_start
        end_time = start_time + duration
        
        schedule[task_id] = {
            'start_time': start_time,
            'end_time': end_time,
            'worker': worker,
            'equipment': equipment
        }
        
        # リソース利用可能時刻の更新
        worker_availability[worker] = end_time
        equipment_availability[equipment] = end_time
    
    return schedule
```

**デコーディングの工夫点**:

1. **リソース競合の自動解決**
   - 同一リソースは先着順で自動的にスケジュール
   - 明示的な競合検出・解決ロジック不要

2. **複雑な順序制約の処理**
   - 「開始-開始」「開始-終了」「終了-開始」「終了-終了」の4パターンに対応
   - 負の時間差（先行実行許可）も正確に処理

3. **段階的制約充足**
   - 最早開始時刻を段階的に更新
   - すべての制約を同時に考慮

## 検証システム

### 包括的検証（validate_schedule）

```python
def validate_schedule(self, chromosome):
    """
    スケジュールの完全性を多角的に検証
    
    検証項目:
    1. メイクスパンの正確性
    2. 順序制約の充足
    3. リソース競合の有無  
    4. 割り当て制約の遵守
    """
    schedule = self.decode_chromosome(chromosome)
    validation_results = {
        'is_valid': True,
        'makespan': 0,
        'violations': [],
        'resource_conflicts': [],
        'constraint_violations': []
    }
    
    # 1. メイクスパン計算
    if schedule:
        validation_results['makespan'] = max([
            task['end_time'] for task in schedule.values()
        ])
    
    # 2. 順序制約の検証
    for constraint in self.order_constraints:
        pred_id = constraint['predecessor']
        succ_id = constraint['successor']
        
        if pred_id in schedule and succ_id in schedule:
            pred_time = (schedule[pred_id]['end_time'] 
                        if constraint['pred_origin'] == '終了'
                        else schedule[pred_id]['start_time'])
            succ_time = (schedule[succ_id]['end_time']
                        if constraint['succ_origin'] == '終了'
                        else schedule[succ_id]['start_time'])
            
            time_diff = succ_time - pred_time
            
            if time_diff < constraint['time_diff_min']:
                validation_results['is_valid'] = False
                validation_results['constraint_violations'].append({
                    'type': 'order_constraint',
                    'predecessor': pred_id,
                    'successor': succ_id,
                    'required_min_diff': constraint['time_diff_min'],
                    'actual_diff': time_diff,
                    'violation_amount': constraint['time_diff_min'] - time_diff
                })
    
    # 3. リソース競合の検証
    # 作業者の競合チェック
    worker_schedules = {}
    for task_id, info in schedule.items():
        worker = info['worker']
        if worker not in worker_schedules:
            worker_schedules[worker] = []
        worker_schedules[worker].append(
            (task_id, info['start_time'], info['end_time'])
        )
    
    for worker, tasks in worker_schedules.items():
        tasks_sorted = sorted(tasks, key=lambda x: x[1])  # 開始時刻順
        for i in range(len(tasks_sorted) - 1):
            if tasks_sorted[i][2] > tasks_sorted[i+1][1]:  # 重複あり
                validation_results['is_valid'] = False
                validation_results['resource_conflicts'].append({
                    'type': 'worker_conflict',
                    'resource': worker,
                    'task1': tasks_sorted[i][0],
                    'task2': tasks_sorted[i+1][0],
                    'overlap': tasks_sorted[i][2] - tasks_sorted[i+1][1]
                })
    
    # 設備の競合チェック（同様の処理）
    # ...
    
    # 4. 割り当て制約の検証
    for gene in chromosome:
        task_id = gene['task']
        worker = gene['worker']  
        equipment = gene['equipment']
        pattern = self.tasks[task_id]['pattern']
        
        if pattern in self.allocation_patterns:
            valid_allocations = self.allocation_patterns[pattern]
            is_valid_allocation = any(
                alloc['worker'] == worker and alloc['equipment'] == equipment
                for alloc in valid_allocations
            )
            
            if not is_valid_allocation:
                validation_results['is_valid'] = False
                validation_results['violations'].append({
                    'type': 'allocation_violation',
                    'task': task_id,
                    'assigned_worker': worker,
                    'assigned_equipment': equipment,
                    'allowed_patterns': valid_allocations
                })
    
    return validation_results
```

### 検証レポート出力

```python
def print_validation_report(self, chromosome):
    """
    検証結果の詳細レポート出力
    """
    results = self.validate_schedule(chromosome)
    
    print("\n" + "="*60)
    print("SCHEDULE VALIDATION REPORT")
    print("="*60)
    
    print(f"\n✓ Makespan: {results['makespan']:.2f} minutes")
    print(f"✓ Schedule is valid: {results['is_valid']}")
    
    if not results['is_valid']:
        print("\n⚠ VIOLATIONS DETECTED:")
        
        # 順序制約違反の報告
        if results['constraint_violations']:
            print("\n[Order Constraint Violations]")
            for v in results['constraint_violations']:
                print(f"  - {v['predecessor']} -> {v['successor']}: ")
                print(f"    Required min diff: {v['required_min_diff']}, "
                      f"Actual: {v['actual_diff']:.2f}")
                print(f"    Violation amount: {v['violation_amount']:.2f} minutes")
        
        # リソース競合の報告
        if results['resource_conflicts']:
            print("\n[Resource Conflicts]")
            for c in results['resource_conflicts']:
                print(f"  - {c['type']} on {c['resource']}: ")
                print(f"    Tasks {c['task1']} and {c['task2']} "
                      f"overlap by {c['overlap']:.2f} minutes")
        
        # 割り当て制約違反の報告  
        if results['violations']:
            print("\n[Allocation Violations]")
            for v in results['violations']:
                if v['type'] == 'allocation_violation':
                    print(f"  - Task {v['task']}: ")
                    print(f"    Assigned: Worker={v['assigned_worker']}, "
                          f"Equipment={v['assigned_equipment']}")
                    print(f"    Not in allowed patterns")
    else:
        print("\n✓ All constraints satisfied!")
        print("✓ No resource conflicts detected!")  
        print("✓ All allocation patterns valid!")
    
    print("="*60)
    
    return results
```

この検証システムにより、GA が生成した解の妥当性を多面的に確認し、問題の要件を完全に満たしているかを判定できます。