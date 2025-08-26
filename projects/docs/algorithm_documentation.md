# Job Shop Scheduling with Additional Constraints: GA Implementation Documentation

## 概要

本システムは、**遺伝的アルゴリズム（Genetic Algorithm, GA）**を用いて、追加制約付きのジョブショップスケジューリング問題を解決します。従来のジョブショップスケジューリングに加えて、**固定タスク**と**連続作業制限**という新たな制約を考慮した最適化を行います。

### システムの背景と必要性

製造業や生産管理の現場では、単純なタスクスケジューリングだけでなく、以下のような複雑な制約が存在します：

- **固定された作業**: 定期メンテナンスや検査など、時間が事前に決まっているタスク
- **作業の連続性制限**: 同じ設備で特定の作業パターンを連続して行うことができない制約

これらの制約を考慮したスケジューリングは、従来の手法では解決が困難であり、遺伝的アルゴリズムのような進化計算手法が有効となります。本システムは、これらの実用的な制約を組み込みながら、全体の作業完了時間（makespan）を最小化することを目標としています。

---

## 1. 遺伝的アルゴリズム（GA）の基礎

### 1.1 GAとは
遺伝的アルゴリズムは、生物の進化過程を模倣した最適化手法です：

- **個体（Individual）**: 問題の解候補
- **個体群（Population）**: 複数の個体の集合
- **染色体（Chromosome）**: 個体の遺伝情報（解の表現）
- **遺伝子（Gene）**: 染色体を構成する要素
- **適応度（Fitness）**: 個体の良さを表す指標

### 1.2 GAの基本的な流れ
```
1. 初期個体群の生成
2. 適応度の評価
3. 選択（Selection）
4. 交叉（Crossover）
5. 突然変異（Mutation）
6. 次世代への更新
7. 終了条件まで2-6を繰り返し
```

### 1.3 なぜGAがスケジューリングに適しているか

ジョブショップスケジューリング問題は、以下の特徴を持つため、GAが適用されます：

- **組み合わせ最適化問題**: タスクの順序と資源の割り当てという離散的な選択が必要
- **制約の複雑さ**: 多数の制約条件が相互に影響し合う
- **解空間の広大さ**: 可能なスケジュールの組み合わせが膨大
- **近似解の許容**: 厳密解よりも現実的な時間で良質な解を得ることが重要

GAは、これらの特徴に対して以下の利点を提供します：

- **柔軟な制約処理**: ペナルティ法により複雑な制約を組み込み可能
- **多様性の維持**: 個体群により複数の解候補を同時に探索
- **局所解回避**: 交叉と突然変異により探索の多様化
- **段階的改善**: 世代を重ねることで解の品質を向上

---

## 2. 問題設定：追加制約付きジョブショップスケジューリング

### 2.1 基本要素
- **通常タスク**: スケジュールで配置を決める必要があるタスク
- **固定タスク**: 開始時刻・終了時刻が既に固定されたタスク
- **作業者（Workers）**: タスクを実行するリソース
- **設備（Equipment）**: タスクで使用される機器

### 2.2 制約条件
1. **リソース制約**: 同一作業者/設備は同時に複数タスクを処理不可
2. **順序制約**: 特定タスク間の先行・後続関係
3. **割り当て制約**: 各タスクは特定の作業者・設備の組み合わせでのみ実行可能
4. **固定タスク制約**: 固定タスクの時間・リソースとの競合回避
5. **連続作業制限**: 特定の設備で特定パターンの連続実行を禁止

### 2.3 最適化目標
**Makespan最小化**：
```
makespan = 終了時刻 - 開始時刻
```
- 開始時刻：固定タスクを含む全タスクの最早開始
- 終了時刻：固定タスクを除く通常タスクの最遅終了

### 2.4 問題の複雑性と挑戦

この問題が従来のジョブショップスケジューリングより困難である理由：

1. **時間的制約の混在**: 固定タスクの絶対時刻制約と通常タスクの相対的制約が混在
2. **制約間の相互作用**: 連続作業制限が順序制約や資源制約と複雑に絡み合う
3. **探索空間の分断**: 固定タスクにより探索空間が非連続的に分割される
4. **動的制約チェック**: 連続作業制限は配置順序に依存するため、動的な検証が必要

これらの複雑さに対処するため、本実装では以下のアプローチを採用しています：

- **階層的制約処理**: ハード制約（必須）とソフト制約（望ましい）を分離
- **段階的配置戦略**: 固定タスクを先に配置し、通常タスクを後から配置
- **適応的ペナルティ**: 制約違反の重要度に応じてペナルティを調整

---

## 3. 染色体設計と遺伝表現

### 3.1 染色体の構造
各染色体は**通常タスクのみ**を表現します（固定タスクは除外）：

```python
chromosome = [
    {'task': 'task-0001', 'worker': 'rsrc-0001', 'equipment': 'rsrc-0010'},
    {'task': 'task-0002', 'worker': 'rsrc-0002', 'equipment': 'rsrc-0015'},
    ...
]
```

### 3.2 遺伝子の意味
- **task**: タスクID（順序情報も含む）
- **worker**: 割り当てられた作業者
- **equipment**: 割り当てられた設備

### 3.3 初期個体群生成

初期個体群の生成では、制約を満たす実行可能解を効率的に作成することが重要です。

#### 生成戦略の考え方

1. **順序制約の事前処理**: トポロジカルソートにより、順序制約を満たすタスク順序を生成
2. **ランダム性の導入**: 同じ制約下でも多様な解を生成するため、ランダム要素を追加
3. **割り当て制約の遵守**: 各タスクに対して許可された作業者・設備の組み合わせのみを選択

```python
def create_individual(self):
    # 1. 順序制約を満たすトポロジカルソート
    task_order = self.topological_sort_with_randomness()
    
    # 2. 各タスクにリソースを割り当て
    for task_id in task_order:
        if task_id in self.regular_task_ids:  # 通常タスクのみ
            pattern = self.tasks[task_id]['pattern']
            allocation = random.choice(self.allocation_patterns[pattern])
            chromosome.append({
                'task': task_id,
                'worker': allocation['worker'],
                'equipment': allocation['equipment']
            })
```

#### なぜ固定タスクを染色体から除外するか

固定タスクを染色体に含めない設計には、以下の利点があります：

- **探索効率の向上**: 変更不可能な遺伝子を除くことで、意味のある探索空間に集中
- **制約満足の確実性**: 固定タスクの制約違反を根本的に回避
- **計算コストの削減**: 染色体サイズが小さくなり、遺伝操作が高速化

---

## 4. デコード処理（染色体→スケジュール変換）

### 4.1 デコード処理の流れ

#### Step 1: 固定タスクでスケジュール初期化
```python
schedule = dict(self.fixed_tasks)  # 固定タスクを事前配置
```

#### Step 2: リソースの忙しい期間記録
```python
# 固定タスクによる占有期間を記録
for task_id, info in self.fixed_tasks.items():
    worker_busy_periods[worker].append((start_time, end_time))
    equipment_busy_periods[equipment].append((start_time, end_time))
```

#### Step 3: 通常タスクの配置
各通常タスクに対して：

1. **空きスロット検索**
   ```python
   earliest_start = self.find_available_slot(
       worker_busy_periods, equipment_busy_periods, duration, worker, equipment
   )
   ```

2. **リソース制約チェック**
   ```python
   earliest_start = max(earliest_start, worker_availability[worker])
   earliest_start = max(earliest_start, equipment_availability[equipment])
   ```

3. **順序制約チェック**
   ```python
   for constraint in self.order_constraints:
       if constraint['successor'] == task_id:
           # 先行タスクとの時間差制約を計算
           min_start = pred_time + constraint['time_diff_min']
           earliest_start = max(earliest_start, min_start)
   ```

4. **連続作業制限チェック**
   ```python
   while not self.check_continuous_constraint(equipment, schedule, pattern, earliest_start):
       earliest_start += 30  # 30分遅らせて再試行
   ```

### 4.2 空きスロット検索アルゴリズム
```python
def find_available_slot(self, worker_busy_periods, equipment_busy_periods, duration, worker, equipment):
    # 固定タスクの最早開始時刻から検索開始
    if self.fixed_tasks:
        earliest_start = min(task['start_time'] for task in self.fixed_tasks.values())
    else:
        earliest_start = 0
    
    # 忙しい期間を統合してソート
    all_busy_periods = worker_busy_periods[worker] + equipment_busy_periods[equipment]
    all_busy_periods.sort()
    
    # 空き時間を順次チェック
    for start, end in all_busy_periods:
        if earliest_start + duration <= start:
            return earliest_start  # このスロットに収まる
        earliest_start = max(earliest_start, end)
    
    return earliest_start
```

---

## 5. 適応度関数

### 5.1 基本適応度計算
```python
def calculate_fitness(self, chromosome):
    schedule = self.decode_chromosome(chromosome)
    
    # Makespan計算
    all_start_times = [task['start_time'] for task in schedule.values()]
    start_point = min(all_start_times)
    
    regular_end_times = [
        task['end_time'] for task_id, task in schedule.items() 
        if task_id not in self.fixed_tasks
    ]
    end_point = max(regular_end_times)
    
    makespan = end_point - start_point
    
    # ペナルティ追加
    penalty = self.calculate_constraint_penalty(schedule)
    
    return makespan + penalty
```

### 5.2 制約違反ペナルティ

#### 順序制約違反
```python
if time_diff < constraint['time_diff_min']:
    penalty += (constraint['time_diff_min'] - time_diff) * 10
```

#### 連続作業制限違反
```python
for constraint in constraints:
    if constraint['prev_pattern'] == curr_pattern and constraint['next_pattern'] == next_pattern:
        penalty += 100  # 大きなペナルティ
```

---

## 6. 遺伝操作

### 6.1 選択（Selection）
**トーナメント選択**を使用：
```python
def selection(self, population, fitness_scores):
    tournament_size = 3
    for _ in range(len(population)):
        # 3個体からランダムに選択
        tournament_idx = random.sample(range(len(population)), tournament_size)
        # 最も適応度の高い個体を選択
        winner_idx = tournament_idx[np.argmin(tournament_fitness)]
```

### 6.2 交叉（Crossover）
**順序交叉（Order Crossover, OX）**を使用：
```python
def crossover(self, parent1, parent2):
    # 1. 交叉点をランダムに決定
    start, end = sorted(random.sample(range(size), 2))
    
    # 2. 親1の部分を子1にコピー
    child1[start:end] = parent1[start:end]
    
    # 3. 残りの遺伝子を親2から順番に埋める
    # （既に含まれているタスクは除外）
```

### 6.3 突然変異（Mutation）
2種類の突然変異を確率的に適用：

#### タスク順序の入れ替え
```python
if random.random() < 0.5:
    idx1, idx2 = random.sample(range(len(mutated)), 2)
    mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
```

#### リソースの再割り当て
```python
if random.random() < 0.5:
    pattern = self.tasks[task_id]['pattern']
    allocation = random.choice(self.allocation_patterns[pattern])
    mutated[idx]['worker'] = allocation['worker']
    mutated[idx]['equipment'] = allocation['equipment']
```

---

## 7. 連続作業制限の実装

### 7.0 連続作業制限の背景と重要性

製造現場では、同一設備で特定の作業パターンを連続して実行することが品質や安全上の理由で禁止される場合があります：

- **品質管理**: 連続する同種の作業により設備の精度が低下する可能性
- **安全性**: 特定パターンの連続作業により事故リスクが増加
- **効率性**: 作業パターンの切り替えにより全体的な効率向上

この制約は、タスクの実行順序に動的に依存するため、従来の静的な制約処理では対応が困難です。

### 7.1 制約チェックアルゴリズム
```python
def check_continuous_constraint(self, equipment, schedule, new_task_pattern, start_time):
    # この設備のタスクを時系列順にソート
    equipment_tasks = []
    for task_id, info in schedule.items():
        if info['equipment'] == equipment:
            pattern = get_task_pattern(task_id)
            equipment_tasks.append((info['end_time'], pattern))
    
    equipment_tasks.sort()
    
    # 直前のタスクのパターンを確認
    prev_pattern = None
    for end_time, pattern in equipment_tasks:
        if end_time <= start_time:
            prev_pattern = pattern
    
    # 連続作業制限をチェック
    if prev_pattern:
        for constraint in self.continuous_constraints[equipment]:
            if constraint['prev_pattern'] == prev_pattern and constraint['next_pattern'] == new_task_pattern:
                return False  # 制限違反
    
    return True
```

### 7.2 制限違反時の対応
制限違反が検出された場合：
1. タスクの開始時刻を30分遅らせる
2. 最大10回まで再試行
3. それでも違反する場合はペナルティで処理

---

## 8. アルゴリズムの全体フロー

### 8.1 メインループ
```python
def run(self):
    # 1. 初期個体群生成
    population = [self.create_individual() for _ in range(self.population_size)]
    
    for generation in range(self.num_generations):
        # 2. 適応度評価
        fitness_scores = [self.calculate_fitness(ind) for ind in population]
        
        # 3. エリート保存
        elite_size = int(self.population_size * self.elite_rate)
        elite = select_elite(population, fitness_scores, elite_size)
        
        # 4. 選択
        selected = self.selection(population, fitness_scores)
        
        # 5. 交叉と突然変異
        offspring = []
        for i in range(0, len(selected) - 1, 2):
            child1, child2 = self.crossover(selected[i], selected[i + 1])
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            offspring.extend([child1, child2])
        
        # 6. 次世代構成
        population = elite + offspring[:self.population_size - elite_size]
```

### 8.2 パラメータ設定
```python
population_size = 50        # 個体群サイズ
crossover_rate = 0.8       # 交叉率
mutation_rate = 0.3        # 突然変異率
elite_rate = 0.1           # エリート保存率
num_generations = 500      # 世代数
```

---

## 9. 実装上の工夫と特徴

### 9.0 アルゴリズム設計の哲学

本実装では、以下の設計思想に基づいてアルゴリズムを構築しています：

#### 制約の階層化
制約を重要度に応じて階層化し、適切な処理方法を選択：
- **ハード制約**: 絶対に違反してはならない制約（リソース競合、固定タスクの時刻）
- **ソフト制約**: 可能な限り満たすべき制約（連続作業制限、効率性）

#### 段階的最適化
複雑な制約を一度に処理するのではなく、段階的にアプローチ：
1. 基本的な実行可能解の生成
2. ソフト制約の段階的満足
3. 最適性の向上

#### 動的適応
スケジューリング過程で動的に制約を評価し、適応的に対応：
- 制約違反時の修復メカニズム
- 探索の多様性維持
- 局所解からの脱出

## 10. 特徴と利点

### 9.1 固定タスクへの対応
- 固定タスクは染色体に含めず、デコード時に事前配置
- 通常タスクは固定タスクの占有期間を避けて配置
- makespanは固定タスクを考慮した新しい計算方法を採用

### 9.2 連続作業制限への対応
- 設備ごとの作業パターンを時系列で追跡
- 禁止された連続パターンを動的にチェック
- 違反時は時刻をずらして再配置を試行

### 9.3 柔軟な制約処理
- ハードな制約違反は不可能解として扱う
- ソフトな制約違反はペナルティで対応
- 制約の優先度に応じてペナルティ重みを調整

### 10.4 効率的な探索
- トポロジカルソートによる実行可能解の生成
- 順序交叉による解の多様性保持
- エリート保存による良解の維持

### 10.5 実用性への配慮

本実装は、学術的な完璧さよりも実用性を重視して設計されています：

#### 計算時間と解品質のバランス
- 厳密解を求める代わりに、現実的な時間で実用的な解を提供
- パラメータ調整により、用途に応じた最適化レベルを選択可能

#### 制約違反への柔軟な対応  
- 完全に制約を満たす解が存在しない場合でも、制約違反を最小化した解を提供
- ペナルティの重み付けにより、制約の重要度を調整可能

#### 拡張性とメンテナンス性
- 新たな制約や目的関数の追加が容易な構造
- モジュール化により、部分的な改修が可能

---

## 11. 拡張性と改善点

### 11.1 将来的な拡張可能性

本システムの設計により、以下の拡張が比較的容易に実現可能です：

#### 制約の拡張
- **リアルタイム制約変更**: 運用中の制約条件の動的更新
- **確率的制約**: 不確実性を考慮した制約の組み込み
- **階層的制約**: より複雑な制約関係の表現

#### 目的関数の多様化
- **多目的最適化**: makespan以外の目標（コスト、品質、エネルギー効率）
- **重み付き目的**: 複数目標の重要度に応じた最適化
- **動的目的**: 時間や状況に応じた目標の変更

#### アルゴリズムの高度化
- **ハイブリッド手法**: GAと他のメタヒューリスティクスの組み合わせ
- **適応的パラメータ**: 探索進捗に応じた自動パラメータ調整
- **並列処理**: マルチコアや分散環境での高速化

### 11.2 実装上の改善点

#### 計算効率の向上
現在の実装で改善可能な点：

- **空きスロット検索の最適化**: より効率的なデータ構造の利用
- **制約チェックの高速化**: 事前計算とキャッシュの活用
- **メモリ使用量の削減**: 不要なデータ構造の最適化

#### アルゴリズムの精度向上
- **初期解生成の改善**: ヒューリスティクスを用いた賢い初期化
- **局所探索の組み合わせ**: GAと局所探索の融合
- **制約違反修復**: より効果的な制約違反回避メカニズム

---

## 12. まとめ

### 12.1 本システムの意義

本実装は、固定タスクと連続作業制限という追加制約を考慮した遺伝的アルゴリズムによるジョブショップスケジューリングシステムです。従来のGAの基本構造を保ちながら、複雑な制約を効率的に処理し、実用的なスケジューリング問題に対応できる柔軟性を持っています。

### 12.2 技術的貢献

#### 制約処理の革新
- **固定タスクの効率的処理**: 染色体から除外することで探索効率を大幅に向上
- **動的制約チェック**: 連続作業制限を実行時に動的評価する新しいアプローチ
- **階層的制約管理**: ハード制約とソフト制約を適切に分離した処理

#### アルゴリズム設計の工夫
- **段階的最適化**: 複雑な問題を段階的に解決する現実的なアプローチ
- **適応的ペナルティ**: 制約違反の重要度に応じた柔軟な処罰機構
- **探索空間の効率化**: 意味のある探索に集中する設計思想

### 12.3 実用的価値

本システムは、以下の実用的価値を提供します：

- **現実的制約への対応**: 製造現場で実際に発生する複雑な制約を組み込み
- **柔軟な運用**: 制約の重要度や目標に応じた調整が可能
- **拡張性**: 新たな制約や要求への対応が容易な構造

### 12.4 今後の展望

本システムは、ジョブショップスケジューリングの実用化に向けた重要な一歩です。固定タスクと連続作業制限という具体的な制約に対する解決策を示すことで、他の複雑な制約への応用の道筋も見えてきます。

製造業のDXが進む中で、このような柔軟で実用的なスケジューリングシステムの需要は今後ますます高まると予想されます。本実装が、その基盤技術の一つとして貢献できれば幸いです。