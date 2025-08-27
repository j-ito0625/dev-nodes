# GA Job Shop Scheduler V2 アルゴリズム解説

## 概要
`ga_jobshop_scheduler_additional_v2.py`は、追加制約（固定タスク・連続作業制限）に対応したジョブショップスケジューリング問題を解く遺伝的アルゴリズム（GA）の基本実装です。V2は仕様変更に対応した安定版で、標準的なGAアプローチを採用しています。

## 主な特徴

### 1. 問題設定
- **基本制約**: 
  - 順序制約（先行・後続関係）
  - リソース制約（作業者・設備の排他利用）
- **追加制約（仕様変更対応）**:
  - **固定タスク**: 特定時刻・リソース（rsrc_0005, rsrc_0046）固定
  - **連続作業制限**: 設備上で連続実行禁止のパターン組み合わせ

### 2. V2の仕様変更点

#### 2.1 連続作業制限の解釈
- **時間差は考慮しない**: タスク間の時間的間隔に関わらず、設備上で順序的に隣接する場合は制限対象
- **判定基準**: 同一設備で実行順序が連続する場合のみチェック

#### 2.2 Makespan計算方法
```python
# 開始時刻：絶対時刻0で固定
start_point = 0
# 終了時刻：固定タスクを除く通常タスクの最遅終了
regular_end_times = [task['end_time'] for task_id, task in schedule.items() 
                     if task_id not in self.fixed_tasks]
end_point = max(regular_end_times) if regular_end_times else 0
makespan = end_point - start_point
```

### 3. アルゴリズム構成

#### 3.1 染色体表現
```python
chromosome = [
    {'task': 'task_0001', 'worker': 'rsrc_0001', 'equipment': 'rsrc_0010'},
    {'task': 'task_0002', 'worker': 'rsrc_0002', 'equipment': 'rsrc_0011'},
    ...
]
```
- タスクの実行順序とリソース割り当てを表現
- 固定タスクは染色体に含まず、デコード時に考慮

#### 3.2 初期化
```python
def create_individual(self):
    # トポロジカルソートでタスク順序を決定
    task_order = self.topological_sort_with_randomness()
    # 各タスクにリソースを割り当て
    for task_id in task_order:
        allocation = random.choice(self.allocation_patterns[pattern])
```
- **順序制約を考慮**: トポロジカルソートで実行可能順序を生成
- **ランダム性**: 制約を満たす範囲でランダムに順序決定

#### 3.3 適応度関数
```python
fitness = makespan + penalty
```
- **主目的**: Makespan（全体完了時間）の最小化
- **ペナルティ**: 制約違反に対する罰則
  - 順序制約違反: `違反量 × 10`
  - 連続作業制限違反: `100`（大きなペナルティ）

### 4. 遺伝的操作

#### 4.1 選択（トーナメント選択）
```python
tournament_size = 3
# 3個体から最良を選択
```
- 選択圧を適度に保ちながら多様性を維持

#### 4.2 交叉（順序交叉 OX）
```python
def crossover(self, parent1, parent2):
    # 部分列を保持しつつ交叉
    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent[start:end]
    # 残りを順序を保って埋める
```
- タスクの順序関係を崩さない交叉方法
- 80%の確率で実行

#### 4.3 突然変異
1. **タスク入れ替え** (50%): 2つのタスクの位置を交換
2. **リソース再割り当て** (50%): タスクの作業者・設備を変更
- 変異率: 30%

### 5. 制約処理

#### 5.1 固定タスクの処理
```python
def decode_chromosome(self, chromosome):
    # 固定タスクでスケジュールを初期化
    schedule = dict(self.fixed_tasks)
    # 固定タスクを避けて通常タスクを配置
    earliest_start = self.find_available_slot(...)
```

#### 5.2 連続作業制限のチェック
```python
def check_continuous_constraint(self, equipment, schedule, new_task_pattern, start_time):
    # 設備上のタスクを時系列順にソート
    equipment_tasks.sort()
    # 前後のタスクパターンをチェック
    if constraint['prev_pattern'] == prev_pattern and 
       constraint['next_pattern'] == new_task_pattern:
        return False  # 制限違反
```

### 6. パラメータ設定（V2標準）

| パラメータ | 値 | 説明 |
|---------|-----|------|
| 集団サイズ | 50 | 標準的なサイズ |
| 世代数 | 500 | 十分な探索時間 |
| 交叉率 | 0.8 | 高い交叉率で探索促進 |
| 突然変異率 | 0.3 | 適度な多様性維持 |
| エリート率 | 0.1 | 優良解を10%保存 |
| トーナメントサイズ | 3 | 適度な選択圧 |

### 7. アルゴリズムフロー

```
1. 初期集団生成（50個体）
   └─ トポロジカルソート + ランダムリソース割当

2. 世代ループ（500世代）
   ├─ 適応度評価
   │   └─ makespan + 制約違反ペナルティ
   ├─ エリート選択（上位10%）
   ├─ トーナメント選択
   ├─ 交叉（OX、80%）
   ├─ 突然変異（30%）
   └─ 次世代生成

3. 最良解の出力
```

### 8. V2の特徴と制限

#### 長所
- **シンプル**: 標準的なGA実装で理解しやすい
- **安定性**: 基本的な遺伝的操作で安定動作
- **汎用性**: 様々な問題サイズに対応

#### 短所
- **収束速度**: 制約が多い場合に収束が遅い
- **局所最適**: 脱出メカニズムが限定的
- **固定ペナルティ**: 問題に応じた調整が必要

### 9. デバッグ版の使い分け

#### 固定タスクのみ版（`ga_jobshop_scheduler_v2_fixed_only.py`）
- 連続作業制限を無効化
- 固定タスクの影響を単独で分析

#### 連続制限のみ版（`ga_jobshop_scheduler_v2_continuous_only.py`）
- 固定タスクを無視
- 連続作業制限の影響を単独で分析

### 10. V3との比較

| 項目 | V2 | V3 |
|------|-----|-----|
| 初期化 | ランダム | ヒューリスティック |
| ペナルティ | 固定 | 適応的 |
| 突然変異 | 2種類 | 4種類 |
| 局所探索 | なし | あり |
| 停滞対処 | なし | 多様性注入 |
| 収束速度 | 標準 | 2-3倍高速 |

### 11. 使用方法

```python
from ga_jobshop_scheduler_additional_v2 import JobShopGASchedulerAdditionalV2

# スケジューラの初期化
scheduler = JobShopGASchedulerAdditionalV2('dataset.xlsx')

# パラメータ調整（オプション）
scheduler.population_size = 50
scheduler.num_generations = 500

# 実行
best_solution, best_fitness, best_history, avg_history = scheduler.run()

# 結果の検証
scheduler.print_validation_report(best_solution)
```

### 12. トラブルシューティング

#### 収束が遅い場合
1. 集団サイズを増やす（50→100）
2. 世代数を増やす（500→1000）
3. 突然変異率を調整（0.3→0.2-0.4）

#### 制約違反が多い場合
1. ペナルティ重みを増やす（コード修正必要）
2. 初期解生成を改善（制約考慮を強化）
3. V3への移行を検討

#### メモリ不足の場合
1. 集団サイズを減らす
2. 世代数を分割実行
3. 不要な履歴記録を削減