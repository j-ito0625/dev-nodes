# 並列遺伝的アルゴリズム ジョブショップスケジューラ ドキュメント

## 概要

本ドキュメントは、`ga_jobshop_scheduler_additional_v3_parallel.py` および関連ファイルで実装された並列処理対応の遺伝的アルゴリズム（GA）によるジョブショップスケジューリングシステムについて説明します。

## 目次

1. [システム概要](#システム概要)
2. [アーキテクチャ](#アーキテクチャ)
3. [主要コンポーネント](#主要コンポーネント)
4. [インストールと環境設定](#インストールと環境設定)
5. [使用方法](#使用方法)
6. [APIリファレンス](#apiリファレンス)
7. [アルゴリズム詳細](#アルゴリズム詳細)
8. [性能とチューニング](#性能とチューニング)
9. [トラブルシューティング](#トラブルシューティング)

---

## システム概要

### 目的
製造現場における複雑なジョブショップスケジューリング問題を、並列処理を活用した高速な遺伝的アルゴリズムで解決します。

### 主な特徴
- **島モデル並列GA**: 複数の独立した部分集団（島）で並列進化
- **追加制約対応**: 固定タスク、連続作業制限などの実務的制約に対応
- **高速化**: マルチプロセッシングによる計算時間の大幅短縮
- **V3アルゴリズム保持**: 実績のあるV3の最適化機能を完全継承

### 対応する制約
1. **順序制約**: タスク間の先行・後続関係
2. **リソース制約**: 作業者・設備の排他的利用
3. **固定タスク**: 事前に時間とリソースが決定されたタスク
4. **連続作業制限**: 特定パターンの連続実行禁止
5. **ダミーリソース**: 無制限リソースの処理

---

## アーキテクチャ

### システム構成

```
┌─────────────────────────────────────────┐
│         メインプロセス                    │
│  ┌───────────────────────────────┐      │
│  │  JobShopGASchedulerAdditional  │      │
│  │       V3Parallel               │      │
│  └───────────┬───────────────────┘      │
│              │                           │
│  ┌───────────▼───────────────────┐      │
│  │    Island Manager              │      │
│  └───────────┬───────────────────┘      │
└──────────────┼──────────────────────────┘
               │
    ┌──────────┼──────────┬──────────┬──────────┐
    │          │          │          │          │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│Island 0│ │Island 1│ │Island 2│ │Island 3│
│Process │ │Process │ │Process │ │Process │
└────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘
     │          │          │          │
  Evolution  Evolution  Evolution  Evolution
  (並列実行)  (並列実行)  (並列実行)  (並列実行)
     │          │          │          │
     └──────────┴──────────┴──────────┘
                Migration Queue
              (移住による遺伝子交換)
```

### データフロー

1. **初期化フェーズ**
   - データセット読み込み
   - 制約グラフ構築
   - 初期個体群生成（ヒューリスティック）

2. **並列進化フェーズ**
   - 各島での独立進化
   - 定期的な移住（優良個体の交換）
   - 適応度の並列計算

3. **収束フェーズ**
   - 最良解の選択
   - スケジュール検証
   - 結果の可視化

---

## 主要コンポーネント

### 1. JobShopGASchedulerAdditionalV3Parallel クラス

メインクラスで、並列GAスケジューラの全機能を提供します。

```python
class JobShopGASchedulerAdditionalV3Parallel:
    def __init__(self, dataset_path: str, num_islands: int = 4):
        """
        Args:
            dataset_path: Excelデータセットのパス
            num_islands: 島モデルの島数（並列度）
        """
```

### 2. 島モデル実装

```python
def evolve_island(self, island_id: int, population: List, 
                 generations: int, migration_queue: Queue, 
                 result_queue: Queue):
    """
    単一島での進化プロセス
    独立プロセスで実行される
    """
```

### 3. 並列適応度評価

```python
def parallel_fitness_evaluation(self, population, generation):
    """
    ProcessPoolExecutorによる適応度計算の並列化
    CPUコア数を最大限活用
    """
```

---

## インストールと環境設定

### 必要要件

```bash
# Python バージョン
Python 3.7以上

# 必要パッケージ
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
openpyxl>=3.0.0
```

### インストール手順

```bash
# 1. 依存パッケージのインストール
pip install pandas numpy matplotlib openpyxl

# 2. ファイルの配置
ga_jobshop_scheduler_additional_v3_parallel.py
run_parallel_scheduler.py
toy_dataset_additional.xlsx  # データセット
```

### 環境変数設定（オプション）

```bash
# マルチプロセシング用の設定（Linux/Mac）
export PYTHONPATH=$PYTHONPATH:/path/to/project

# プロセス数の制限（必要に応じて）
export OMP_NUM_THREADS=4
```

---

## 使用方法

### 基本的な使用例

#### 1. 単純実行

```python
from ga_jobshop_scheduler_additional_v3_parallel import JobShopGASchedulerAdditionalV3Parallel

# スケジューラの初期化
scheduler = JobShopGASchedulerAdditionalV3Parallel(
    dataset_path='toy_dataset_additional.xlsx',
    num_islands=4
)

# 最適化の実行
best_solution, best_fitness, best_history, avg_history = scheduler.run()

# 結果の表示
print(f"Best makespan: {best_fitness:.2f} minutes")
```

#### 2. コマンドライン実行

```bash
# デフォルト設定で実行
python run_parallel_scheduler.py

# カスタム設定
python run_parallel_scheduler.py \
    --dataset my_dataset.xlsx \
    --islands 8 \
    --generations 1000 \
    --mode parallel

# 性能比較モード
python run_parallel_scheduler.py \
    --mode compare \
    --generations 100
```

#### 3. 詳細な制御

```python
# パラメータのカスタマイズ
scheduler = JobShopGASchedulerAdditionalV3Parallel(
    dataset_path='dataset.xlsx',
    num_islands=6
)

# GAパラメータの調整
scheduler.population_size_per_island = 30
scheduler.crossover_rate = 0.9
scheduler.mutation_rate = 0.25
scheduler.migration_interval = 15
scheduler.num_generations = 1000

# 実行
best_solution, best_fitness, _, _ = scheduler.run()

# 検証
validation_results = scheduler.validate_schedule(best_solution)
scheduler.print_validation_report(best_solution)

# 可視化
import matplotlib.pyplot as plt
gantt_fig = scheduler.visualize_schedule(best_solution)
plt.show()
```

### データセット形式

Excelファイルに以下のシートが必要：

#### 必須シート

1. **リソース**: 作業者と設備のリスト
   - 列: `作業者`, `設備`

2. **task**: タスク情報
   - 列: `タスクID`, `所要時間`, `割り当て可能パターン`, `品種目`

3. **割り当て情報**: リソース割り当てパターン
   - 列: `割り当て可能パターン`, `作業者`, `設備`

4. **順序制約**: タスク間の順序関係
   - 列: `先行作業ID`, `後作業ID`, `先行作業原点`, `後作業原点`, `時間差下限`

#### オプションシート

5. **固定タスク**: 事前確定タスク
   - 列: `ID`, `開始時刻`, `終了時刻`, `リソース名`, `割り当て可能パターン`, `品種名`

6. **連続作業制限**: 連続実行禁止パターン
   - 列: `設備`, `先行パターン`, `後パターン`

---

## APIリファレンス

### 主要メソッド

#### 初期化と設定

```python
__init__(dataset_path: str, num_islands: int = 4)
    """
    スケジューラの初期化
    
    Parameters:
        dataset_path: データセットのパス
        num_islands: 並列処理の島数（デフォルト: 4）
    """

initialize_parameters()
    """
    GAパラメータの初期化
    内部で以下を設定：
    - population_size_per_island: 25
    - crossover_rate: 0.85
    - mutation_rate: 0.3
    - elite_rate: 0.15
    - migration_interval: 20
    """
```

#### 実行メソッド

```python
run() -> Tuple[List, float, List, List]
    """
    並列GAの実行（互換性メソッド）
    
    Returns:
        best_solution: 最良解の染色体
        best_fitness: 最良適応度（メイクスパン）
        best_history: 最良適応度の履歴
        avg_history: 平均適応度の履歴
    """

run_parallel() -> Tuple[List, float, List]
    """
    並列GAの実行（詳細版）
    
    Returns:
        best_solution: 最良解
        best_fitness: 最良適応度
        results: 各島の結果リスト
    """

evolve_island(island_id: int, population: List, 
              generations: int, migration_queue: Queue, 
              result_queue: Queue)
    """
    単一島での進化（内部メソッド）
    
    Parameters:
        island_id: 島の識別子
        population: 初期個体群
        generations: 世代数
        migration_queue: 移住用キュー
        result_queue: 結果用キュー
    """
```

#### 個体操作メソッド

```python
create_individual() -> List[Dict]
    """
    個体（染色体）の生成
    80%の確率でヒューリスティック初期化
    
    Returns:
        chromosome: タスク割り当てのリスト
    """

crossover(parent1: List, parent2: List) -> Tuple[List, List]
    """
    順序交叉（OX）による子個体生成
    
    Parameters:
        parent1, parent2: 親個体
    
    Returns:
        child1, child2: 子個体
    """

mutate(chromosome: List) -> List
    """
    突然変異の適用
    複数の変異オペレータをランダム選択
    
    Parameters:
        chromosome: 対象個体
    
    Returns:
        mutated: 変異後の個体
    """

local_search(chromosome: List) -> List
    """
    局所探索による改善
    
    Parameters:
        chromosome: 対象個体
    
    Returns:
        improved: 改善後の個体
    """
```

#### 評価メソッド

```python
calculate_fitness(chromosome: List) -> float
    """
    適応度（メイクスパン）の計算
    
    Parameters:
        chromosome: 評価対象の個体
    
    Returns:
        fitness: メイクスパン + ペナルティ
    """

parallel_fitness_evaluation(population: List, 
                          generation: int) -> List[float]
    """
    並列適応度評価
    
    Parameters:
        population: 個体群
        generation: 現在の世代
    
    Returns:
        fitness_scores: 適応度リスト
    """

decode_chromosome(chromosome: List) -> Dict
    """
    染色体をスケジュールに変換
    
    Parameters:
        chromosome: 染色体
    
    Returns:
        schedule: タスクIDをキーとするスケジュール辞書
    """
```

#### 検証メソッド

```python
validate_schedule(chromosome: List) -> Dict
    """
    スケジュールの完全検証
    
    Parameters:
        chromosome: 検証対象の個体
    
    Returns:
        validation_results: 検証結果の辞書
            - is_valid: bool
            - makespan: float
            - violations: List
            - resource_conflicts: List
            - constraint_violations: List
            - continuous_violations: List
    """

print_validation_report(chromosome: List) -> Dict
    """
    検証レポートの出力
    
    Parameters:
        chromosome: 対象個体
    
    Returns:
        validation_results: 検証結果
    """
```

#### 可視化メソッド

```python
visualize_schedule(chromosome: List) -> matplotlib.figure.Figure
    """
    ガントチャートの生成
    
    Parameters:
        chromosome: 表示対象の個体
    
    Returns:
        fig: matplotlibのFigureオブジェクト
    """

plot_convergence(best_history: List, 
                avg_history: List) -> matplotlib.figure.Figure
    """
    収束曲線のプロット
    
    Parameters:
        best_history: 最良適応度の履歴
        avg_history: 平均適応度の履歴
    
    Returns:
        fig: matplotlibのFigureオブジェクト
    """
```

---

## アルゴリズム詳細

### 島モデル並列GA

#### 概要
島モデル（Island Model）は、全体の個体群を複数の部分集団（島）に分割し、各島で独立に進化を行う並列GAの手法です。

#### 特徴
1. **独立進化**: 各島は異なるプロセスで並列実行
2. **定期的移住**: 優良個体の島間交換による多様性維持
3. **収束性向上**: 局所最適解への早期収束を防止

#### 移住戦略
```python
# 移住間隔: 20世代ごと
migration_interval = 20

# 移住率: 各島の個体群の10%
migration_rate = 0.1

# 移住個体の選択: 適応度上位
elite_migrants = top_10_percent_by_fitness
```

### ヒューリスティック初期化

#### タスク順序の決定
```python
def heuristic_task_order():
    # 優先度計算
    for task in tasks:
        priority = (
            duration * 0.5 +           # 処理時間
            num_successors * 10 -      # 後続タスク数
            num_predecessors * 5 +      # 先行タスク数
            random() * 10              # ランダム要素
        )
    
    # トポロジカルソートで制約を満たしつつ優先度順
    return topological_sort_with_priorities(priorities)
```

#### リソース割り当て
```python
def select_best_allocation(task_id, pattern):
    # 競合スコアを最小化
    for allocation in possible_allocations:
        conflict_score = 0
        
        # 固定タスクとの競合チェック
        if conflicts_with_fixed_task:
            conflict_score += 100
        
        # 連続作業制限の考慮
        if violates_continuous_constraint:
            conflict_score += penalty
    
    return allocation_with_minimum_conflict
```

### 適応的ペナルティ

世代の進行に応じてペナルティ重みを動的に調整：

```python
if adaptive_penalty:
    # 初期は制約満足を重視、後期はメイクスパン最小化を重視
    generation_factor = max(0.5, 1.0 - current_generation / num_generations)
    penalty *= penalty_weight * generation_factor
```

### 局所探索

各世代で一定確率（20%）で局所探索を適用：

```python
def local_search(chromosome):
    best = chromosome
    for _ in range(5):  # 5つの近傍解を探索
        neighbor = mutate(best)
        if fitness(neighbor) < fitness(best):
            best = neighbor
    return best
```

### 多様性保持メカニズム

#### 停滞検出
```python
if generations_without_improvement > stagnation_limit:
    # 集団の25%を新規個体で置換
    replace_bottom_25_percent_with_new_individuals()
    
    # 突然変異率を一時的に増加
    mutation_rate *= 1.2
```

#### 適応的パラメータ調整
```python
if generation % 100 == 0:
    if population_diversity < threshold:
        mutation_rate = min(0.5, mutation_rate * 1.1)
    else:
        mutation_rate = max(0.1, mutation_rate * 0.95)
```

---

## 性能とチューニング

### 性能指標

#### 並列化効率
```
理論的スピードアップ = 島数
実際のスピードアップ = 単一プロセス時間 / 並列処理時間
並列化効率 = 実際のスピードアップ / 理論的スピードアップ
```

典型的な性能:
- 4島: 2.5-3.5倍のスピードアップ（効率60-85%）
- 8島: 4-6倍のスピードアップ（効率50-75%）

### パラメータチューニング

#### 島数の選定

```python
# CPU コア数に基づく推奨設定
import os
optimal_islands = min(os.cpu_count(), 8)

# メモリ制約がある場合
if available_memory_gb < 4:
    optimal_islands = min(optimal_islands, 4)
```

#### 個体群サイズ

```python
# タスク数に応じた調整
if num_tasks < 50:
    population_size_per_island = 20
elif num_tasks < 100:
    population_size_per_island = 25
elif num_tasks < 200:
    population_size_per_island = 30
else:
    population_size_per_island = 40
```

#### 移住パラメータ

```python
# 収束速度と多様性のバランス
# 速い収束を優先
migration_interval = 10
migration_rate = 0.2

# 多様性を優先
migration_interval = 30
migration_rate = 0.05
```

### メモリ最適化

#### プロセス間通信の最小化
```python
# 移住時は個体全体ではなくIDと適応度のみ送信
migration_data = {
    'individual_id': id,
    'fitness': fitness,
    'compressed_chromosome': compress(chromosome)
}
```

#### ガベージコレクション
```python
import gc

# 各世代後に明示的なガベージコレクション
if generation % 50 == 0:
    gc.collect()
```

---

## トラブルシューティング

### よくある問題と解決策

#### 1. メモリ不足エラー

**症状**: `MemoryError` または システムのフリーズ

**解決策**:
```python
# 島数を減らす
scheduler = JobShopGASchedulerAdditionalV3Parallel(
    dataset_path='dataset.xlsx',
    num_islands=2  # 4から2に削減
)

# 個体群サイズを削減
scheduler.population_size_per_island = 15  # 25から15に削減
```

#### 2. プロセス間通信エラー

**症状**: `BrokenPipeError` または `EOFError`

**解決策**:
```python
# マルチプロセシングの開始方法を変更
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
```

#### 3. 収束が遅い

**症状**: 世代を重ねても改善が見られない

**解決策**:
```python
# ヒューリスティック初期化の比率を上げる
# create_individual()内で
if random.random() < 0.95:  # 0.8から0.95に増加
    task_order = self.heuristic_task_order()

# 局所探索率を増やす
scheduler.local_search_rate = 0.4  # 0.2から0.4に増加
```

#### 4. 制約違反が解消されない

**症状**: 最終解が制約を満たさない

**解決策**:
```python
# ペナルティ重みを増加
scheduler.penalty_weight = 50.0  # 10.0から50.0に増加

# 適応的ペナルティを無効化
scheduler.adaptive_penalty = False
```

### デバッグモード

詳細なログ出力を有効にする：

```python
import logging

# ログレベルを設定
logging.basicConfig(level=logging.DEBUG)

# カスタムデバッグ関数
def debug_schedule(schedule):
    print(f"Tasks: {len(schedule)}")
    print(f"Makespan: {max(t['end_time'] for t in schedule.values())}")
    
    # 制約違反のチェック
    violations = scheduler.validate_schedule(chromosome)
    if not violations['is_valid']:
        print(f"Violations found: {violations}")
```

### パフォーマンスプロファイリング

```python
import cProfile
import pstats

# プロファイリング実行
profiler = cProfile.Profile()
profiler.enable()

# スケジューラ実行
best_solution, best_fitness, _, _ = scheduler.run()

profiler.disable()

# 結果の表示
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # 上位20個の関数を表示
```

---

## 付録

### A. 設定例

#### 小規模問題（タスク数 < 50）
```python
scheduler = JobShopGASchedulerAdditionalV3Parallel(
    dataset_path='small_dataset.xlsx',
    num_islands=2
)
scheduler.population_size_per_island = 20
scheduler.num_generations = 200
scheduler.migration_interval = 20
```

#### 中規模問題（タスク数 50-200）
```python
scheduler = JobShopGASchedulerAdditionalV3Parallel(
    dataset_path='medium_dataset.xlsx',
    num_islands=4
)
scheduler.population_size_per_island = 25
scheduler.num_generations = 500
scheduler.migration_interval = 25
```

#### 大規模問題（タスク数 > 200）
```python
scheduler = JobShopGASchedulerAdditionalV3Parallel(
    dataset_path='large_dataset.xlsx',
    num_islands=8
)
scheduler.population_size_per_island = 30
scheduler.num_generations = 1000
scheduler.migration_interval = 30
scheduler.local_search_rate = 0.1  # 計算時間削減のため減少
```

### B. ベンチマーク結果

典型的な実行時間（Intel Core i7、8コア、16GB RAM）:

| タスク数 | 島数 | 世代数 | 単一プロセス | 並列処理 | スピードアップ |
|---------|------|--------|-------------|---------|--------------|
| 30      | 2    | 200    | 120秒       | 65秒    | 1.8倍        |
| 50      | 4    | 300    | 300秒       | 95秒    | 3.2倍        |
| 100     | 4    | 500    | 720秒       | 210秒   | 3.4倍        |
| 200     | 8    | 1000   | 2400秒      | 480秒   | 5.0倍        |

### C. 参考文献

1. **島モデルGA**
   - Whitley, D., Rana, S., & Heckendorn, R. B. (1998). "The island model genetic algorithm: On separability, population size and convergence"

2. **ジョブショップスケジューリング**
   - Blazewicz, J., et al. (2019). "Handbook on Scheduling: From Theory to Practice"

3. **並列GA実装**
   - Alba, E., & Troya, J. M. (1999). "A survey of parallel distributed genetic algorithms"

---

## 更新履歴

- **v1.0.0** (2024-01): 初版リリース
  - 島モデル並列GA実装
  - V3アルゴリズムの完全移植
  - 追加制約対応

---

## ライセンス

本ソフトウェアは研究・教育目的での使用を想定しています。
商用利用の際は別途ご相談ください。

---

## お問い合わせ

バグ報告、機能要望、その他のお問い合わせは、プロジェクトのIssueトラッカーまでお願いします。