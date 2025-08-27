# GA Job Shop Scheduler V3 アルゴリズム解説

## 概要
`ga_jobshop_scheduler_additional_v3.py`は、追加制約（固定タスク・連続作業制限）に対応したジョブショップスケジューリング問題を解く遺伝的アルゴリズム（GA）の改善版実装です。V3では収束性能を大幅に改善するための最適化手法を導入しています。

## 主な特徴と改善点

### 1. 問題設定
- **基本制約**: 順序制約、リソース制約（作業者・設備）
- **追加制約**:
  - **固定タスク**: 特定の時刻・リソース（rsrc_0005, rsrc_0046）で実行が固定されたタスク
  - **連続作業制限**: 特定の設備で連続して実行してはいけないパターンの組み合わせ

### 2. V3の主要な改善技術

#### 2.1 ヒューリスティック初期化（80%の個体）
```python
def heuristic_task_order(self):
    # 優先度計算: 所要時間、後続タスク数、制約数を考慮
    priority = duration * 0.5 + num_successors * 10 - num_predecessors * 5
```
- **目的**: ランダムではなく、問題の構造を考慮した賢い初期解生成
- **効果**: 初期世代から良質な解を含む

#### 2.2 適応的ペナルティ重み
```python
if self.adaptive_penalty:
    # 世代が進むにつれてペナルティを減らし、makespanの改善を重視
    generation_factor = max(0.5, 1.0 - self.current_generation / self.num_generations)
    penalty *= self.penalty_weight * generation_factor
```
- **初期**: 制約満足を重視（ペナルティ大）
- **後期**: makespan最小化を重視（ペナルティ小）
- **効果**: 段階的に最適化の焦点を移行

#### 2.3 局所探索の統合（20%の確率）
```python
def local_search(self, chromosome):
    # 近傍解を5つ試して最良を選択
    for _ in range(5):
        neighbor = self.mutate(copy.deepcopy(best_chromosome))
        if neighbor_fitness < best_fitness:
            best_chromosome = neighbor
```
- **目的**: GAの大域探索と局所探索のハイブリッド化
- **効果**: 解の精度向上

#### 2.4 複数の突然変異オペレータ
1. **タスク入れ替え** (30%): 2つのタスクの順序を交換
2. **部分列逆転** (30%): 最大5タスクの順序を逆転
3. **リソース再割当** (30%): タスクのリソース割当を変更
4. **挿入突然変異** (10%): タスクを別の位置に移動

#### 2.5 停滞検出と多様性注入
```python
if self.best_fitness_no_improve > self.stagnation_limit:
    # 集団の25%を新しい個体で置き換え
    num_replace = self.population_size // 4
    for i in range(num_replace):
        population[-(i+1)] = self.create_individual()
```
- **検出**: 50世代改善なし
- **対処**: 新個体注入と突然変異率の一時的増加
- **効果**: 局所最適からの脱出

### 3. 制約処理の最適化

#### 3.1 固定タスクの処理
- スケジュール初期化時に固定タスクを配置
- 他のタスクは固定タスクを避けて配置
- `find_available_slot`メソッドで空き時間を効率的に探索

#### 3.2 連続作業制限の処理
```python
def check_continuous_constraint(self, equipment, schedule, new_task_pattern, start_time):
    # 設備上で連続して実行されるタスクのパターンをチェック
    # 時間差は考慮せず、実行順序のみで判定
```
- 違反時は段階的に遅延（30, 60, 120, 240, 480分）
- ペナルティは100→20に削減（V3）

### 4. パラメータ設定（V3最適化済み）

| パラメータ | V2 | V3 | 理由 |
|---------|-----|-----|------|
| 集団サイズ | 50 | 100 | 多様性確保 |
| 交叉率 | 0.8 | 0.85 | 探索効率向上 |
| 突然変異率 | 0.3 | 0.3（動的） | 適応的調整 |
| エリート率 | 0.1 | 0.15 | 優良解の保存強化 |
| トーナメントサイズ | 3 | 5 | 選択圧の調整 |

### 5. 収束改善のメカニズム

```
初期世代 (0-100)
  ├─ ヒューリスティック初期化 (80%)
  ├─ 高ペナルティ重み
  └─ 制約満足を優先

中期世代 (100-300)
  ├─ 適応的ペナルティ調整
  ├─ 局所探索の活用
  └─ バランスの取れた最適化

後期世代 (300-500)
  ├─ 低ペナルティ重み
  ├─ makespan最小化重視
  └─ 停滞時の多様性注入
```

### 6. 実装の特徴

#### 6.1 事前計算による高速化
- 制約グラフの事前構築
- リソース競合スコアの計算

#### 6.2 メモリ効率
- 必要最小限のコピー
- 効率的なデータ構造

#### 6.3 デバッグ機能
- 詳細な検証レポート
- 収束曲線の可視化
- ガントチャート出力

### 7. 期待される効果

1. **収束速度**: V2比で約2-3倍高速
2. **解の質**: 10-20%のmakespan改善
3. **安定性**: 停滞の自動回復
4. **スケーラビリティ**: 大規模問題への対応力向上

### 8. 使用方法

```python
from ga_jobshop_scheduler_additional_v3 import JobShopGASchedulerAdditionalV3

# スケジューラの初期化
scheduler = JobShopGASchedulerAdditionalV3('dataset.xlsx')

# 実行
best_solution, best_fitness, best_history, avg_history = scheduler.run()

# 結果の検証
scheduler.print_validation_report(best_solution)
```

### 9. チューニングガイド

問題の特性に応じて以下のパラメータを調整：

- **制約が厳しい場合**: 
  - `penalty_weight`を増加（20-50）
  - `stagnation_limit`を減少（30-40）

- **大規模問題の場合**:
  - `population_size`を増加（150-200）
  - `local_search_rate`を減少（0.1）

- **収束が遅い場合**:
  - `tournament_size`を増加（7-10）
  - `elite_rate`を増加（0.2）

### 10. 今後の改善可能性

1. **並列化**: 適応度計算の並列実行
2. **機械学習統合**: 過去の解から学習
3. **動的パラメータ調整**: 問題特性の自動認識
4. **ハイブリッドアルゴリズム**: シミュレーテッドアニーリングとの統合