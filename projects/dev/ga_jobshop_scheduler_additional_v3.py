import pandas as pd
import numpy as np
import random
import copy
import time
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import ast

class JobShopGASchedulerAdditionalV3:
    def __init__(self, dataset_path: str):
        """
        追加制約対応の遺伝的アルゴリズムを用いたジョブショップスケジューラ（改善版V3）
        収束性能を改善するための最適化を実装
        
        Args:
            dataset_path: Excelデータセットのパス
        """
        self.load_data(dataset_path)
        self.initialize_parameters()
        
    def load_data(self, dataset_path: str):
        """データセットの読み込み（追加制約対応）"""
        # 基本データの読み込み
        self.resources_df = pd.read_excel(dataset_path, sheet_name='リソース')
        self.tasks_df = pd.read_excel(dataset_path, sheet_name='task')
        self.allocation_df = pd.read_excel(dataset_path, sheet_name='割り当て情報')
        self.order_constraints_df = pd.read_excel(dataset_path, sheet_name='順序制約')
        
        # 追加制約の読み込み
        try:
            self.fixed_tasks_df = pd.read_excel(dataset_path, sheet_name='固定タスク')
            self.continuous_constraints_df = pd.read_excel(dataset_path, sheet_name='連続作業制限')
        except:
            print("Warning: 追加制約シートが見つかりません。基本制約のみで動作します。")
            self.fixed_tasks_df = pd.DataFrame()
            self.continuous_constraints_df = pd.DataFrame()
        
        # データの前処理
        self.process_data()
        
    def process_data(self):
        """データの前処理と制約の構築（追加制約対応）"""
        # 基本的なタスク情報の辞書化
        self.tasks = {}
        for _, row in self.tasks_df.iterrows():
            task_id = row['タスクID']
            self.tasks[task_id] = {
                'duration': row['所要時間'],
                'pattern': row['割り当て可能パターン'],
                'product': row['品種目']
            }
        
        # 固定タスクの処理
        self.fixed_tasks = {}
        if not self.fixed_tasks_df.empty:
            for _, row in self.fixed_tasks_df.iterrows():
                task_id = row['ID']
                # リソース名をパース
                resource_str = row['リソース名']
                # 文字列形式のタプルをパース
                if resource_str.startswith("("):
                    parts = resource_str.strip("()").replace("'", "").split(",")
                    worker = parts[0].strip()
                    equipment = parts[1].strip()
                else:
                    # 別の形式の場合の処理
                    worker, equipment = eval(resource_str)
                
                self.fixed_tasks[task_id] = {
                    'start_time': row['開始時刻'],
                    'end_time': row['終了時刻'],
                    'worker': worker,
                    'equipment': equipment,
                    'pattern': row['割り当て可能パターン'],
                    'product': row['品種名']
                }
        
        # 連続作業制限の処理
        self.continuous_constraints = {}
        if not self.continuous_constraints_df.empty:
            for _, row in self.continuous_constraints_df.iterrows():
                equipment = row['設備']
                if equipment not in self.continuous_constraints:
                    self.continuous_constraints[equipment] = []
                self.continuous_constraints[equipment].append({
                    'prev_pattern': row['先行パターン'],
                    'next_pattern': row['後パターン']
                })
        
        # 割り当て可能な作業者・設備の組み合わせを辞書化
        self.allocation_patterns = {}
        for _, row in self.allocation_df.iterrows():
            pattern = row['割り当て可能パターン']
            if pattern not in self.allocation_patterns:
                self.allocation_patterns[pattern] = []
            self.allocation_patterns[pattern].append({
                'worker': row['作業者'],
                'equipment': row['設備']
            })
        
        # 順序制約の辞書化（固定タスクも含む）
        self.order_constraints = []
        for _, row in self.order_constraints_df.iterrows():
            self.order_constraints.append({
                'predecessor': row['先行作業ID'],
                'successor': row['後作業ID'],
                'pred_origin': row['先行作業原点'],
                'succ_origin': row['後作業原点'],
                'time_diff_min': row['時間差下限']
            })
        
        # リソースのリスト化
        self.workers = [w for w in self.resources_df['作業者'].dropna()]
        self.equipments = [e for e in self.resources_df['設備'].dropna()]
        
        # 通常タスク（固定タスク以外）のIDリスト
        self.regular_task_ids = [tid for tid in self.tasks.keys() if tid not in self.fixed_tasks]
        self.num_regular_tasks = len(self.regular_task_ids)
        
        print(f"通常タスク数: {self.num_regular_tasks}")
        print(f"固定タスク数: {len(self.fixed_tasks)}")
        
        # 最適化: 制約グラフの事前計算
        self.build_constraint_graph()
        
    def build_constraint_graph(self):
        """制約グラフの構築（最適化用）"""
        self.predecessors = {task: [] for task in self.regular_task_ids}
        self.successors = {task: [] for task in self.regular_task_ids}
        
        for constraint in self.order_constraints:
            pred = constraint['predecessor']
            succ = constraint['successor']
            if pred in self.regular_task_ids and succ in self.regular_task_ids:
                self.predecessors[succ].append(pred)
                self.successors[pred].append(succ)
                
    def initialize_parameters(self):
        """GAパラメータの初期化（改善版）"""
        # 基本パラメータ
        self.population_size = 100  # 増加させて多様性を確保
        self.crossover_rate = 0.85
        self.mutation_rate = 0.3
        self.elite_rate = 0.15
        self.num_generations = 500
        
        # 改善版の追加パラメータ
        self.tournament_size = 5  # トーナメントサイズを増加
        self.local_search_rate = 0.2  # 局所探索の確率
        self.adaptive_penalty = True  # 適応的ペナルティ
        self.penalty_weight = 10.0  # 初期ペナルティ重み（適応的に変更）
        self.diversity_preservation = True  # 多様性保持メカニズム
        
        # 収束監視用
        self.stagnation_limit = 50  # 停滞世代数の限界
        self.best_fitness_no_improve = 0
        
    def create_individual(self):
        """個体（染色体）の生成（改善版：ヒューリスティック初期化）"""
        chromosome = []
        
        # 80%は制約を考慮したヒューリスティック、20%はランダム
        if random.random() < 0.8:
            task_order = self.heuristic_task_order()
        else:
            task_order = self.topological_sort_with_randomness()
        
        for task_id in task_order:
            if task_id in self.regular_task_ids:
                pattern = self.tasks[task_id]['pattern']
                if pattern in self.allocation_patterns:
                    # リソース選択も最適化
                    allocation = self.select_best_allocation(task_id, pattern)
                    chromosome.append({
                        'task': task_id,
                        'worker': allocation['worker'],
                        'equipment': allocation['equipment']
                    })
                else:
                    chromosome.append({
                        'task': task_id,
                        'worker': random.choice(self.workers[:-1]),
                        'equipment': random.choice(self.equipments[:-1])
                    })
        
        return chromosome
    
    def heuristic_task_order(self):
        """ヒューリスティックによるタスク順序生成"""
        # 優先度ベースのタスク順序生成
        task_priorities = {}
        
        for task in self.regular_task_ids:
            # 優先度計算: 所要時間、後続タスク数、制約数を考慮
            duration = self.tasks[task]['duration']
            num_successors = len(self.successors.get(task, []))
            num_predecessors = len(self.predecessors.get(task, []))
            
            # 長いタスク、多くの後続を持つタスクを優先
            priority = duration * 0.5 + num_successors * 10 - num_predecessors * 5
            task_priorities[task] = priority + random.random() * 10  # ランダム性を追加
        
        # 制約を満たしながら優先度順にソート
        return self.topological_sort_with_priorities(task_priorities)
    
    def topological_sort_with_priorities(self, priorities):
        """優先度を考慮したトポロジカルソート"""
        in_degree = {task: len(self.predecessors[task]) for task in self.regular_task_ids}
        result = []
        available = [task for task in self.regular_task_ids if in_degree[task] == 0]
        
        while available:
            # 優先度でソート
            available.sort(key=lambda t: priorities.get(t, 0), reverse=True)
            task = available.pop(0)
            result.append(task)
            
            for successor in self.successors.get(task, []):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    available.append(successor)
        
        # 制約に関係ないタスクを追加
        for task in self.regular_task_ids:
            if task not in result:
                result.append(task)
        
        return result
    
    def select_best_allocation(self, task_id, pattern):
        """最適なリソース割り当ての選択"""
        allocations = self.allocation_patterns[pattern]
        
        # 固定タスクと競合しないリソースを優先
        best_allocation = None
        min_conflict = float('inf')
        
        for allocation in allocations:
            conflict_score = 0
            
            # 固定タスクとの競合をチェック
            for fixed_task in self.fixed_tasks.values():
                if (allocation['worker'] == fixed_task['worker'] or 
                    allocation['equipment'] == fixed_task['equipment']):
                    conflict_score += 100
            
            # 連続作業制限を考慮
            if allocation['equipment'] in self.continuous_constraints:
                conflict_score += len(self.continuous_constraints[allocation['equipment']])
            
            if conflict_score < min_conflict:
                min_conflict = conflict_score
                best_allocation = allocation
        
        return best_allocation if best_allocation else random.choice(allocations)
    
    def topological_sort_with_randomness(self):
        """順序制約を満たしつつランダム性を持たせたトポロジカルソート（固定タスク考慮）"""
        # 通常タスクのみを対象とする
        in_degree = {task: 0 for task in self.regular_task_ids}
        graph = {task: [] for task in self.regular_task_ids}
        
        for constraint in self.order_constraints:
            pred = constraint['predecessor']
            succ = constraint['successor']
            
            # 両方が通常タスクの場合のみグラフに追加
            if pred in graph and succ in in_degree:
                graph[pred].append(succ)
                in_degree[succ] += 1
            # 固定タスクが先行の場合は入次数のみ増やす
            elif pred in self.fixed_tasks and succ in in_degree:
                in_degree[succ] += 1
        
        # トポロジカルソート（ランダム性あり）
        result = []
        available = [task for task in self.regular_task_ids if in_degree[task] == 0]
        
        while available:
            random.shuffle(available)
            task = available.pop(0)
            result.append(task)
            
            if task in graph:
                for successor in graph[task]:
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        available.append(successor)
        
        # 制約に関係ないタスクを追加
        for task in self.regular_task_ids:
            if task not in result:
                result.append(task)
        
        return result
    
    def find_available_slot(self, worker_busy_periods, equipment_busy_periods, duration, worker, equipment):
        """利用可能な時間スロットを見つける"""
        # 開始時刻は0から（仕様変更：絶対時刻0をスタート）
        earliest_start = 0
        
        # 作業者と設備の忙しい期間を結合
        all_busy_periods = []
        
        if worker != 'dummy-type-0001' and worker in worker_busy_periods:
            all_busy_periods.extend(worker_busy_periods[worker])
        if equipment != 'dummy-type-0002' and equipment in equipment_busy_periods:
            all_busy_periods.extend(equipment_busy_periods[equipment])
        
        # 期間をソート
        all_busy_periods.sort(key=lambda x: x[0])
        
        # 空き時間を探す
        for start, end in all_busy_periods:
            if earliest_start + duration <= start:
                # このスロットに収まる
                return earliest_start
            earliest_start = max(earliest_start, end)
        
        return earliest_start
    
    def check_continuous_constraint(self, equipment, schedule, new_task_pattern, start_time):
        """
        連続作業制限のチェック（仕様変更版）
        先行パターンと後パターンの間に別タスクが存在する場合のみ制約を満たす
        時間差は考慮しない（設備上で連続して実行される場合は禁止）
        """
        if equipment not in self.continuous_constraints:
            return True  # この設備に制限がない
        
        # この設備で実行されるタスクを時系列順にソート
        equipment_tasks = []
        for task_id, info in schedule.items():
            if info['equipment'] == equipment:
                # パターンを取得
                if task_id in self.fixed_tasks:
                    pattern = self.fixed_tasks[task_id]['pattern']
                else:
                    pattern = self.tasks[task_id]['pattern']
                equipment_tasks.append((info['start_time'], info['end_time'], pattern, task_id))
        
        # 新しいタスクを仮想的に追加
        equipment_tasks.append((start_time, start_time + 1, new_task_pattern, 'NEW_TASK'))
        equipment_tasks.sort()
        
        # 新しいタスクの位置を見つける
        new_task_idx = -1
        for i, task in enumerate(equipment_tasks):
            if task[3] == 'NEW_TASK':
                new_task_idx = i
                break
        
        # 直前のタスクがある場合（設備上で連続して実行される）
        if new_task_idx > 0:
            prev_task = equipment_tasks[new_task_idx - 1]
            prev_pattern = prev_task[2]
            
            # 連続作業制限をチェック（時間差は考慮しない、順序のみ）
            for constraint in self.continuous_constraints[equipment]:
                if constraint['prev_pattern'] == prev_pattern and constraint['next_pattern'] == new_task_pattern:
                    # この設備で連続して実行されるため制限違反
                    return False
        
        # 直後のタスクがある場合（設備上で連続して実行される）
        if new_task_idx < len(equipment_tasks) - 1:
            next_task = equipment_tasks[new_task_idx + 1]
            next_pattern = next_task[2]
            
            # 連続作業制限をチェック
            for constraint in self.continuous_constraints[equipment]:
                if constraint['prev_pattern'] == new_task_pattern and constraint['next_pattern'] == next_pattern:
                    # この設備で連続して実行されるため制限違反
                    return False
        
        return True
    
    def decode_chromosome(self, chromosome):
        """染色体をスケジュールにデコード（固定タスクと連続作業制限を考慮）"""
        # 固定タスクでスケジュールを初期化
        schedule = dict(self.fixed_tasks)
        
        # リソースの忙しい期間を記録
        worker_busy_periods = {}
        equipment_busy_periods = {}
        
        # 固定タスクの期間を記録
        for task_id, info in self.fixed_tasks.items():
            worker = info['worker']
            equipment = info['equipment']
            
            if worker not in worker_busy_periods:
                worker_busy_periods[worker] = []
            worker_busy_periods[worker].append((info['start_time'], info['end_time']))
            
            if equipment not in equipment_busy_periods:
                equipment_busy_periods[equipment] = []
            equipment_busy_periods[equipment].append((info['start_time'], info['end_time']))
        
        # 通常のリソース利用可能時刻（仕様変更：0から開始）
        worker_availability = {worker: 0 for worker in self.workers if worker != 'dummy-type-0001'}
        equipment_availability = {equipment: 0 for equipment in self.equipments if equipment != 'dummy-type-0002'}
        
        for gene in chromosome:
            task_id = gene['task']
            worker = gene['worker']
            equipment = gene['equipment']
            duration = self.tasks[task_id]['duration']
            pattern = self.tasks[task_id]['pattern']
            
            # 最早開始時刻を計算（固定タスクを避ける）
            earliest_start = self.find_available_slot(
                worker_busy_periods, equipment_busy_periods, duration, worker, equipment
            )
            
            # 通常のリソース制約も考慮
            if worker != 'dummy-type-0001':
                earliest_start = max(earliest_start, worker_availability.get(worker, 0))
            if equipment != 'dummy-type-0002':
                earliest_start = max(earliest_start, equipment_availability.get(equipment, 0))
            
            # 順序制約を考慮
            for constraint in self.order_constraints:
                if constraint['successor'] == task_id:
                    pred_id = constraint['predecessor']
                    if pred_id in schedule:
                        pred_time = schedule[pred_id]['end_time'] if constraint['pred_origin'] == '終了' else schedule[pred_id]['start_time']
                        succ_time_point = 'start_time' if constraint['succ_origin'] == '開始' else 'end_time'
                        
                        if succ_time_point == 'start_time':
                            min_start = pred_time + constraint['time_diff_min']
                            earliest_start = max(earliest_start, min_start)
                        else:
                            min_end = pred_time + constraint['time_diff_min']
                            min_start = min_end - duration
                            earliest_start = max(earliest_start, min_start)
            
            # 連続作業制限をチェック（改善: より効率的な回避）
            if not self.check_continuous_constraint(equipment, schedule, pattern, earliest_start):
                # 制約違反の場合、別の時間スロットを探す
                for offset in [30, 60, 120, 240, 480]:  # 段階的に遅延
                    test_start = earliest_start + offset
                    if self.check_continuous_constraint(equipment, schedule, pattern, test_start):
                        earliest_start = test_start
                        break
            
            # スケジュールに追加
            start_time = earliest_start
            end_time = start_time + duration
            
            schedule[task_id] = {
                'start_time': start_time,
                'end_time': end_time,
                'worker': worker,
                'equipment': equipment
            }
            
            # リソースの利用可能時刻を更新
            if worker != 'dummy-type-0001':
                worker_availability[worker] = end_time
                if worker not in worker_busy_periods:
                    worker_busy_periods[worker] = []
                worker_busy_periods[worker].append((start_time, end_time))
            
            if equipment != 'dummy-type-0002':
                equipment_availability[equipment] = end_time
                if equipment not in equipment_busy_periods:
                    equipment_busy_periods[equipment] = []
                equipment_busy_periods[equipment].append((start_time, end_time))
        
        return schedule
    
    def calculate_fitness(self, chromosome):
        """適応度（メイクスパン）の計算（改善版）"""
        schedule = self.decode_chromosome(chromosome)
        
        if schedule is None:
            return float('inf')
        
        # 新しいmakespan計算（仕様変更）
        # 開始時刻：絶対時刻0
        start_point = 0
        
        # 終了時刻：固定タスクを除く通常タスクの最遅終了
        regular_end_times = [
            task['end_time'] for task_id, task in schedule.items() 
            if task_id not in self.fixed_tasks
        ]
        end_point = max(regular_end_times) if regular_end_times else 0
        
        # makespan = 終了時刻 - 開始時刻
        makespan = end_point - start_point
        
        # 制約違反のペナルティを追加（適応的重み付け）
        penalty = self.calculate_constraint_penalty(schedule)
        
        # 適応的ペナルティ重み
        if self.adaptive_penalty:
            # 世代が進むにつれてペナルティを減らし、makespanの改善を重視
            generation_factor = max(0.5, 1.0 - self.current_generation / self.num_generations)
            penalty *= self.penalty_weight * generation_factor
        else:
            penalty *= self.penalty_weight
        
        return makespan + penalty
    
    def calculate_constraint_penalty(self, schedule):
        """制約違反のペナルティ計算（改善版）"""
        penalty = 0
        
        # 順序制約のペナルティ（重みを調整）
        for constraint in self.order_constraints:
            pred_id = constraint['predecessor']
            succ_id = constraint['successor']
            
            if pred_id in schedule and succ_id in schedule:
                pred_time = schedule[pred_id]['end_time'] if constraint['pred_origin'] == '終了' else schedule[pred_id]['start_time']
                succ_time = schedule[succ_id]['end_time'] if constraint['succ_origin'] == '終了' else schedule[succ_id]['start_time']
                
                time_diff = succ_time - pred_time
                
                if time_diff < constraint['time_diff_min']:
                    # ペナルティを違反量に比例させる
                    violation = constraint['time_diff_min'] - time_diff
                    penalty += violation * 0.5  # 重みを減らす
        
        # 連続作業制限のペナルティ（仕様：時間差は考慮しない）
        for equipment, constraints in self.continuous_constraints.items():
            # この設備のタスクを時系列順にソート
            equipment_tasks = []
            for task_id, info in schedule.items():
                if info['equipment'] == equipment:
                    if task_id in self.fixed_tasks:
                        pattern = self.fixed_tasks[task_id]['pattern']
                    else:
                        pattern = self.tasks[task_id]['pattern']
                    equipment_tasks.append((info['start_time'], info['end_time'], pattern, task_id))
            
            equipment_tasks.sort()
            
            # 連続するタスクの制約をチェック（設備上で隣接する場合）
            for i in range(len(equipment_tasks) - 1):
                curr_task = equipment_tasks[i]
                next_task = equipment_tasks[i+1]
                
                # 設備上で連続して実行される（時間差は考慮しない）
                curr_pattern = curr_task[2]
                next_pattern = next_task[2]
                
                for constraint in constraints:
                    if constraint['prev_pattern'] == curr_pattern and constraint['next_pattern'] == next_pattern:
                        # 連続作業制限違反（ペナルティを調整）
                        penalty += 20  # 100から20に減少
        
        return penalty
    
    def crossover(self, parent1, parent2):
        """順序交叉（OX: Order Crossover）- 改善版"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        size = len(parent1)
        
        # より良い交叉点の選択
        start = random.randint(0, size // 2)
        end = random.randint(start + 1, min(start + size // 2, size))
        
        child1 = [None] * size
        child2 = [None] * size
        
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        def fill_remaining(child, parent):
            tasks_in_child = {gene['task'] for gene in child if gene is not None}
            remaining = [gene for gene in parent if gene['task'] not in tasks_in_child]
            
            idx = 0
            for i in range(size):
                if child[i] is None:
                    child[i] = remaining[idx]
                    idx += 1
            return child
        
        child1 = fill_remaining(child1, parent2)
        child2 = fill_remaining(child2, parent1)
        
        return child1, child2
    
    def mutate(self, chromosome):
        """突然変異（改善版：複数の突然変異オペレータ）"""
        if random.random() > self.mutation_rate:
            return chromosome
        
        mutated = copy.deepcopy(chromosome)
        mutation_type = random.random()
        
        if mutation_type < 0.3 and len(mutated) > 1:
            # タスクの入れ替え
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
            
        elif mutation_type < 0.6 and len(mutated) > 2:
            # 部分列の逆転
            start = random.randint(0, len(mutated) - 2)
            end = random.randint(start + 1, min(start + 5, len(mutated)))
            mutated[start:end] = reversed(mutated[start:end])
            
        elif mutation_type < 0.9 and len(mutated) > 0:
            # リソースの再割り当て
            idx = random.randint(0, len(mutated) - 1)
            task_id = mutated[idx]['task']
            pattern = self.tasks[task_id]['pattern']
            
            if pattern in self.allocation_patterns:
                allocation = random.choice(self.allocation_patterns[pattern])
                mutated[idx]['worker'] = allocation['worker']
                mutated[idx]['equipment'] = allocation['equipment']
        else:
            # 挿入突然変異
            if len(mutated) > 1:
                idx = random.randint(0, len(mutated) - 1)
                gene = mutated.pop(idx)
                new_idx = random.randint(0, len(mutated))
                mutated.insert(new_idx, gene)
        
        return mutated
    
    def local_search(self, chromosome):
        """局所探索による改善"""
        if random.random() > self.local_search_rate:
            return chromosome
        
        best_chromosome = copy.deepcopy(chromosome)
        best_fitness = self.calculate_fitness(best_chromosome)
        
        # 近傍解を探索
        for _ in range(5):  # 5つの近傍解を試す
            neighbor = self.mutate(copy.deepcopy(best_chromosome))
            neighbor_fitness = self.calculate_fitness(neighbor)
            
            if neighbor_fitness < best_fitness:
                best_chromosome = neighbor
                best_fitness = neighbor_fitness
        
        return best_chromosome
    
    def selection(self, population, fitness_scores):
        """トーナメント選択（改善版）"""
        selected = []
        
        for _ in range(len(population)):
            tournament_idx = random.sample(range(len(population)), self.tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmin(tournament_fitness)]
            selected.append(copy.deepcopy(population[winner_idx]))
        
        return selected
    
    def run(self):
        """遺伝的アルゴリズムの実行（改善版）"""
        print("Starting Genetic Algorithm V3 (Improved Convergence)...")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.num_generations}")
        print(f"Number of regular tasks: {self.num_regular_tasks}")
        print(f"Number of fixed tasks: {len(self.fixed_tasks)}")
        print("\n[V3 Improvements:]")
        print("- Heuristic initialization")
        print("- Adaptive penalty weights")
        print("- Local search optimization")
        print("- Multiple mutation operators")
        print("- Diversity preservation")
        
        # 初期集団の生成（ヒューリスティック）
        population = [self.create_individual() for _ in range(self.population_size)]
        
        best_fitness_history = []
        avg_fitness_history = []
        diversity_history = []
        best_solution = None
        best_fitness = float('inf')
        
        for generation in range(self.num_generations):
            self.current_generation = generation
            
            fitness_scores = [self.calculate_fitness(ind) for ind in population]
            
            min_fitness_idx = np.argmin(fitness_scores)
            current_best = fitness_scores[min_fitness_idx]
            
            if current_best < best_fitness:
                best_fitness = current_best
                best_solution = copy.deepcopy(population[min_fitness_idx])
                self.best_fitness_no_improve = 0
            else:
                self.best_fitness_no_improve += 1
            
            best_fitness_history.append(best_fitness)
            avg_fitness = np.mean(fitness_scores)
            avg_fitness_history.append(avg_fitness)
            
            # 多様性の計算
            diversity = np.std(fitness_scores)
            diversity_history.append(diversity)
            
            if generation % 50 == 0:
                print(f"Generation {generation}: Best = {best_fitness:.2f}, "
                      f"Avg = {avg_fitness:.2f}, Diversity = {diversity:.2f}")
            
            # 停滞検出と対処
            if self.best_fitness_no_improve > self.stagnation_limit:
                print(f"  Stagnation detected at generation {generation}. Injecting diversity...")
                # 集団の一部を新しい個体で置き換え
                num_replace = self.population_size // 4
                for i in range(num_replace):
                    population[-(i+1)] = self.create_individual()
                self.best_fitness_no_improve = 0
                self.mutation_rate = min(0.5, self.mutation_rate * 1.2)  # 突然変異率を一時的に増加
            
            # エリート選択
            elite_size = int(self.population_size * self.elite_rate)
            elite_idx = np.argsort(fitness_scores)[:elite_size]
            elite = [copy.deepcopy(population[i]) for i in elite_idx]
            
            # 選択
            selected = self.selection(population, fitness_scores)
            
            # 交叉と突然変異
            offspring = []
            for i in range(0, len(selected) - 1, 2):
                child1, child2 = self.crossover(selected[i], selected[i + 1])
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # 局所探索
                child1 = self.local_search(child1)
                child2 = self.local_search(child2)
                
                offspring.extend([child1, child2])
            
            # 次世代の構成
            population = elite + offspring[:self.population_size - elite_size]
            
            # 適応的パラメータ調整
            if generation > 0 and generation % 100 == 0:
                # 収束状況に応じてパラメータを調整
                if diversity < 10:
                    self.mutation_rate = min(0.5, self.mutation_rate * 1.1)
                else:
                    self.mutation_rate = max(0.1, self.mutation_rate * 0.95)
        
        print(f"\nOptimization completed!")
        print(f"Best makespan: {best_fitness:.2f} minutes")
        print(f"Final diversity: {diversity_history[-1]:.2f}")
        
        return best_solution, best_fitness, best_fitness_history, avg_fitness_history
    
    # 以下、検証と可視化メソッドは前バージョンと同じ
    def validate_schedule(self, chromosome):
        """スケジュールの完全な検証（追加制約対応、仕様変更版）"""
        schedule = self.decode_chromosome(chromosome)
        validation_results = {
            'is_valid': True,
            'makespan': 0,
            'violations': [],
            'resource_conflicts': [],
            'constraint_violations': [],
            'continuous_violations': []
        }
        
        # 1. Makespanの計算（仕様変更版）
        if schedule:
            # 開始時刻：絶対時刻0
            start_point = 0
            
            # 終了時刻：固定タスクを除く通常タスクの最遅終了
            regular_end_times = [
                task['end_time'] for task_id, task in schedule.items() 
                if task_id not in self.fixed_tasks
            ]
            end_point = max(regular_end_times) if regular_end_times else 0
            
            # makespan = 終了時刻 - 開始時刻
            validation_results['makespan'] = end_point - start_point
        
        # 2. 順序制約の検証
        for constraint in self.order_constraints:
            pred_id = constraint['predecessor']
            succ_id = constraint['successor']
            
            if pred_id in schedule and succ_id in schedule:
                pred_time = schedule[pred_id]['end_time'] if constraint['pred_origin'] == '終了' else schedule[pred_id]['start_time']
                succ_time = schedule[succ_id]['end_time'] if constraint['succ_origin'] == '終了' else schedule[succ_id]['start_time']
                
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
            if worker != 'dummy-type-0001':
                if worker not in worker_schedules:
                    worker_schedules[worker] = []
                worker_schedules[worker].append((task_id, info['start_time'], info['end_time']))
        
        for worker, tasks in worker_schedules.items():
            tasks_sorted = sorted(tasks, key=lambda x: x[1])
            for i in range(len(tasks_sorted) - 1):
                if tasks_sorted[i][2] > tasks_sorted[i+1][1]:
                    validation_results['is_valid'] = False
                    validation_results['resource_conflicts'].append({
                        'type': 'worker_conflict',
                        'resource': worker,
                        'task1': tasks_sorted[i][0],
                        'task2': tasks_sorted[i+1][0],
                        'overlap': tasks_sorted[i][2] - tasks_sorted[i+1][1]
                    })
        
        # 設備の競合チェック
        equipment_schedules = {}
        for task_id, info in schedule.items():
            equipment = info['equipment']
            if equipment != 'dummy-type-0002':
                if equipment not in equipment_schedules:
                    equipment_schedules[equipment] = []
                equipment_schedules[equipment].append((task_id, info['start_time'], info['end_time']))
        
        for equipment, tasks in equipment_schedules.items():
            tasks_sorted = sorted(tasks, key=lambda x: x[1])
            for i in range(len(tasks_sorted) - 1):
                if tasks_sorted[i][2] > tasks_sorted[i+1][1]:
                    validation_results['is_valid'] = False
                    validation_results['resource_conflicts'].append({
                        'type': 'equipment_conflict',
                        'resource': equipment,
                        'task1': tasks_sorted[i][0],
                        'task2': tasks_sorted[i+1][0],
                        'overlap': tasks_sorted[i][2] - tasks_sorted[i+1][1]
                    })
        
        # 4. 連続作業制限の検証（仕様：時間差は考慮しない）
        for equipment, constraints in self.continuous_constraints.items():
            equipment_tasks = []
            for task_id, info in schedule.items():
                if info['equipment'] == equipment:
                    if task_id in self.fixed_tasks:
                        pattern = self.fixed_tasks[task_id]['pattern']
                    else:
                        pattern = self.tasks[task_id]['pattern']
                    equipment_tasks.append((info['start_time'], info['end_time'], task_id, pattern))
            
            equipment_tasks.sort()
            
            # 設備上で連続するタスクをチェック（時間差は考慮しない）
            for i in range(len(equipment_tasks) - 1):
                curr_task = equipment_tasks[i]
                next_task = equipment_tasks[i+1]
                
                # 設備上で連続して実行される（時間差関係なし）
                curr_pattern = curr_task[3]
                next_pattern = next_task[3]
                
                for constraint in constraints:
                    if constraint['prev_pattern'] == curr_pattern and constraint['next_pattern'] == next_pattern:
                        validation_results['is_valid'] = False
                        validation_results['continuous_violations'].append({
                            'equipment': equipment,
                            'task1': curr_task[2],
                            'task2': next_task[2],
                            'pattern1': curr_pattern,
                            'pattern2': next_pattern,
                            'sequence': f"Task {i+1} -> Task {i+2} on equipment"
                        })
        
        return validation_results
    
    def print_validation_report(self, chromosome):
        """検証レポートの出力（追加制約対応、改善版V3）"""
        results = self.validate_schedule(chromosome)
        
        print("\n" + "="*60)
        print("SCHEDULE VALIDATION REPORT (V3 - Improved Convergence)")
        print("="*60)
        
        print(f"\n✓ Makespan: {results['makespan']:.2f} minutes")
        print(f"  (Calculated as: End time of latest regular task - 0)")
        print(f"✓ Schedule is valid: {results['is_valid']}")
        
        if not results['is_valid']:
            print("\n⚠ VIOLATIONS DETECTED:")
            
            if results['constraint_violations']:
                print("\n[Order Constraint Violations]")
                for v in results['constraint_violations']:
                    print(f"  - {v['predecessor']} -> {v['successor']}: ")
                    print(f"    Required min diff: {v['required_min_diff']}, Actual: {v['actual_diff']:.2f}")
                    print(f"    Violation amount: {v['violation_amount']:.2f} minutes")
            
            if results['resource_conflicts']:
                print("\n[Resource Conflicts]")
                for c in results['resource_conflicts']:
                    print(f"  - {c['type']} on {c['resource']}: ")
                    print(f"    Tasks {c['task1']} and {c['task2']} overlap by {c['overlap']:.2f} minutes")
            
            if results['continuous_violations']:
                print("\n[Continuous Work Restriction Violations]")
                for v in results['continuous_violations']:
                    print(f"  - Equipment {v['equipment']}: ")
                    print(f"    {v['task1']} ({v['pattern1']}) -> {v['task2']} ({v['pattern2']})")
                    print(f"    Consecutive execution not allowed (time gap irrelevant)")
        else:
            print("\n✓ All constraints satisfied!")
            print("✓ No resource conflicts detected!")
            print("✓ All continuous work restrictions satisfied!")
            print(f"✓ Fixed tasks: {len(self.fixed_tasks)} properly integrated")
        
        print("="*60)
        
        return results
    
    def visualize_schedule(self, chromosome):
        """ガントチャートの描画（固定タスクを強調表示）"""
        schedule = self.decode_chromosome(chromosome)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # カラーマップ
        colors = {
            'regular': 'skyblue',
            'fixed': 'salmon'
        }
        
        # 作業者ごとのガントチャート
        worker_tasks = {}
        for task_id, info in schedule.items():
            worker = info['worker']
            if worker not in worker_tasks:
                worker_tasks[worker] = []
            is_fixed = task_id in self.fixed_tasks
            worker_tasks[worker].append((task_id, info['start_time'], info['end_time'], is_fixed))
        
        y_pos = 0
        worker_labels = []
        for worker, tasks in sorted(worker_tasks.items()):
            for task_id, start, end, is_fixed in sorted(tasks, key=lambda x: x[1]):
                color = colors['fixed'] if is_fixed else colors['regular']
                ax1.barh(y_pos, end - start, left=start, height=0.8, 
                        color=color, edgecolor='black', linewidth=0.5)
                # タスクIDを短縮表示
                label = task_id.split('_')[-1] if '_' in task_id else task_id[-4:]
                ax1.text(start + (end - start) / 2, y_pos, label, 
                        ha='center', va='center', fontsize=7)
            worker_labels.append(worker)
            y_pos += 1
        
        ax1.set_yticks(range(len(worker_labels)))
        ax1.set_yticklabels(worker_labels, fontsize=8)
        ax1.set_xlabel('Time (minutes)')
        ax1.set_title('Worker Schedule (Red: Fixed Tasks, Blue: Regular Tasks)')
        ax1.grid(True, alpha=0.3)
        
        # 凡例を追加
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='salmon', label='Fixed Tasks'),
                          Patch(facecolor='skyblue', label='Regular Tasks')]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # 設備ごとのガントチャート
        equipment_tasks = {}
        for task_id, info in schedule.items():
            equipment = info['equipment']
            if equipment not in equipment_tasks:
                equipment_tasks[equipment] = []
            is_fixed = task_id in self.fixed_tasks
            equipment_tasks[equipment].append((task_id, info['start_time'], info['end_time'], is_fixed))
        
        y_pos = 0
        equipment_labels = []
        for equipment, tasks in sorted(equipment_tasks.items()):
            for task_id, start, end, is_fixed in sorted(tasks, key=lambda x: x[1]):
                color = colors['fixed'] if is_fixed else colors['regular']
                ax2.barh(y_pos, end - start, left=start, height=0.8,
                        color=color, edgecolor='black', linewidth=0.5)
                label = task_id.split('_')[-1] if '_' in task_id else task_id[-4:]
                ax2.text(start + (end - start) / 2, y_pos, label, 
                        ha='center', va='center', fontsize=7)
            equipment_labels.append(equipment)
            y_pos += 1
        
        ax2.set_yticks(range(len(equipment_labels)))
        ax2.set_yticklabels(equipment_labels, fontsize=8)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_title('Equipment Schedule (Red: Fixed Tasks, Blue: Regular Tasks)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def plot_convergence(self, best_fitness_history, avg_fitness_history):
        """収束曲線のプロット"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        generations = range(len(best_fitness_history))
        ax.plot(generations, best_fitness_history, 'b-', label='Best Fitness', linewidth=2)
        ax.plot(generations, avg_fitness_history, 'r--', label='Average Fitness', alpha=0.7)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness (Makespan in minutes)')
        ax.set_title('GA Convergence V3 (Improved Convergence)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig

if __name__ == "__main__":
    # テスト実行
    import os
    
    # データセットのパスを確認
    dataset_path = 'toy_dataset_additional.xlsx'
    
    # データセットが存在しない場合は生成
    if not os.path.exists(dataset_path):
        print("Dataset not found. Please generate it first using generate_toy_dataset_additional_v2.py")
        exit(1)
    
    # スケジューラの初期化と実行
    scheduler = JobShopGASchedulerAdditionalV3(dataset_path)
    
    best_solution, best_fitness, best_history, avg_history = scheduler.run()
    
    # 検証レポート
    scheduler.print_validation_report(best_solution)
    
    # 結果の可視化
    gantt_fig = scheduler.visualize_schedule(best_solution)
    gantt_fig.savefig('schedule_gantt_v3.png', dpi=300, bbox_inches='tight')
    
    convergence_fig = scheduler.plot_convergence(best_history, avg_history)
    convergence_fig.savefig('convergence_v3.png', dpi=300, bbox_inches='tight')
    
    plt.show()