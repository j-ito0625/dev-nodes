import pandas as pd
import numpy as np
import random
import copy
import time
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class JobShopGASchedulerContinuousOnly:
    def __init__(self, dataset_path: str):
        """
        連続作業制限のみ対応の遺伝的アルゴリズムを用いたジョブショップスケジューラ（デバッグ用）
        
        Args:
            dataset_path: Excelデータセットのパス
        """
        self.load_data(dataset_path)
        self.initialize_parameters()
        
    def load_data(self, dataset_path: str):
        """データセットの読み込み（連続作業制限のみ）"""
        # 基本データの読み込み
        self.resources_df = pd.read_excel(dataset_path, sheet_name='リソース')
        self.tasks_df = pd.read_excel(dataset_path, sheet_name='task')
        self.allocation_df = pd.read_excel(dataset_path, sheet_name='割り当て情報')
        self.order_constraints_df = pd.read_excel(dataset_path, sheet_name='順序制約')
        
        # 固定タスクは無視
        self.fixed_tasks_df = pd.DataFrame()
        
        # 連続作業制限の読み込み
        try:
            self.continuous_constraints_df = pd.read_excel(dataset_path, sheet_name='連続作業制限')
        except:
            print("Warning: 連続作業制限シートが見つかりません。")
            self.continuous_constraints_df = pd.DataFrame()
        
        # データの前処理
        self.process_data()
        
    def process_data(self):
        """データの前処理と制約の構築（連続作業制限のみ）"""
        # 基本的なタスク情報の辞書化
        self.tasks = {}
        for _, row in self.tasks_df.iterrows():
            task_id = row['タスクID']
            self.tasks[task_id] = {
                'duration': row['所要時間'],
                'pattern': row['割り当て可能パターン'],
                'product': row['品種目']
            }
        
        # 固定タスクは空
        self.fixed_tasks = {}
        
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
        
        # 順序制約の辞書化（固定タスクなし）
        self.order_constraints = []
        for _, row in self.order_constraints_df.iterrows():
            # 固定タスクに関する制約は除外
            if not (row['先行作業ID'].startswith('fix_') or row['後作業ID'].startswith('fix_')):
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
        
        # 通常タスクのみ（固定タスクなし）
        self.regular_task_ids = list(self.tasks.keys())
        self.num_regular_tasks = len(self.regular_task_ids)
        
        print(f"通常タスク数: {self.num_regular_tasks}")
        print("固定タスク: 無効（連続作業制限のみモード）")
        print(f"連続作業制限数: {sum(len(v) for v in self.continuous_constraints.values())}")
        
    def initialize_parameters(self):
        """GAパラメータの初期化"""
        self.population_size = 50
        self.crossover_rate = 0.8
        self.mutation_rate = 0.3
        self.elite_rate = 0.1
        self.num_generations = 300
        
    def create_individual(self):
        """個体（染色体）の生成"""
        chromosome = []
        
        task_order = self.topological_sort_with_randomness()
        
        for task_id in task_order:
            pattern = self.tasks[task_id]['pattern']
            if pattern in self.allocation_patterns:
                allocation = random.choice(self.allocation_patterns[pattern])
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
    
    def topological_sort_with_randomness(self):
        """順序制約を満たしつつランダム性を持たせたトポロジカルソート"""
        in_degree = {task: 0 for task in self.regular_task_ids}
        graph = {task: [] for task in self.regular_task_ids}
        
        for constraint in self.order_constraints:
            pred = constraint['predecessor']
            succ = constraint['successor']
            
            if pred in graph and succ in in_degree:
                graph[pred].append(succ)
                in_degree[succ] += 1
        
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
        
        for task in self.regular_task_ids:
            if task not in result:
                result.append(task)
        
        return result
    
    def check_continuous_constraint(self, equipment, schedule, new_task_pattern, start_time):
        """
        連続作業制限のチェック
        時間差は考慮しない（設備上で連続して実行される場合は禁止）
        """
        if equipment not in self.continuous_constraints:
            return True
        
        # この設備で実行されるタスクを時系列順にソート
        equipment_tasks = []
        for task_id, info in schedule.items():
            if info['equipment'] == equipment:
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
        
        # 直前のタスクがある場合
        if new_task_idx > 0:
            prev_task = equipment_tasks[new_task_idx - 1]
            prev_pattern = prev_task[2]
            
            for constraint in self.continuous_constraints[equipment]:
                if constraint['prev_pattern'] == prev_pattern and constraint['next_pattern'] == new_task_pattern:
                    return False
        
        # 直後のタスクがある場合
        if new_task_idx < len(equipment_tasks) - 1:
            next_task = equipment_tasks[new_task_idx + 1]
            next_pattern = next_task[2]
            
            for constraint in self.continuous_constraints[equipment]:
                if constraint['prev_pattern'] == new_task_pattern and constraint['next_pattern'] == next_pattern:
                    return False
        
        return True
    
    def decode_chromosome(self, chromosome):
        """染色体をスケジュールにデコード（連続作業制限のみ考慮、固定タスクなし）"""
        schedule = {}
        
        # リソース利用可能時刻
        worker_availability = {worker: 0 for worker in self.workers if worker != 'dummy-type-0001'}
        equipment_availability = {equipment: 0 for equipment in self.equipments if equipment != 'dummy-type-0002'}
        
        for gene in chromosome:
            task_id = gene['task']
            worker = gene['worker']
            equipment = gene['equipment']
            duration = self.tasks[task_id]['duration']
            pattern = self.tasks[task_id]['pattern']
            
            # 最早開始時刻を計算
            earliest_start = 0
            
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
            
            # 連続作業制限をチェック
            if not self.check_continuous_constraint(equipment, schedule, pattern, earliest_start):
                # 制約違反の場合、少し遅らせる
                for offset in [30, 60, 120, 240, 480]:
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
            
            if equipment != 'dummy-type-0002':
                equipment_availability[equipment] = end_time
        
        return schedule
    
    def calculate_fitness(self, chromosome):
        """適応度（メイクスパン）の計算"""
        schedule = self.decode_chromosome(chromosome)
        
        if schedule is None:
            return float('inf')
        
        # makespan計算
        start_point = 0
        end_times = [task['end_time'] for task in schedule.values()]
        end_point = max(end_times) if end_times else 0
        
        makespan = end_point - start_point
        
        # 制約違反のペナルティを追加
        penalty = self.calculate_constraint_penalty(schedule)
        
        return makespan + penalty
    
    def calculate_constraint_penalty(self, schedule):
        """制約違反のペナルティ計算（連続作業制限あり）"""
        penalty = 0
        
        # 順序制約のペナルティ
        for constraint in self.order_constraints:
            pred_id = constraint['predecessor']
            succ_id = constraint['successor']
            
            if pred_id in schedule and succ_id in schedule:
                pred_time = schedule[pred_id]['end_time'] if constraint['pred_origin'] == '終了' else schedule[pred_id]['start_time']
                succ_time = schedule[succ_id]['end_time'] if constraint['succ_origin'] == '終了' else schedule[succ_id]['start_time']
                
                time_diff = succ_time - pred_time
                
                if time_diff < constraint['time_diff_min']:
                    penalty += (constraint['time_diff_min'] - time_diff) * 10
        
        # 連続作業制限のペナルティ
        for equipment, constraints in self.continuous_constraints.items():
            equipment_tasks = []
            for task_id, info in schedule.items():
                if info['equipment'] == equipment:
                    pattern = self.tasks[task_id]['pattern']
                    equipment_tasks.append((info['start_time'], info['end_time'], pattern, task_id))
            
            equipment_tasks.sort()
            
            # 連続するタスクの制約をチェック（時間差は考慮しない）
            for i in range(len(equipment_tasks) - 1):
                curr_task = equipment_tasks[i]
                next_task = equipment_tasks[i+1]
                
                curr_pattern = curr_task[2]
                next_pattern = next_task[2]
                
                for constraint in constraints:
                    if constraint['prev_pattern'] == curr_pattern and constraint['next_pattern'] == next_pattern:
                        penalty += 100  # 連続作業制限違反
        
        return penalty
    
    def crossover(self, parent1, parent2):
        """順序交叉（OX: Order Crossover）"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
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
        """突然変異"""
        if random.random() > self.mutation_rate:
            return chromosome
        
        mutated = copy.deepcopy(chromosome)
        
        if random.random() < 0.5 and len(mutated) > 1:
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        if random.random() < 0.5 and len(mutated) > 0:
            idx = random.randint(0, len(mutated) - 1)
            task_id = mutated[idx]['task']
            pattern = self.tasks[task_id]['pattern']
            
            if pattern in self.allocation_patterns:
                allocation = random.choice(self.allocation_patterns[pattern])
                mutated[idx]['worker'] = allocation['worker']
                mutated[idx]['equipment'] = allocation['equipment']
        
        return mutated
    
    def selection(self, population, fitness_scores):
        """トーナメント選択"""
        tournament_size = 3
        selected = []
        
        for _ in range(len(population)):
            tournament_idx = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmin(tournament_fitness)]
            selected.append(copy.deepcopy(population[winner_idx]))
        
        return selected
    
    def run(self):
        """遺伝的アルゴリズムの実行"""
        print("Starting GA (Continuous Constraints Only Mode)...")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.num_generations}")
        print(f"Number of tasks: {self.num_regular_tasks}")
        print("Fixed tasks: DISABLED")
        print(f"Continuous constraints: {sum(len(v) for v in self.continuous_constraints.values())}")
        
        population = [self.create_individual() for _ in range(self.population_size)]
        
        best_fitness_history = []
        avg_fitness_history = []
        best_solution = None
        best_fitness = float('inf')
        
        for generation in range(self.num_generations):
            fitness_scores = [self.calculate_fitness(ind) for ind in population]
            
            min_fitness_idx = np.argmin(fitness_scores)
            if fitness_scores[min_fitness_idx] < best_fitness:
                best_fitness = fitness_scores[min_fitness_idx]
                best_solution = copy.deepcopy(population[min_fitness_idx])
            
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(np.mean(fitness_scores))
            
            if generation % 50 == 0:
                print(f"Generation {generation}: Best = {best_fitness:.2f}, Avg = {np.mean(fitness_scores):.2f}")
            
            elite_size = int(self.population_size * self.elite_rate)
            elite_idx = np.argsort(fitness_scores)[:elite_size]
            elite = [copy.deepcopy(population[i]) for i in elite_idx]
            
            selected = self.selection(population, fitness_scores)
            
            offspring = []
            for i in range(0, len(selected) - 1, 2):
                child1, child2 = self.crossover(selected[i], selected[i + 1])
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                offspring.extend([child1, child2])
            
            population = elite + offspring[:self.population_size - elite_size]
        
        print(f"\nOptimization completed!")
        print(f"Best makespan: {best_fitness:.2f} minutes")
        
        return best_solution, best_fitness, best_fitness_history, avg_fitness_history
    
    def validate_schedule(self, chromosome):
        """スケジュールの検証（連続作業制限あり）"""
        schedule = self.decode_chromosome(chromosome)
        validation_results = {
            'is_valid': True,
            'makespan': 0,
            'constraint_violations': [],
            'continuous_violations': []
        }
        
        if schedule:
            start_point = 0
            end_times = [task['end_time'] for task in schedule.values()]
            end_point = max(end_times) if end_times else 0
            validation_results['makespan'] = end_point - start_point
        
        # 順序制約の検証
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
                        'actual_diff': time_diff
                    })
        
        # 連続作業制限の検証
        for equipment, constraints in self.continuous_constraints.items():
            equipment_tasks = []
            for task_id, info in schedule.items():
                if info['equipment'] == equipment:
                    pattern = self.tasks[task_id]['pattern']
                    equipment_tasks.append((info['start_time'], info['end_time'], task_id, pattern))
            
            equipment_tasks.sort()
            
            for i in range(len(equipment_tasks) - 1):
                curr_task = equipment_tasks[i]
                next_task = equipment_tasks[i+1]
                
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
                            'pattern2': next_pattern
                        })
        
        return validation_results
    
    def print_validation_report(self, chromosome):
        """検証レポートの出力"""
        results = self.validate_schedule(chromosome)
        
        print("\n" + "="*60)
        print("SCHEDULE VALIDATION REPORT (Continuous Constraints Only)")
        print("="*60)
        
        print(f"\n✓ Makespan: {results['makespan']:.2f} minutes")
        print(f"✓ Schedule is valid: {results['is_valid']}")
        print("✓ Fixed tasks: NOT CHECKED (disabled)")
        print(f"✓ Continuous constraints checked: {len(self.continuous_constraints)} equipments")
        
        if not results['is_valid']:
            print("\n⚠ VIOLATIONS DETECTED:")
            
            if results['constraint_violations']:
                print("\n[Order Constraint Violations]")
                for v in results['constraint_violations']:
                    print(f"  - {v['predecessor']} -> {v['successor']}: ")
                    print(f"    Required: {v['required_min_diff']}, Actual: {v['actual_diff']:.2f}")
            
            if results['continuous_violations']:
                print("\n[Continuous Work Restriction Violations]")
                for v in results['continuous_violations']:
                    print(f"  - Equipment {v['equipment']}: ")
                    print(f"    {v['task1']} ({v['pattern1']}) -> {v['task2']} ({v['pattern2']})")
                    print(f"    Consecutive execution not allowed")
        
        print("="*60)
        
        return results

if __name__ == "__main__":
    import os
    
    dataset_path = 'toy_dataset_additional.xlsx'
    
    if not os.path.exists(dataset_path):
        print("Dataset not found. Please generate it first.")
        exit(1)
    
    scheduler = JobShopGASchedulerContinuousOnly(dataset_path)
    best_solution, best_fitness, best_history, avg_history = scheduler.run()
    scheduler.print_validation_report(best_solution)