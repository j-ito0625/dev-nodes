import pandas as pd
import numpy as np
import random
import copy
import time
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class JobShopGASchedulerFixedOnly:
    def __init__(self, dataset_path: str):
        """
        固定タスクのみ対応の遺伝的アルゴリズムを用いたジョブショップスケジューラ（デバッグ用）
        
        Args:
            dataset_path: Excelデータセットのパス
        """
        self.load_data(dataset_path)
        self.initialize_parameters()
        
    def load_data(self, dataset_path: str):
        """データセットの読み込み（固定タスクのみ）"""
        # 基本データの読み込み
        self.resources_df = pd.read_excel(dataset_path, sheet_name='リソース')
        self.tasks_df = pd.read_excel(dataset_path, sheet_name='task')
        self.allocation_df = pd.read_excel(dataset_path, sheet_name='割り当て情報')
        self.order_constraints_df = pd.read_excel(dataset_path, sheet_name='順序制約')
        
        # 固定タスクの読み込み
        try:
            self.fixed_tasks_df = pd.read_excel(dataset_path, sheet_name='固定タスク')
        except:
            print("Warning: 固定タスクシートが見つかりません。")
            self.fixed_tasks_df = pd.DataFrame()
        
        # 連続作業制限は無視
        self.continuous_constraints_df = pd.DataFrame()
        
        # データの前処理
        self.process_data()
        
    def process_data(self):
        """データの前処理と制約の構築（固定タスクのみ）"""
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
                if resource_str.startswith("("):
                    parts = resource_str.strip("()").replace("'", "").split(",")
                    worker = parts[0].strip()
                    equipment = parts[1].strip()
                else:
                    worker, equipment = eval(resource_str)
                
                self.fixed_tasks[task_id] = {
                    'start_time': row['開始時刻'],
                    'end_time': row['終了時刻'],
                    'worker': worker,
                    'equipment': equipment,
                    'pattern': row['割り当て可能パターン'],
                    'product': row['品種名']
                }
        
        # 連続作業制限は空
        self.continuous_constraints = {}
        
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
        print("連続作業制限: 無効（固定タスクのみモード）")
        
    def initialize_parameters(self):
        """GAパラメータの初期化"""
        self.population_size = 50
        self.crossover_rate = 0.8
        self.mutation_rate = 0.3
        self.elite_rate = 0.1
        self.num_generations = 300
        
    def create_individual(self):
        """個体（染色体）の生成（固定タスクを除外）"""
        chromosome = []
        
        # 通常タスクのみで染色体を構成
        task_order = self.topological_sort_with_randomness()
        
        for task_id in task_order:
            if task_id in self.regular_task_ids:
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
            elif pred in self.fixed_tasks and succ in in_degree:
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
    
    def find_available_slot(self, worker_busy_periods, equipment_busy_periods, duration, worker, equipment):
        """利用可能な時間スロットを見つける"""
        earliest_start = 0
        
        all_busy_periods = []
        
        if worker != 'dummy-type-0001' and worker in worker_busy_periods:
            all_busy_periods.extend(worker_busy_periods[worker])
        if equipment != 'dummy-type-0002' and equipment in equipment_busy_periods:
            all_busy_periods.extend(equipment_busy_periods[equipment])
        
        all_busy_periods.sort(key=lambda x: x[0])
        
        for start, end in all_busy_periods:
            if earliest_start + duration <= start:
                return earliest_start
            earliest_start = max(earliest_start, end)
        
        return earliest_start
    
    def decode_chromosome(self, chromosome):
        """染色体をスケジュールにデコード（固定タスクのみ考慮、連続作業制限なし）"""
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
        
        # 通常のリソース利用可能時刻
        worker_availability = {worker: 0 for worker in self.workers if worker != 'dummy-type-0001'}
        equipment_availability = {equipment: 0 for equipment in self.equipments if equipment != 'dummy-type-0002'}
        
        for gene in chromosome:
            task_id = gene['task']
            worker = gene['worker']
            equipment = gene['equipment']
            duration = self.tasks[task_id]['duration']
            
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
            
            # 連続作業制限はチェックしない
            
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
        """適応度（メイクスパン）の計算"""
        schedule = self.decode_chromosome(chromosome)
        
        if schedule is None:
            return float('inf')
        
        # makespan計算
        start_point = 0
        regular_end_times = [
            task['end_time'] for task_id, task in schedule.items() 
            if task_id not in self.fixed_tasks
        ]
        end_point = max(regular_end_times) if regular_end_times else 0
        
        makespan = end_point - start_point
        
        # 制約違反のペナルティを追加（連続作業制限なし）
        penalty = self.calculate_constraint_penalty(schedule)
        
        return makespan + penalty
    
    def calculate_constraint_penalty(self, schedule):
        """制約違反のペナルティ計算（連続作業制限なし）"""
        penalty = 0
        
        # 順序制約のペナルティのみ
        for constraint in self.order_constraints:
            pred_id = constraint['predecessor']
            succ_id = constraint['successor']
            
            if pred_id in schedule and succ_id in schedule:
                pred_time = schedule[pred_id]['end_time'] if constraint['pred_origin'] == '終了' else schedule[pred_id]['start_time']
                succ_time = schedule[succ_id]['end_time'] if constraint['succ_origin'] == '終了' else schedule[succ_id]['start_time']
                
                time_diff = succ_time - pred_time
                
                if time_diff < constraint['time_diff_min']:
                    penalty += (constraint['time_diff_min'] - time_diff) * 10
        
        # 連続作業制限のペナルティはなし
        
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
        print("Starting GA (Fixed Tasks Only Mode)...")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.num_generations}")
        print(f"Number of regular tasks: {self.num_regular_tasks}")
        print(f"Number of fixed tasks: {len(self.fixed_tasks)}")
        print("Continuous constraints: DISABLED")
        
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
        """スケジュールの検証（固定タスクのみ）"""
        schedule = self.decode_chromosome(chromosome)
        validation_results = {
            'is_valid': True,
            'makespan': 0,
            'constraint_violations': [],
            'resource_conflicts': []
        }
        
        if schedule:
            start_point = 0
            regular_end_times = [
                task['end_time'] for task_id, task in schedule.items() 
                if task_id not in self.fixed_tasks
            ]
            end_point = max(regular_end_times) if regular_end_times else 0
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
        
        return validation_results
    
    def print_validation_report(self, chromosome):
        """検証レポートの出力"""
        results = self.validate_schedule(chromosome)
        
        print("\n" + "="*60)
        print("SCHEDULE VALIDATION REPORT (Fixed Tasks Only)")
        print("="*60)
        
        print(f"\n✓ Makespan: {results['makespan']:.2f} minutes")
        print(f"✓ Schedule is valid: {results['is_valid']}")
        print(f"✓ Fixed tasks integrated: {len(self.fixed_tasks)}")
        print("✓ Continuous constraints: NOT CHECKED (disabled)")
        
        if not results['is_valid']:
            print("\n⚠ VIOLATIONS DETECTED:")
            if results['constraint_violations']:
                print("\n[Order Constraint Violations]")
                for v in results['constraint_violations']:
                    print(f"  - {v['predecessor']} -> {v['successor']}: ")
                    print(f"    Required: {v['required_min_diff']}, Actual: {v['actual_diff']:.2f}")
        
        print("="*60)
        
        return results

if __name__ == "__main__":
    import os
    
    dataset_path = 'toy_dataset_additional.xlsx'
    
    if not os.path.exists(dataset_path):
        print("Dataset not found. Please generate it first.")
        exit(1)
    
    scheduler = JobShopGASchedulerFixedOnly(dataset_path)
    best_solution, best_fitness, best_history, avg_history = scheduler.run()
    scheduler.print_validation_report(best_solution)