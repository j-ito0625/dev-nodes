"""
分散遺伝的アルゴリズムによるジョブショップスケジューラ（dev版対応）
ga_jobshop_scheduler_dev.pyのダミーリソース処理等を含む分散処理版
"""

import pandas as pd
import numpy as np
import random
import copy
import time
from typing import List, Dict, Tuple
from multiprocessing import Pool, Manager, Process, Queue
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

class DistributedGASchedulerDev:
    """
    分散処理対応のGAスケジューラ（dev版）
    - 島モデル（Island Model）による並列GA
    - 適応度計算の並列化
    - 非同期進化
    - ダミーリソース処理対応
    """
    
    def __init__(self, dataset_path: str, num_islands: int = 4):
        """
        Args:
            dataset_path: データセットのパス
            num_islands: 島（サブ個体群）の数
        """
        self.dataset_path = dataset_path
        self.num_islands = num_islands
        self.load_data(dataset_path)
        self.initialize_parameters()
        
    def load_data(self, dataset_path: str):
        """データセットの読み込み"""
        self.resources_df = pd.read_excel(dataset_path, sheet_name='リソース')
        self.tasks_df = pd.read_excel(dataset_path, sheet_name='task')
        self.allocation_df = pd.read_excel(dataset_path, sheet_name='割り当て情報')
        self.order_constraints_df = pd.read_excel(dataset_path, sheet_name='順序制約')
        self.process_data()
        
    def process_data(self):
        """データの前処理と制約の構築"""
        # タスク情報の辞書化
        self.tasks = {}
        for _, row in self.tasks_df.iterrows():
            task_id = row['タスクID']
            self.tasks[task_id] = {
                'duration': row['所要時間'],
                'pattern': row['割り当て可能パターン'],
                'product': row['品種目']
            }
        
        # 割り当て可能パターンの辞書化
        self.allocation_patterns = {}
        for _, row in self.allocation_df.iterrows():
            pattern = row['割り当て可能パターン']
            if pattern not in self.allocation_patterns:
                self.allocation_patterns[pattern] = []
            self.allocation_patterns[pattern].append({
                'worker': row['作業者'],
                'equipment': row['設備']
            })
        
        # 順序制約の辞書化
        self.order_constraints = []
        for _, row in self.order_constraints_df.iterrows():
            self.order_constraints.append({
                'predecessor': row['先行作業ID'],
                'successor': row['後作業ID'],
                'pred_origin': row['先行作業原点'],
                'succ_origin': row['後作業原点'],
                'time_diff_min': row['時間差下限']
            })
        
        self.workers = [w for w in self.resources_df['作業者'].dropna()]
        self.equipments = [e for e in self.resources_df['設備'].dropna()]
        self.num_tasks = len(self.tasks)
        self.task_ids = list(self.tasks.keys())
        
    def initialize_parameters(self):
        """パラメータ初期化"""
        self.population_size_per_island = 25  # 各島の個体数
        self.crossover_rate = 0.8
        self.mutation_rate = 0.3
        self.elite_rate = 0.1
        self.num_generations = 100
        self.migration_interval = 10  # 移住間隔（世代）
        self.migration_rate = 0.1  # 移住率
        
    def parallel_fitness_evaluation(self, population):
        """
        適応度計算の並列化
        マルチプロセシングによる高速化
        """
        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = [executor.submit(self.calculate_fitness_wrapper, ind) for ind in population]
            fitness_scores = [future.result() for future in as_completed(futures)]
        return fitness_scores
    
    def calculate_fitness_wrapper(self, chromosome):
        """適応度計算のラッパー（プロセス間通信用）"""
        return self.calculate_fitness(chromosome)
    
    def calculate_fitness(self, chromosome):
        """適応度計算"""
        schedule = self.decode_chromosome(chromosome)
        if schedule is None:
            return float('inf')
        
        makespan = max([task['end_time'] for task in schedule.values()])
        penalty = self.calculate_constraint_penalty(schedule)
        return makespan + penalty
    
    def decode_chromosome(self, chromosome):
        """染色体をスケジュールにデコード（ダミーリソース対応）"""
        schedule = {}
        worker_availability = {worker: 0 for worker in self.workers if worker != 'dummy-type-0001'}
        equipment_availability = {equipment: 0 for equipment in self.equipments if equipment != 'dummy-type-0002'}
        
        for gene in chromosome:
            task_id = gene['task']
            worker = gene['worker']
            equipment = gene['equipment']
            duration = self.tasks[task_id]['duration']
            
            # ダミーリソースの処理
            if worker != 'dummy-type-0001':
                worker_start = worker_availability.get(worker, 0)
            else:
                worker_start = 0
            
            if equipment != 'dummy-type-0002':
                equipment_start = equipment_availability.get(equipment, 0)
            else:
                equipment_start = 0
            
            earliest_start = max(worker_start, equipment_start)
            
            # 順序制約の考慮
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
                            # 終了時刻制約の場合
                            min_end = pred_time + constraint['time_diff_min']
                            min_start = min_end - duration
                            earliest_start = max(earliest_start, min_start)
            
            # スケジュールに追加
            start_time = earliest_start
            end_time = start_time + duration
            
            schedule[task_id] = {
                'start_time': start_time,
                'end_time': end_time,
                'worker': worker,
                'equipment': equipment
            }
            
            # リソースの利用可能時刻を更新（ダミーリソース除外）
            if worker != 'dummy-type-0001':
                worker_availability[worker] = end_time
            if equipment != 'dummy-type-0002':
                equipment_availability[equipment] = end_time
        
        return schedule
    
    def calculate_constraint_penalty(self, schedule):
        """制約違反ペナルティ計算"""
        penalty = 0
        for constraint in self.order_constraints:
            pred_id = constraint['predecessor']
            succ_id = constraint['successor']
            
            if pred_id in schedule and succ_id in schedule:
                pred_time = schedule[pred_id]['end_time'] if constraint['pred_origin'] == '終了' else schedule[pred_id]['start_time']
                succ_time = schedule[succ_id]['end_time'] if constraint['succ_origin'] == '終了' else schedule[succ_id]['start_time']
                
                time_diff = succ_time - pred_time
                if time_diff < constraint['time_diff_min']:
                    penalty += (constraint['time_diff_min'] - time_diff) * 10
        
        return penalty
    
    def validate_schedule(self, chromosome):
        """スケジュールの完全な検証（ダミーリソース対応）"""
        schedule = self.decode_chromosome(chromosome)
        validation_results = {
            'is_valid': True,
            'makespan': 0,
            'violations': [],
            'resource_conflicts': [],
            'constraint_violations': []
        }
        
        # 1. Makespanの計算
        if schedule:
            validation_results['makespan'] = max([task['end_time'] for task in schedule.values()])
        
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
        
        # 3. リソース競合の検証（ダミーリソース除外）
        # 作業者の競合チェック
        worker_schedules = {}
        for task_id, info in schedule.items():
            worker = info['worker']
            if worker not in worker_schedules:
                worker_schedules[worker] = []
            worker_schedules[worker].append((task_id, info['start_time'], info['end_time']))
        
        for worker, tasks in worker_schedules.items():
            if worker != 'dummy-type-0001':  # ダミー作業者は除外
                tasks_sorted = sorted(tasks, key=lambda x: x[1])
                for i in range(len(tasks_sorted) - 1):
                    if tasks_sorted[i][2] > tasks_sorted[i+1][1]:  # 終了時刻 > 次の開始時刻
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
            if equipment != 'dummy-type-0002':  # ダミー設備は除外
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
    
    def print_validation_report(self, chromosome):
        """検証レポートの出力"""
        results = self.validate_schedule(chromosome)
        
        print("\n" + "="*60)
        print("SCHEDULE VALIDATION REPORT")
        print("="*60)
        
        print(f"\n✓ Makespan: {results['makespan']:.2f} minutes")
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
            
            if results['violations']:
                print("\n[Allocation Violations]")
                for v in results['violations']:
                    if v['type'] == 'allocation_violation':
                        print(f"  - Task {v['task']}: ")
                        print(f"    Assigned: Worker={v['assigned_worker']}, Equipment={v['assigned_equipment']}")
                        print(f"    Not in allowed patterns")
        else:
            print("\n✓ All constraints satisfied!")
            print("✓ No resource conflicts detected!")
            print("✓ All allocation patterns valid!")
        
        print("="*60)
        
        return results
    
    def evolve_island(self, island_id: int, population: List, generations: int, 
                     migration_queue: Queue, result_queue: Queue):
        """
        単一島での進化プロセス
        各島は独立したプロセスで実行
        """
        print(f"Island {island_id}: Starting evolution...")
        best_fitness = float('inf')
        best_solution = None
        
        for gen in range(generations):
            # 適応度計算
            fitness_scores = [self.calculate_fitness(ind) for ind in population]
            
            # 最良個体の更新
            min_idx = np.argmin(fitness_scores)
            if fitness_scores[min_idx] < best_fitness:
                best_fitness = fitness_scores[min_idx]
                best_solution = copy.deepcopy(population[min_idx])
            
            # 移住処理
            if gen % self.migration_interval == 0 and gen > 0:
                # 優良個体を他の島へ送信
                num_migrants = int(len(population) * self.migration_rate)
                elite_indices = np.argsort(fitness_scores)[:num_migrants]
                migrants = [copy.deepcopy(population[i]) for i in elite_indices]
                
                migration_queue.put({
                    'from_island': island_id,
                    'migrants': migrants,
                    'generation': gen
                })
                
                # 他の島からの移民を受け入れ
                try:
                    while not migration_queue.empty():
                        migration_data = migration_queue.get_nowait()
                        if migration_data['from_island'] != island_id:
                            # ランダムな個体を移民で置換
                            for migrant in migration_data['migrants']:
                                replace_idx = random.randint(0, len(population) - 1)
                                population[replace_idx] = copy.deepcopy(migrant)
                except:
                    pass
            
            # 選択、交叉、突然変異（通常のGA操作）
            population = self.genetic_operations(population, fitness_scores)
            
            if gen % 20 == 0:
                print(f"Island {island_id}, Gen {gen}: Best fitness = {best_fitness:.2f}")
        
        # 最終結果を返す
        result_queue.put({
            'island_id': island_id,
            'best_solution': best_solution,
            'best_fitness': best_fitness
        })
        
        print(f"Island {island_id}: Evolution completed. Best fitness = {best_fitness:.2f}")
    
    def genetic_operations(self, population, fitness_scores):
        """遺伝的操作（選択、交叉、突然変異）"""
        new_population = []
        
        # エリート保存
        elite_size = int(len(population) * self.elite_rate)
        elite_indices = np.argsort(fitness_scores)[:elite_size]
        for idx in elite_indices:
            new_population.append(copy.deepcopy(population[idx]))
        
        # 選択と交叉
        while len(new_population) < len(population):
            # トーナメント選択
            parent1 = self.tournament_selection(population, fitness_scores)
            parent2 = self.tournament_selection(population, fitness_scores)
            
            # 交叉
            if random.random() < self.crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                child = copy.deepcopy(parent1)
            
            # 突然変異
            if random.random() < self.mutation_rate:
                child = self.mutate(child)
            
            new_population.append(child)
        
        return new_population[:len(population)]
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """トーナメント選択"""
        indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in indices]
        winner_idx = indices[np.argmin(tournament_fitness)]
        return copy.deepcopy(population[winner_idx])
    
    def crossover(self, parent1, parent2):
        """順序交叉（OX）"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [None] * size
        child[start:end] = parent1[start:end]
        
        tasks_in_child = {gene['task'] for gene in child if gene is not None}
        remaining = [gene for gene in parent2 if gene['task'] not in tasks_in_child]
        
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = remaining[idx]
                idx += 1
        
        return child
    
    def mutate(self, chromosome):
        """突然変異"""
        mutated = copy.deepcopy(chromosome)
        
        # タスクの順序入れ替え
        if random.random() < 0.5:
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        # リソース再割り当て
        if random.random() < 0.5:
            idx = random.randint(0, len(mutated) - 1)
            task_id = mutated[idx]['task']
            pattern = self.tasks[task_id]['pattern']
            
            if pattern in self.allocation_patterns:
                allocation = random.choice(self.allocation_patterns[pattern])
                mutated[idx]['worker'] = allocation['worker']
                mutated[idx]['equipment'] = allocation['equipment']
        
        return mutated
    
    def create_individual(self):
        """個体生成（トポロジカルソート対応）"""
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
        # 各タスクの入次数を計算
        in_degree = {task: 0 for task in self.task_ids}
        graph = {task: [] for task in self.task_ids}
        
        for constraint in self.order_constraints:
            pred = constraint['predecessor']
            succ = constraint['successor']
            if pred in graph and succ in in_degree:
                graph[pred].append(succ)
                in_degree[succ] += 1
        
        # トポロジカルソート（ランダム性あり）
        result = []
        available = [task for task in self.task_ids if in_degree[task] == 0]
        
        while available:
            # ランダムに次のタスクを選択
            random.shuffle(available)
            task = available.pop(0)
            result.append(task)
            
            # 後続タスクの入次数を減らす
            for successor in graph[task]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    available.append(successor)
        
        # 制約に関係ないタスクを追加
        for task in self.task_ids:
            if task not in result:
                result.append(task)
        
        return result
    
    def run_distributed(self):
        """
        分散GAの実行
        複数の島で並列に進化を実行
        """
        print(f"\n{'='*60}")
        print("Starting Distributed Genetic Algorithm (Dev Version)")
        print(f"{'='*60}")
        print(f"Number of islands: {self.num_islands}")
        print(f"Population per island: {self.population_size_per_island}")
        print(f"Total population: {self.num_islands * self.population_size_per_island}")
        print(f"Generations: {self.num_generations}")
        print(f"Migration interval: every {self.migration_interval} generations")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # マネージャーとキューの作成
        manager = Manager()
        migration_queue = manager.Queue()
        result_queue = manager.Queue()
        
        # 各島の初期個体群を生成
        island_populations = []
        for i in range(self.num_islands):
            population = [self.create_individual() for _ in range(self.population_size_per_island)]
            island_populations.append(population)
        
        # 各島を別プロセスで実行
        processes = []
        for i in range(self.num_islands):
            p = Process(target=self.evolve_island, 
                       args=(i, island_populations[i], self.num_generations, 
                            migration_queue, result_queue))
            p.start()
            processes.append(p)
        
        # すべてのプロセスの完了を待つ
        for p in processes:
            p.join()
        
        # 結果の収集
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # 最良解の選択
        best_island = min(results, key=lambda x: x['best_fitness'])
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("Distributed GA Completed!")
        print(f"{'='*60}")
        print(f"Best fitness: {best_island['best_fitness']:.2f}")
        print(f"Best island: {best_island['island_id']}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Speedup: {self.num_islands:.1f}x theoretical maximum")
        print(f"{'='*60}\n")
        
        return best_island['best_solution'], best_island['best_fitness'], results
    
    def run_comparison(self):
        """
        通常のGAと分散GAの性能比較
        """
        print("\n=== Performance Comparison ===\n")
        
        # 通常のGA（単一プロセス）
        print("Running standard GA (single process)...")
        start_time = time.time()
        
        population = [self.create_individual() 
                     for _ in range(self.num_islands * self.population_size_per_island)]
        best_fitness_single = float('inf')
        
        for gen in range(self.num_generations):
            fitness_scores = [self.calculate_fitness(ind) for ind in population]
            min_idx = np.argmin(fitness_scores)
            if fitness_scores[min_idx] < best_fitness_single:
                best_fitness_single = fitness_scores[min_idx]
            
            population = self.genetic_operations(population, fitness_scores)
            
            if gen % 20 == 0:
                print(f"  Gen {gen}: Best fitness = {best_fitness_single:.2f}")
        
        single_time = time.time() - start_time
        
        # 分散GA
        print("\nRunning distributed GA (multi-process)...")
        best_solution, best_fitness_dist, _ = self.run_distributed()
        dist_time = time.time() - start_time - single_time
        
        # 結果の比較
        print("\n=== Results ===")
        print(f"Single Process GA:")
        print(f"  Time: {single_time:.2f} seconds")
        print(f"  Best fitness: {best_fitness_single:.2f}")
        print(f"\nDistributed GA ({self.num_islands} islands):")
        print(f"  Time: {dist_time:.2f} seconds")
        print(f"  Best fitness: {best_fitness_dist:.2f}")
        print(f"\nSpeedup: {single_time/dist_time:.2f}x")
        print(f"Efficiency: {(single_time/dist_time)/self.num_islands:.2%}")
        
        return best_solution, best_fitness_dist


if __name__ == "__main__":
    # 使用例
    scheduler = DistributedGASchedulerDev('toy_dataset.xlsx', num_islands=4)
    
    # 分散GAの実行
    best_solution, best_fitness, results = scheduler.run_distributed()
    
    # 結果の検証レポート
    scheduler.print_validation_report(best_solution)
    
    # 性能比較の実行
    # scheduler.run_comparison()