import pandas as pd
import numpy as np
import random
import copy
import time
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class JobShopGAScheduler:
    def __init__(self, dataset_path: str):
        """
        遺伝的アルゴリズムを用いたジョブショップスケジューラ
        
        Args:
            dataset_path: Excelデータセットのパス
        """
        self.load_data(dataset_path)
        self.initialize_parameters()
        
    def load_data(self, dataset_path: str):
        """データセットの読み込み"""
        self.resources_df = pd.read_excel(dataset_path, sheet_name='リソース')
        self.tasks_df = pd.read_excel(dataset_path, sheet_name='task')
        self.allocation_df = pd.read_excel(dataset_path, sheet_name='割り当て情報')
        self.order_constraints_df = pd.read_excel(dataset_path, sheet_name='順序制約')
        
        # データの前処理
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
        
        # リソースのリスト化
        self.workers = [w for w in self.resources_df['作業者'].dropna()]
        self.equipments = [e for e in self.resources_df['設備'].dropna()]
        
        self.num_tasks = len(self.tasks)
        self.task_ids = list(self.tasks.keys())
        
    def initialize_parameters(self):
        """GAパラメータの初期化"""
        self.population_size = 50
        self.crossover_rate = 0.8
        self.mutation_rate = 0.3
        self.elite_rate = 0.1
        self.num_generations = 500
        
    def create_individual(self):
        """個体（染色体）の生成"""
        # 染色体の構造: [(task_id, worker, equipment), ...]
        chromosome = []
        
        # タスクの順序をランダムにシャッフル（トポロジカルソートを考慮）
        task_order = self.topological_sort_with_randomness()
        
        for task_id in task_order:
            pattern = self.tasks[task_id]['pattern']
            # 割り当て可能な組み合わせからランダムに選択
            if pattern in self.allocation_patterns:
                allocation = random.choice(self.allocation_patterns[pattern])
                chromosome.append({
                    'task': task_id,
                    'worker': allocation['worker'],
                    'equipment': allocation['equipment']
                })
            else:
                # パターンが見つからない場合はランダムに割り当て
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
    
    def calculate_fitness(self, chromosome):
        """適応度（メイクスパン）の計算"""
        schedule = self.decode_chromosome(chromosome)
        
        if schedule is None:
            return float('inf')  # 制約違反の場合
        
        # メイクスパンを計算
        makespan = max([task['end_time'] for task in schedule.values()])
        
        # 制約違反のペナルティを追加
        penalty = self.calculate_constraint_penalty(schedule)
        
        return makespan + penalty
    
    def decode_chromosome(self, chromosome):
        """染色体をスケジュールにデコード"""
        schedule = {}
        worker_availability = {worker: 0 for worker in self.workers if worker != 'dummy-type-0001'}
        equipment_availability = {equipment: 0 for equipment in self.equipments if equipment != 'dummy-type-0002'}
        
        for gene in chromosome:
            task_id = gene['task']
            worker = gene['worker']
            equipment = gene['equipment']
            duration = self.tasks[task_id]['duration']
            
            if worker != 'dummy-type-0001':
                worker_start = worker_availability.get(worker, 0)
            else:
                worker_start = 0
            if equipment != 'dummy-type-0002':
                equipment_start = equipment_availability.get(equipment, 0)
            else:
                equipment_start = 0
            earliest_start = max(worker_start, equipment_start)
            # 最早開始時刻を計算
            # earliest_start = max(
            #     worker_availability.get(worker, 0),
            #     equipment_availability.get(equipment, 0)
            # )
            
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
            
            # リソースの利用可能時刻を更新
            # worker_availability[worker] = end_time
            # equipment_availability[equipment] = end_time
            if worker != 'dummy-type-0001':
                worker_availability[worker] = end_time
            if equipment != 'dummy-type-0002':
                equipment_availability[equipment] = end_time

        return schedule
    
    def calculate_constraint_penalty(self, schedule):
        """制約違反のペナルティ計算"""
        penalty = 0
        
        for constraint in self.order_constraints:
            pred_id = constraint['predecessor']
            succ_id = constraint['successor']
            
            if pred_id in schedule and succ_id in schedule:
                pred_time = schedule[pred_id]['end_time'] if constraint['pred_origin'] == '終了' else schedule[pred_id]['start_time']
                succ_time = schedule[succ_id]['end_time'] if constraint['succ_origin'] == '終了' else schedule[succ_id]['start_time']
                
                time_diff = succ_time - pred_time
                
                if time_diff < constraint['time_diff_min']:
                    penalty += (constraint['time_diff_min'] - time_diff) * 10  # ペナルティ係数
        
        return penalty
    
    def validate_schedule(self, chromosome):
        """スケジュールの完全な検証"""
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
        
        # 3. リソース競合の検証（同一リソースの重複利用チェック）
        # 作業者の競合チェック
        worker_schedules = {}
        for task_id, info in schedule.items():
            worker = info['worker']
            if worker not in worker_schedules:
                worker_schedules[worker] = []
            worker_schedules[worker].append((task_id, info['start_time'], info['end_time']))
        
        for worker, tasks in worker_schedules.items():
            if worker != 'dummy-type-0001':
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
        
        # 4. 割り当て制約の検証（タスクが許可されたリソースを使用しているか）
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
    
    def crossover(self, parent1, parent2):
        """順序交叉（OX: Order Crossover）"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        size = len(parent1)
        # 交叉点をランダムに選択
        start, end = sorted(random.sample(range(size), 2))
        
        # 子個体の初期化
        child1 = [None] * size
        child2 = [None] * size
        
        # 親1の部分を子1にコピー
        child1[start:end] = parent1[start:end]
        child2[start:end] = parent2[start:end]
        
        # 残りの遺伝子を順番に埋める
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
        """突然変異（タスクの入れ替えとリソースの再割り当て）"""
        if random.random() > self.mutation_rate:
            return chromosome
        
        mutated = copy.deepcopy(chromosome)
        
        # タスクの順序を入れ替え
        if random.random() < 0.5:
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        # リソースの再割り当て
        if random.random() < 0.5:
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
        print("Starting Genetic Algorithm...")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.num_generations}")
        print(f"Number of tasks: {self.num_tasks}")
        
        # 初期個体群の生成
        population = [self.create_individual() for _ in range(self.population_size)]
        
        best_fitness_history = []
        avg_fitness_history = []
        best_solution = None
        best_fitness = float('inf')
        
        for generation in range(self.num_generations):
            # 適応度の計算
            fitness_scores = [self.calculate_fitness(ind) for ind in population]
            
            # 最良個体の更新
            min_fitness_idx = np.argmin(fitness_scores)
            if fitness_scores[min_fitness_idx] < best_fitness:
                best_fitness = fitness_scores[min_fitness_idx]
                best_solution = copy.deepcopy(population[min_fitness_idx])
            
            # 統計情報の記録
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(np.mean(fitness_scores))
            
            if generation % 50 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness:.2f}, Avg fitness = {np.mean(fitness_scores):.2f}")
            
            # エリート保存
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
                offspring.extend([child1, child2])
            
            # 次世代の構成
            population = elite + offspring[:self.population_size - elite_size]
        
        print(f"\nOptimization completed!")
        print(f"Best makespan: {best_fitness:.2f} minutes")
        
        return best_solution, best_fitness, best_fitness_history, avg_fitness_history
    
    def visualize_schedule(self, chromosome):
        """ガントチャートの描画"""
        schedule = self.decode_chromosome(chromosome)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 作業者ごとのガントチャート
        worker_tasks = {}
        for task_id, info in schedule.items():
            worker = info['worker']
            if worker not in worker_tasks:
                worker_tasks[worker] = []
            worker_tasks[worker].append((task_id, info['start_time'], info['end_time']))
        
        y_pos = 0
        worker_labels = []
        for worker, tasks in sorted(worker_tasks.items()):
            for task_id, start, end in tasks:
                ax1.barh(y_pos, end - start, left=start, height=0.8, 
                        label=task_id if y_pos == 0 else "")
                ax1.text(start + (end - start) / 2, y_pos, task_id[-4:], 
                        ha='center', va='center', fontsize=8)
            worker_labels.append(worker)
            y_pos += 1
        
        ax1.set_yticks(range(len(worker_labels)))
        ax1.set_yticklabels(worker_labels)
        ax1.set_xlabel('Time (minutes)')
        ax1.set_title('Worker Schedule')
        ax1.grid(True, alpha=0.3)
        
        # 設備ごとのガントチャート
        equipment_tasks = {}
        for task_id, info in schedule.items():
            equipment = info['equipment']
            if equipment not in equipment_tasks:
                equipment_tasks[equipment] = []
            equipment_tasks[equipment].append((task_id, info['start_time'], info['end_time']))
        
        y_pos = 0
        equipment_labels = []
        for equipment, tasks in sorted(equipment_tasks.items()):
            for task_id, start, end in tasks:
                ax2.barh(y_pos, end - start, left=start, height=0.8)
                ax2.text(start + (end - start) / 2, y_pos, task_id[-4:], 
                        ha='center', va='center', fontsize=8)
            equipment_labels.append(equipment)
            y_pos += 1
        
        ax2.set_yticks(range(len(equipment_labels)))
        ax2.set_yticklabels(equipment_labels)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_title('Equipment Schedule')
        ax2.grid(True, alpha=0.3)
        
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
        ax.set_title('GA Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig

if __name__ == "__main__":
    # スケジューラの初期化と実行
    scheduler = JobShopGAScheduler('toy_dataset.xlsx')
    
    # パラメータの設定（オプション）
    scheduler.population_size = 50
    scheduler.num_generations = 300
    
    # GAの実行
    best_solution, best_fitness, best_history, avg_history = scheduler.run()
    
    # 結果の可視化
    gantt_fig = scheduler.visualize_schedule(best_solution)
    gantt_fig.savefig('schedule_gantt.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    convergence_fig = scheduler.plot_convergence(best_history, avg_history)
    convergence_fig.savefig('convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 最適スケジュールの詳細出力
    print("\n=== Optimal Schedule Details ===")
    schedule = scheduler.decode_chromosome(best_solution)
    for task_id in sorted(schedule.keys()):
        info = schedule[task_id]
        print(f"{task_id}: Start={info['start_time']:.1f}, End={info['end_time']:.1f}, "
              f"Worker={info['worker']}, Equipment={info['equipment']}")