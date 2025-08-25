#!/usr/bin/env python3
"""
分散遺伝的アルゴリズムによるJob Shop Schedulingの実行スクリプト
"""

import os
import sys

# 必要なライブラリのインストール確認
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import openpyxl
    import multiprocessing as mp
except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
    print("以下のコマンドでインストールしてください:")
    print("pip install pandas numpy matplotlib openpyxl")
    sys.exit(1)

from generate_toy_dataset import generate_toy_dataset
from distributed_ga_scheduler import DistributedGAScheduler

def main():
    """メイン実行関数"""
    
    print("="*70)
    print("Distributed Job Shop Scheduling with Genetic Algorithm")
    print("="*70)
    
    # Step 1: データセットの選択と生成
    print("\n[Step 1] Dataset Selection")
    print("1. Small (30 tasks, 7 workers, 23 equipments)")
    print("2. Medium (50 tasks, 10 workers, 30 equipments)")
    print("3. Large (100 tasks, 15 workers, 50 equipments)")
    print("4. Extra Large (200 tasks, 20 workers, 70 equipments)")
    print("5. Custom")
    print("6. Use existing file")
    
    choice = input("\nSelect dataset size (1-6, default=3): ") or "3"
    
    dataset_file = './large_dataset.xlsx'

    if choice == "1":
        print("\nGenerating small dataset...")
        dataset_file = 'toy_dataset.xlsx'
        data = generate_toy_dataset()
    elif choice == "2":
        print("\nGenerating medium dataset...")
        dataset_file = 'medium_dataset.xlsx'
        data = generate_toy_dataset(
            num_workers=10,
            num_equipments=30,
            num_tasks=50,
            num_patterns=25,
            constraint_density=0.25,
            output_file=dataset_file
        )
    elif choice == "3":
        print("\nGenerating large dataset...")
        dataset_file = 'large_dataset.xlsx'
        data = generate_toy_dataset(
            num_workers=15,
            num_equipments=50,
            num_tasks=100,
            num_patterns=40,
            constraint_density=0.2,
            output_file=dataset_file
        )
    elif choice == "4":
        print("\nGenerating extra large dataset...")
        dataset_file = 'extra_large_dataset.xlsx'
        data = generate_toy_dataset(
            num_workers=20,
            num_equipments=70,
            num_tasks=200,
            num_patterns=60,
            constraint_density=0.15,
            output_file=dataset_file
        )
    elif choice == "5":
        print("\n=== Custom Dataset Generation ===")
        num_workers = int(input("Number of workers (default=15): ") or 15)
        num_equipments = int(input("Number of equipments (default=50): ") or 50)
        num_tasks = int(input("Number of tasks (default=100): ") or 100)
        num_patterns = int(input("Number of allocation patterns (default=40): ") or 40)
        constraint_density = float(input("Constraint density 0-1 (default=0.2): ") or 0.2)
        dataset_file = input("Output filename (default=custom_distributed_dataset.xlsx): ") or 'custom_distributed_dataset.xlsx'
        
        data = generate_toy_dataset(
            num_workers=num_workers,
            num_equipments=num_equipments,
            num_tasks=num_tasks,
            num_patterns=num_patterns,
            constraint_density=constraint_density,
            output_file=dataset_file
        )
    elif choice == "6":
        dataset_file = input("Enter dataset filename (default=large_dataset.xlsx): ") or 'large_dataset.xlsx'
        if not os.path.exists(dataset_file):
            print(f"File {dataset_file} not found. Generating default large dataset...")
            data = generate_toy_dataset(
                num_workers=15,
                num_equipments=50,
                num_tasks=100,
                num_patterns=40,
                constraint_density=0.2,
                output_file='large_dataset.xlsx'
            )
            dataset_file = 'large_dataset.xlsx'
        else:
            print(f"✓ Using existing dataset: {dataset_file}")
    else:
        print("Invalid choice. Using default large dataset...")
        data = generate_toy_dataset(
            num_workers=15,
            num_equipments=50,
            num_tasks=100,
            num_patterns=40,
            constraint_density=0.2,
            output_file='large_dataset.xlsx'
        )
        dataset_file = 'large_dataset.xlsx'
    
    # Step 2: 分散GAスケジューラの初期化
    print(f"\n[Step 2] Initializing Distributed GA Scheduler with {dataset_file}...")
    
    # CPUコア数の取得と推奨島数の表示
    cpu_count = mp.cpu_count()
    recommended_islands = min(cpu_count, 8)  # 最大8島まで推奨
    print(f"  Available CPU cores: {cpu_count}")
    print(f"  Recommended number of islands: {recommended_islands}")
    
    try:
        num_islands = input(f"Number of islands (default={recommended_islands}): ")
        if num_islands:
            num_islands = int(num_islands)
        else:
            num_islands = recommended_islands
        
        if num_islands > cpu_count:
            print(f"Warning: Number of islands ({num_islands}) exceeds CPU cores ({cpu_count})")
            confirm = input("Continue anyway? (y/n): ")
            if confirm.lower() != 'y':
                num_islands = cpu_count
    except ValueError:
        print("Invalid input. Using recommended value.")
        num_islands = recommended_islands
    
    scheduler = DistributedGAScheduler(dataset_file, num_islands=num_islands)
    
    # Step 3: パラメータ設定（ユーザー入力）
    print(f"\n[Step 3] Distributed GA Parameters (press Enter for default values):")
    
    try:
        pop_size_per_island = input(f"Population size per island (default={scheduler.population_size_per_island}): ")
        if pop_size_per_island:
            scheduler.population_size_per_island = int(pop_size_per_island)
        
        num_gen = input(f"Number of generations (default={scheduler.num_generations}): ")
        if num_gen:
            scheduler.num_generations = int(num_gen)
        
        cross_rate = input(f"Crossover rate (default={scheduler.crossover_rate}): ")
        if cross_rate:
            scheduler.crossover_rate = float(cross_rate)
        
        mut_rate = input(f"Mutation rate (default={scheduler.mutation_rate}): ")
        if mut_rate:
            scheduler.mutation_rate = float(mut_rate)
        
        migration_interval = input(f"Migration interval in generations (default={scheduler.migration_interval}): ")
        if migration_interval:
            scheduler.migration_interval = int(migration_interval)
        
        migration_rate = input(f"Migration rate (default={scheduler.migration_rate}): ")
        if migration_rate:
            scheduler.migration_rate = float(migration_rate)
            
    except ValueError as e:
        print(f"Invalid input: {e}. Using default values.")
    
    total_population = scheduler.num_islands * scheduler.population_size_per_island
    
    print(f"\n✓ Parameters set:")
    print(f"  - Number of islands: {scheduler.num_islands}")
    print(f"  - Population per island: {scheduler.population_size_per_island}")
    print(f"  - Total population: {total_population}")
    print(f"  - Generations: {scheduler.num_generations}")
    print(f"  - Crossover rate: {scheduler.crossover_rate}")
    print(f"  - Mutation rate: {scheduler.mutation_rate}")
    print(f"  - Migration interval: {scheduler.migration_interval} generations")
    print(f"  - Migration rate: {scheduler.migration_rate}")
    
    # Step 4: 実行モードの選択
    print(f"\n[Step 4] Execution Mode Selection:")
    print("1. Distributed GA only")
    print("2. Performance comparison (Single vs Distributed)")
    
    mode = input("Select mode (1-2, default=1): ") or "1"
    
    if mode == "2":
        print("\n" + "="*60)
        print("Running performance comparison...")
        print("="*60)
        
        best_solution, best_fitness = scheduler.run_comparison()
        
    else:
        print("\n" + "="*60)
        print("Running distributed GA...")
        print("="*60)
        
        best_solution, best_fitness, island_results = scheduler.run_distributed()
        
        # 島ごとの結果表示
        print(f"\n{'='*60}")
        print("Island Results Summary")
        print(f"{'='*60}")
        for result in sorted(island_results, key=lambda x: x['island_id']):
            print(f"Island {result['island_id']}: Best fitness = {result['best_fitness']:.2f}")
    
    # Step 5: 結果の保存と可視化
    print(f"\n[Step 5] Visualizing results...")
    
    # スケジュールのデコード
    schedule = scheduler.decode_chromosome(best_solution)
    
    if schedule:
        # ガントチャートの生成と保存（通常のスケジューラのメソッドを流用）
        try:
            print("  - Creating Gantt chart...")
            # 簡易ガントチャート生成（matplotlib使用）
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # 作業者別の色マッピング
            workers = list(set(info['worker'] for info in schedule.values()))
            colors = plt.cm.Set3(np.linspace(0, 1, len(workers)))
            worker_colors = dict(zip(workers, colors))
            
            y_pos = 0
            y_labels = []
            
            for worker in workers:
                worker_tasks = [(task_id, info) for task_id, info in schedule.items() if info['worker'] == worker]
                worker_tasks.sort(key=lambda x: x[1]['start_time'])
                
                for task_id, info in worker_tasks:
                    duration = info['end_time'] - info['start_time']
                    ax.barh(y_pos, duration, left=info['start_time'], 
                           color=worker_colors[worker], alpha=0.7, edgecolor='black')
                    ax.text(info['start_time'] + duration/2, y_pos, 
                           f"T{task_id}", ha='center', va='center', fontsize=8)
                
                y_labels.append(f"Worker {worker}")
                y_pos += 1
            
            ax.set_yticks(range(len(workers)))
            ax.set_yticklabels(y_labels)
            ax.set_xlabel('Time')
            ax.set_ylabel('Workers')
            ax.set_title('Job Shop Schedule (Gantt Chart)')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('distributed_schedule_gantt.png', dpi=300, bbox_inches='tight')
            print("  ✓ Gantt chart saved as 'distributed_schedule_gantt.png'")
        except Exception as e:
            print(f"  ! Error creating Gantt chart: {e}")
        
        # スケジュールの詳細をCSVに保存
        print("  - Saving schedule details...")
        schedule_data = []
        for task_id in sorted(schedule.keys()):
            info = schedule[task_id]
            schedule_data.append({
                'Task ID': task_id,
                'Start Time': info['start_time'],
                'End Time': info['end_time'],
                'Duration': info['end_time'] - info['start_time'],
                'Worker': info['worker'],
                'Equipment': info['equipment']
            })
        
        schedule_df = pd.DataFrame(schedule_data)
        schedule_df.to_csv('distributed_optimal_schedule.csv', index=False)
        print("  ✓ Schedule details saved as 'distributed_optimal_schedule.csv'")
    
    # Step 6: サマリーの表示
    print("\n" + "="*70)
    print("DISTRIBUTED GA OPTIMIZATION RESULTS SUMMARY")
    print("="*70)
    print(f"Best Makespan: {best_fitness:.2f} minutes")
    if schedule:
        print(f"Number of tasks scheduled: {len(schedule)}")
        print(f"Total workers used: {len(set(s['worker'] for s in schedule.values()))}")
        print(f"Total equipment used: {len(set(s['equipment'] for s in schedule.values()))}")
    
    print(f"\nDistributed GA Configuration:")
    print(f"  - Number of islands: {scheduler.num_islands}")
    print(f"  - Total population: {total_population}")
    print(f"  - Generations: {scheduler.num_generations}")
    print(f"  - Migration every {scheduler.migration_interval} generations")
    
    print("\n" + "="*70)
    print("Optimization completed successfully!")
    print("Output files:")
    print("  - distributed_schedule_gantt.png: Gantt chart visualization")
    print("  - distributed_optimal_schedule.csv: Detailed schedule")
    print("="*70)
    
    # グラフの表示（オプション）
    show_plots = input("\nShow plots? (y/n): ")
    if show_plots.lower() == 'y':
        plt.show()

if __name__ == "__main__":
    # マルチプロセシング対応（Windows環境での警告を回避）
    mp.set_start_method('spawn', force=True)
    main()