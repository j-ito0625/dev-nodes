#!/usr/bin/env python3
"""
Job Shop Schedulingの実行スクリプト
"""

import os
import sys

# 必要なライブラリのインストール確認
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import openpyxl
except ImportError as e:
    print(f"必要なライブラリがインストールされていません: {e}")
    print("以下のコマンドでインストールしてください:")
    print("pip install pandas numpy matplotlib openpyxl")
    sys.exit(1)

from generate_toy_dataset import generate_toy_dataset
from ga_jobshop_scheduler import JobShopGAScheduler

def main():
    """メイン実行関数"""
    
    print("="*60)
    print("Job Shop Scheduling with Genetic Algorithm")
    print("="*60)
    
    # Step 1: データセットの選択と生成
    print("\n[Step 1] Dataset Selection")
    print("1. Small (30 tasks, 7 workers, 23 equipments)")
    print("2. Medium (50 tasks, 10 workers, 30 equipments)")
    print("3. Large (100 tasks, 15 workers, 50 equipments)")
    print("4. Custom")
    print("5. Use existing file")
    
    choice = input("\nSelect dataset size (1-5, default=1): ") or "1"
    
    # dataset_file = 'toy_dataset.xlsx'
    dataset_file = './large_dataset.xlsx'

    if choice == "1":
        print("\nGenerating small dataset...")
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
        print("\n=== Custom Dataset Generation ===")
        num_workers = int(input("Number of workers (default=7): ") or 7)
        num_equipments = int(input("Number of equipments (default=23): ") or 23)
        num_tasks = int(input("Number of tasks (default=30): ") or 30)
        num_patterns = int(input("Number of allocation patterns (default=20): ") or 20)
        constraint_density = float(input("Constraint density 0-1 (default=0.3): ") or 0.3)
        dataset_file = input("Output filename (default=custom_dataset.xlsx): ") or 'custom_dataset.xlsx'
        
        data = generate_toy_dataset(
            num_workers=num_workers,
            num_equipments=num_equipments,
            num_tasks=num_tasks,
            num_patterns=num_patterns,
            constraint_density=constraint_density,
            output_file=dataset_file
        )
    elif choice == "5":
        dataset_file = input("Enter dataset filename (default=toy_dataset.xlsx): ") or 'toy_dataset.xlsx'
        if not os.path.exists(dataset_file):
            print(f"File {dataset_file} not found. Generating default dataset...")
            data = generate_toy_dataset()
            dataset_file = 'toy_dataset.xlsx'
        else:
            print(f"✓ Using existing dataset: {dataset_file}")
    else:
        print("Invalid choice. Using default small dataset...")
        data = generate_toy_dataset()
    
    # Step 2: GAスケジューラの初期化
    print(f"\n[Step 2] Initializing GA Scheduler with {dataset_file}...")
    scheduler = JobShopGAScheduler(dataset_file)
    
    # パラメータ設定（ユーザー入力）
    print("\n[Step 3] GA Parameters (press Enter for default values):")
    
    try:
        pop_size = input(f"Population size (default={scheduler.population_size}): ")
        if pop_size:
            scheduler.population_size = int(pop_size)
        
        num_gen = input(f"Number of generations (default={scheduler.num_generations}): ")
        if num_gen:
            scheduler.num_generations = int(num_gen)
        
        cross_rate = input(f"Crossover rate (default={scheduler.crossover_rate}): ")
        if cross_rate:
            scheduler.crossover_rate = float(cross_rate)
        
        mut_rate = input(f"Mutation rate (default={scheduler.mutation_rate}): ")
        if mut_rate:
            scheduler.mutation_rate = float(mut_rate)
    except ValueError as e:
        print(f"Invalid input: {e}. Using default values.")
    
    print(f"\n✓ Parameters set:")
    print(f"  - Population size: {scheduler.population_size}")
    print(f"  - Generations: {scheduler.num_generations}")
    print(f"  - Crossover rate: {scheduler.crossover_rate}")
    print(f"  - Mutation rate: {scheduler.mutation_rate}")
    
    # Step 4: GAの実行
    print("\n[Step 4] Running Genetic Algorithm...")
    print("-"*40)
    best_solution, best_fitness, best_history, avg_history = scheduler.run()
    print("-"*40)
    
    # Step 5: 結果の保存と可視化
    print("\n[Step 5] Visualizing results...")
    
    # ガントチャートの生成と保存
    print("  - Creating Gantt chart...")
    gantt_fig = scheduler.visualize_schedule(best_solution)
    gantt_fig.savefig('schedule_gantt.png', dpi=300, bbox_inches='tight')
    print("  ✓ Gantt chart saved as 'schedule_gantt.png'")
    
    # 収束曲線の生成と保存
    print("  - Creating convergence plot...")
    convergence_fig = scheduler.plot_convergence(best_history, avg_history)
    convergence_fig.savefig('convergence.png', dpi=300, bbox_inches='tight')
    print("  ✓ Convergence plot saved as 'convergence.png'")
    
    # スケジュールの詳細をCSVに保存
    print("  - Saving schedule details...")
    schedule = scheduler.decode_chromosome(best_solution)
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
    schedule_df.to_csv('optimal_schedule.csv', index=False)
    print("  ✓ Schedule details saved as 'optimal_schedule.csv'")
    
    # Step 6: サマリーの表示
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*60)
    print(f"Best Makespan: {best_fitness:.2f} minutes")
    print(f"Number of tasks scheduled: {len(schedule)}")
    print(f"Total workers used: {len(set(s['worker'] for s in schedule.values()))}")
    print(f"Total equipment used: {len(set(s['equipment'] for s in schedule.values()))}")
    
    # 詳細な検証レポートの表示
    validation_results = scheduler.print_validation_report(best_solution)
    
    print("\n" + "="*60)
    print("Optimization completed successfully!")
    print("Output files:")
    print("  - schedule_gantt.png: Gantt chart visualization")
    print("  - convergence.png: GA convergence plot")
    print("  - optimal_schedule.csv: Detailed schedule")
    print("="*60)
    
    # グラフの表示（オプション）
    show_plots = input("\nShow plots? (y/n): ")
    if show_plots.lower() == 'y':
        plt.show()

if __name__ == "__main__":
    main()