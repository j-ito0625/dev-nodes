#!/usr/bin/env python3
"""
追加制約対応版Job Shop Schedulingの実行スクリプト（仕様変更版V2）
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

from generate_toy_dataset_additional_v2 import generate_toy_dataset_additional
from ga_jobshop_scheduler_additional_v2 import JobShopGASchedulerAdditionalV2

def main():
    """メイン実行関数"""
    
    print("="*60)
    print("Job Shop Scheduling with Additional Constraints (V2)")
    print("(Updated Specifications for Fixed Tasks & Continuous Work)")
    print("="*60)
    print("\n[Specification Updates in V2:]")
    print("1. Makespan = End time of latest regular task - 0")
    print("2. Continuous constraints: Only violated when directly consecutive")
    print("="*60)
    
    # Step 1: データセットの選択と生成
    print("\n[Step 1] Dataset Selection")
    print("1. Small (30 tasks, 5 fixed, 7 workers, 23 equipments)")
    print("2. Medium (50 tasks, 8 fixed, 10 workers, 30 equipments)")
    print("3. Large (100 tasks, 15 fixed, 15 workers, 50 equipments)")
    print("4. Custom")
    print("5. Use existing file")
    
    choice = input("\nSelect dataset size (1-5, default=1): ") or "1"
    
    dataset_file = 'toy_dataset_additional.xlsx'

    if choice == "1":
        print("\nGenerating small dataset with additional constraints...")
        data = generate_toy_dataset_additional()
    elif choice == "2":
        print("\nGenerating medium dataset with additional constraints...")
        dataset_file = 'medium_dataset_additional.xlsx'
        data = generate_toy_dataset_additional(
            num_workers=10,
            num_equipments=30,
            num_tasks=50,
            num_fixed_tasks=8,
            num_patterns=25,
            num_continuous_constraints=15,
            constraint_density=0.25,
            output_file=dataset_file
        )
    elif choice == "3":
        print("\nGenerating large dataset with additional constraints...")
        dataset_file = 'large_dataset_additional.xlsx'
        data = generate_toy_dataset_additional(
            num_workers=15,
            num_equipments=50,
            num_tasks=100,
            num_fixed_tasks=15,
            num_patterns=40,
            num_continuous_constraints=25,
            constraint_density=0.2,
            output_file=dataset_file
        )
    elif choice == "4":
        print("\n=== Custom Dataset Generation ===")
        num_workers = int(input("Number of workers (default=7): ") or 7)
        num_equipments = int(input("Number of equipments (default=23): ") or 23)
        num_tasks = int(input("Number of regular tasks (default=30): ") or 30)
        num_fixed_tasks = int(input("Number of fixed tasks (default=5): ") or 5)
        num_patterns = int(input("Number of allocation patterns (default=20): ") or 20)
        num_continuous = int(input("Number of continuous work restrictions (default=10): ") or 10)
        constraint_density = float(input("Constraint density 0-1 (default=0.3): ") or 0.3)
        dataset_file = input("Output filename (default=custom_dataset_additional.xlsx): ") or 'custom_dataset_additional.xlsx'
        
        data = generate_toy_dataset_additional(
            num_workers=num_workers,
            num_equipments=num_equipments,
            num_tasks=num_tasks,
            num_fixed_tasks=num_fixed_tasks,
            num_patterns=num_patterns,
            num_continuous_constraints=num_continuous,
            constraint_density=constraint_density,
            output_file=dataset_file
        )
    elif choice == "5":
        dataset_file = input("Enter dataset filename (default=toy_dataset_additional.xlsx): ") or 'toy_dataset_additional.xlsx'
        if not os.path.exists(dataset_file):
            print(f"File {dataset_file} not found. Generating default dataset...")
            data = generate_toy_dataset_additional()
            dataset_file = 'toy_dataset_additional.xlsx'
        else:
            print(f"✓ Using existing dataset: {dataset_file}")
    else:
        print("Invalid choice. Using default small dataset...")
        data = generate_toy_dataset_additional()
    
    # Step 2: GAスケジューラの初期化
    print(f"\n[Step 2] Initializing GA Scheduler V2 with {dataset_file}...")
    scheduler = JobShopGASchedulerAdditionalV2(dataset_file)
    
    # パラメータ設定（ユーザー入力）
    print("\n[Step 3] GA Parameters (press Enter for default values):")
    print("Note: V2 uses updated constraint handling for better convergence")
    
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
    print("\n[Step 4] Running Genetic Algorithm V2...")
    print("-"*40)
    best_solution, best_fitness, best_history, avg_history = scheduler.run()
    print("-"*40)
    
    # Step 5: 結果の保存と可視化
    print("\n[Step 5] Visualizing results...")
    
    # ガントチャートの生成と保存
    print("  - Creating Gantt chart with fixed tasks highlighted...")
    gantt_fig = scheduler.visualize_schedule(best_solution)
    gantt_filename = 'schedule_gantt_v2.png'
    gantt_fig.savefig(gantt_filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Gantt chart saved as '{gantt_filename}'")
    
    # 収束曲線の生成と保存
    print("  - Creating convergence plot...")
    convergence_fig = scheduler.plot_convergence(best_history, avg_history)
    convergence_filename = 'convergence_v2.png'
    convergence_fig.savefig(convergence_filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Convergence plot saved as '{convergence_filename}'")
    
    # スケジュールの詳細をCSVに保存
    print("  - Saving schedule details...")
    schedule = scheduler.decode_chromosome(best_solution)
    schedule_data = []
    for task_id in sorted(schedule.keys()):
        info = schedule[task_id]
        is_fixed = task_id in scheduler.fixed_tasks
        schedule_data.append({
            'Task ID': task_id,
            'Task Type': 'Fixed' if is_fixed else 'Regular',
            'Start Time': info['start_time'],
            'End Time': info['end_time'],
            'Duration': info['end_time'] - info['start_time'],
            'Worker': info['worker'],
            'Equipment': info['equipment']
        })
    
    schedule_df = pd.DataFrame(schedule_data)
    csv_filename = 'optimal_schedule_v2.csv'
    schedule_df.to_csv(csv_filename, index=False)
    print(f"  ✓ Schedule details saved as '{csv_filename}'")
    
    # Step 6: サマリーの表示
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS SUMMARY (V2 - Updated Specifications)")
    print("="*60)
    
    # V2の新しいmakespan計算
    start_point = 0  # 固定で0
    regular_end_times = [s['end_time'] for tid, s in schedule.items() if tid not in scheduler.fixed_tasks]
    end_point = max(regular_end_times) if regular_end_times else 0
    makespan = end_point - start_point
    
    print(f"Best Makespan (V2 calculation): {makespan:.2f} minutes")
    print(f"  - Start point (fixed): {start_point:.2f} minutes")
    print(f"  - End point (latest regular task): {end_point:.2f} minutes")
    print(f"  - Formula: Makespan = {end_point:.2f} - {start_point:.2f} = {makespan:.2f}")
    
    print(f"\nTask Statistics:")
    print(f"  - Regular tasks scheduled: {len([t for t in schedule if t not in scheduler.fixed_tasks])}")
    print(f"  - Fixed tasks integrated: {len(scheduler.fixed_tasks)}")
    print(f"  - Total tasks: {len(schedule)}")
    
    print(f"\nResource Usage:")
    print(f"  - Workers used: {len(set(s['worker'] for s in schedule.values()))}")
    print(f"  - Equipment used: {len(set(s['equipment'] for s in schedule.values()))}")
    
    # 連続作業制限の統計
    if scheduler.continuous_constraints:
        total_continuous_constraints = sum(len(v) for v in scheduler.continuous_constraints.values())
        print(f"\nConstraints:")
        print(f"  - Order constraints: {len(scheduler.order_constraints)}")
        print(f"  - Continuous work restrictions: {total_continuous_constraints}")
    
    # 詳細な検証レポートの表示
    print("\n" + "="*60)
    print("SCHEDULE VALIDATION")
    print("="*60)
    validation_results = scheduler.print_validation_report(best_solution)
    
    # 改善度の表示
    if len(best_history) > 1:
        initial_fitness = best_history[0]
        final_fitness = best_history[-1]
        improvement = (initial_fitness - final_fitness) / initial_fitness * 100 if initial_fitness > 0 else 0
        print("\n📊 Optimization Statistics:")
        print(f"  - Initial makespan: {initial_fitness:.2f} minutes")
        print(f"  - Final makespan: {final_fitness:.2f} minutes")
        print(f"  - Improvement: {improvement:.1f}%")
    
    print("\n" + "="*60)
    print("Optimization completed successfully!")
    print("\nOutput files:")
    print(f"  - {gantt_filename}: Gantt chart visualization")
    print(f"  - {convergence_filename}: GA convergence plot")
    print(f"  - {csv_filename}: Detailed schedule in CSV format")
    print("="*60)
    
    # グラフの表示（オプション）
    show_plots = input("\nShow plots? (y/n): ")
    if show_plots.lower() == 'y':
        plt.show()

if __name__ == "__main__":
    main()