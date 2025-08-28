#!/usr/bin/env python
"""
並列版GAスケジューラの実行スクリプト
V3アルゴリズムの並列処理版と通常版の比較実行
"""

import os
import sys
import time
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# 並列版と通常版のインポート
from ga_jobshop_scheduler_additional_v3_parallel import JobShopGASchedulerAdditionalV3Parallel
from ga_jobshop_scheduler_additional_v3 import JobShopGASchedulerAdditionalV3


def run_comparison(dataset_path, num_islands=4, generations=100):
    """
    通常版と並列版の性能比較を実行
    
    Args:
        dataset_path: データセットのパス
        num_islands: 並列処理の島数
        generations: 世代数
    """
    print("\n" + "="*70)
    print("JOB SHOP SCHEDULING GA - PERFORMANCE COMPARISON")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Generations: {generations}")
    print(f"Number of islands (parallel): {num_islands}")
    print("="*70 + "\n")
    
    results_comparison = {}
    
    # 1. 通常版（V3）の実行
    print("\n[1] Running Standard GA V3 (Single Process)...")
    print("-" * 50)
    
    scheduler_v3 = JobShopGASchedulerAdditionalV3(dataset_path)
    scheduler_v3.num_generations = generations
    
    start_time = time.time()
    best_solution_v3, best_fitness_v3, best_history_v3, avg_history_v3 = scheduler_v3.run()
    elapsed_v3 = time.time() - start_time
    
    results_comparison['standard'] = {
        'fitness': best_fitness_v3,
        'time': elapsed_v3,
        'solution': best_solution_v3
    }
    
    print(f"\n✓ Standard V3 completed!")
    print(f"  Best fitness: {best_fitness_v3:.2f}")
    print(f"  Elapsed time: {elapsed_v3:.2f} seconds")
    
    # 2. 並列版の実行
    print("\n[2] Running Parallel GA V3 (Island Model)...")
    print("-" * 50)
    
    scheduler_parallel = JobShopGASchedulerAdditionalV3Parallel(dataset_path, num_islands=num_islands)
    scheduler_parallel.num_generations = generations
    
    start_time = time.time()
    best_solution_parallel, best_fitness_parallel, best_history_parallel, avg_history_parallel = scheduler_parallel.run()
    elapsed_parallel = time.time() - start_time
    
    results_comparison['parallel'] = {
        'fitness': best_fitness_parallel,
        'time': elapsed_parallel,
        'solution': best_solution_parallel
    }
    
    print(f"\n✓ Parallel V3 completed!")
    print(f"  Best fitness: {best_fitness_parallel:.2f}")
    print(f"  Elapsed time: {elapsed_parallel:.2f} seconds")
    
    # 3. 性能比較結果の表示
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON RESULTS")
    print("="*70)
    
    speedup = elapsed_v3 / elapsed_parallel
    efficiency = speedup / num_islands
    fitness_improvement = ((best_fitness_v3 - best_fitness_parallel) / best_fitness_v3) * 100
    
    print(f"\n[Execution Time]")
    print(f"  Standard V3: {elapsed_v3:.2f} seconds")
    print(f"  Parallel V3: {elapsed_parallel:.2f} seconds")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Parallel Efficiency: {efficiency:.1%}")
    
    print(f"\n[Solution Quality]")
    print(f"  Standard V3 fitness: {best_fitness_v3:.2f}")
    print(f"  Parallel V3 fitness: {best_fitness_parallel:.2f}")
    if fitness_improvement > 0:
        print(f"  Improvement: {fitness_improvement:.1f}% better")
    elif fitness_improvement < 0:
        print(f"  Difference: {abs(fitness_improvement):.1f}% worse")
    else:
        print(f"  Identical fitness values")
    
    print("\n" + "="*70 + "\n")
    
    return results_comparison, (scheduler_v3, scheduler_parallel)


def run_parallel_only(dataset_path, num_islands=4, generations=500):
    """
    並列版のみを実行（高速実行用）
    
    Args:
        dataset_path: データセットのパス
        num_islands: 並列処理の島数
        generations: 世代数
    """
    print("\n" + "="*70)
    print("JOB SHOP SCHEDULING - PARALLEL GA V3 EXECUTION")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Generations: {generations}")
    print(f"Number of islands: {num_islands}")
    print(f"CPUs available: {os.cpu_count()}")
    print("="*70 + "\n")
    
    # スケジューラの初期化
    scheduler = JobShopGASchedulerAdditionalV3Parallel(dataset_path, num_islands=num_islands)
    scheduler.num_generations = generations
    
    # 実行
    print("Starting parallel GA optimization...")
    start_time = time.time()
    best_solution, best_fitness, best_history, avg_history = scheduler.run()
    elapsed_time = time.time() - start_time
    
    print(f"\n✓ Optimization completed!")
    print(f"  Best fitness (makespan): {best_fitness:.2f} minutes")
    print(f"  Total elapsed time: {elapsed_time:.2f} seconds")
    print(f"  Average time per generation: {elapsed_time/generations:.3f} seconds")
    
    # 検証レポート
    print("\n" + "-"*70)
    print("VALIDATION REPORT")
    print("-"*70)
    validation_results = scheduler.print_validation_report(best_solution)
    
    # 可視化
    print("\n" + "-"*70)
    print("GENERATING VISUALIZATIONS")
    print("-"*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ガントチャート
    gantt_fig = scheduler.visualize_schedule(best_solution)
    gantt_filename = f'gantt_parallel_{timestamp}.png'
    gantt_fig.savefig(gantt_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Gantt chart saved: {gantt_filename}")
    
    # 収束曲線
    convergence_fig = scheduler.plot_convergence(best_history, avg_history)
    convergence_filename = f'convergence_parallel_{timestamp}.png'
    convergence_fig.savefig(convergence_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Convergence plot saved: {convergence_filename}")
    
    return scheduler, best_solution, best_fitness


def main():
    """
    メイン実行関数
    """
    parser = argparse.ArgumentParser(description='Run Parallel GA Scheduler for Job Shop Problem')
    parser.add_argument('--dataset', type=str, default='toy_dataset_additional.xlsx',
                       help='Path to dataset Excel file')
    parser.add_argument('--islands', type=int, default=4,
                       help='Number of islands for parallel processing')
    parser.add_argument('--generations', type=int, default=500,
                       help='Number of generations')
    parser.add_argument('--mode', type=str, choices=['parallel', 'compare'], default='parallel',
                       help='Execution mode: parallel only or comparison')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # データセットの存在確認
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset '{args.dataset}' not found!")
        print("Please ensure the dataset exists or generate it first.")
        sys.exit(1)
    
    # 実行モードに応じて処理
    if args.mode == 'compare':
        # 比較モード
        results, schedulers = run_comparison(
            args.dataset, 
            num_islands=args.islands,
            generations=min(args.generations, 100)  # 比較時は世代数を制限
        )
        
        # 並列版の詳細検証
        scheduler_parallel = schedulers[1]
        if results['parallel']['solution']:
            print("\nDetailed validation for parallel solution:")
            scheduler_parallel.print_validation_report(results['parallel']['solution'])
            
            if not args.no_viz:
                # 可視化
                gantt_fig = scheduler_parallel.visualize_schedule(results['parallel']['solution'])
                gantt_fig.savefig('gantt_comparison_parallel.png', dpi=300, bbox_inches='tight')
                print("✓ Gantt chart saved: gantt_comparison_parallel.png")
    
    else:
        # 並列実行のみ
        scheduler, best_solution, best_fitness = run_parallel_only(
            args.dataset,
            num_islands=args.islands,
            generations=args.generations
        )
    
    print("\n" + "="*70)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")
    
    if not args.no_viz:
        plt.show()


if __name__ == "__main__":
    # マルチプロセシング用の設定
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    main()