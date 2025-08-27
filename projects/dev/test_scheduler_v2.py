#!/usr/bin/env python3
"""
修正版スケジューラのテストスクリプト
仕様変更に対応した実装の動作確認用
"""

import os
import sys
from ga_jobshop_scheduler_additional_v2 import JobShopGASchedulerAdditionalV2

def test_scheduler():
    """修正版スケジューラのテスト実行"""
    
    # データセットのパスを確認
    dataset_path = 'toy_dataset_additional.xlsx'
    
    # データセットが存在しない場合
    if not os.path.exists(dataset_path):
        print("Error: Dataset not found at", dataset_path)
        print("Please generate the dataset first using generate_toy_dataset_additional.py")
        return False
    
    print("="*60)
    print("Testing Updated GA Scheduler (V2)")
    print("="*60)
    print("\n[Applied Specification Changes:]")
    print("1. Continuous constraints: Only violated when tasks are directly consecutive")
    print("2. Makespan calculation: End time of latest regular task - 0")
    print("="*60)
    
    try:
        # スケジューラの初期化
        print("\nInitializing scheduler...")
        scheduler = JobShopGASchedulerAdditionalV2(dataset_path)
        
        # パラメータ設定（テスト用に小さめの値）
        scheduler.population_size = 30
        scheduler.num_generations = 100
        scheduler.mutation_rate = 0.3
        scheduler.crossover_rate = 0.8
        
        print(f"\nGA Parameters:")
        print(f"- Population size: {scheduler.population_size}")
        print(f"- Generations: {scheduler.num_generations}")
        print(f"- Mutation rate: {scheduler.mutation_rate}")
        print(f"- Crossover rate: {scheduler.crossover_rate}")
        
        # GAの実行
        print("\nRunning genetic algorithm...")
        best_solution, best_fitness, best_history, avg_history = scheduler.run()
        
        # 結果の検証
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        validation_results = scheduler.print_validation_report(best_solution)
        
        # スケジュールの詳細表示
        if validation_results['is_valid']:
            print("\n✅ Schedule is valid with all constraints satisfied!")
            print(f"Final Makespan: {validation_results['makespan']:.2f} minutes")
            
            # ガントチャートの保存
            print("\nGenerating Gantt chart...")
            gantt_fig = scheduler.visualize_schedule(best_solution)
            gantt_fig.savefig('test_schedule_gantt_v2.png', dpi=150, bbox_inches='tight')
            print("Gantt chart saved as: test_schedule_gantt_v2.png")
            
            # 収束曲線の保存
            print("\nGenerating convergence plot...")
            convergence_fig = scheduler.plot_convergence(best_history, avg_history)
            convergence_fig.savefig('test_convergence_v2.png', dpi=150, bbox_inches='tight')
            print("Convergence plot saved as: test_convergence_v2.png")
            
        else:
            print("\n⚠️  Schedule has constraint violations!")
            print("Please check the validation report above for details.")
        
        # 改善度の表示
        if len(best_history) > 0:
            initial_fitness = best_history[0]
            final_fitness = best_history[-1]
            improvement = (initial_fitness - final_fitness) / initial_fitness * 100
            print(f"\n📊 Optimization Statistics:")
            print(f"- Initial makespan: {initial_fitness:.2f} minutes")
            print(f"- Final makespan: {final_fitness:.2f} minutes")
            print(f"- Improvement: {improvement:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_scheduler()
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1)