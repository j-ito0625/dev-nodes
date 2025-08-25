import pandas as pd
import numpy as np
import random

def generate_toy_dataset(num_workers=7, num_equipments=23, num_tasks=30, num_patterns=20, 
                        constraint_density=0.3, output_file='toy_dataset.xlsx'):
    """
    設定に基づいたtoy datasetを生成
    
    Args:
        num_workers: 作業者数
        num_equipments: 設備数
        num_tasks: タスク数
        num_patterns: 割り当てパターン数
        constraint_density: 順序制約の密度 (0-1)
        output_file: 出力ファイル名
    """
    
    # 1. リソースタブの生成
    workers = [f"rsrc-{i:04d}" for i in range(1, num_workers + 1)]
    workers.append("dummy-type-0001")
    
    equipments = [f"rsrc-{i:04d}" for i in range(num_workers + 1, num_workers + num_equipments + 1)]
    equipments.append("dummy-type-0002")
    
    # リソースデータフレーム作成
    max_len = max(len(workers), len(equipments))
    workers_padded = workers + [None] * (max_len - len(workers))
    equipments_padded = equipments + [None] * (max_len - len(equipments))
    
    resources_df = pd.DataFrame({
        '作業者': workers_padded,
        '設備': equipments_padded
    })
    
    # 2. 割り当て情報タブの生成
    # 各割り当て可能パターンに対して、複数の作業者・設備の組み合わせを生成
    allocation_patterns = []
    pattern_names = []
    
    for i in range(num_patterns):  # 割り当てパターンを生成
        pattern_name = f"procedure_node_{i:05d}"
        # 各パターンに対して2-4個の組み合わせを生成
        num_combinations = random.randint(2, 4)
        for _ in range(num_combinations):
            worker = random.choice(workers[:-1])  # dummy以外から選択
            equipment = random.choice(equipments[:-1])  # dummy以外から選択
            allocation_patterns.append({
                '割り当て可能パターン': pattern_name,
                '作業者': worker,
                '設備': equipment
            })
        pattern_names.append(pattern_name)
    
    allocation_df = pd.DataFrame(allocation_patterns)
    
    # 3. タスクタブの生成
    tasks = []
    
    for i in range(num_tasks):
        task_id = f"task-{i:04d}"
        product_type = f"種目{(i % 3) + 1}"  # 3種類の品種目
        pattern = random.choice(pattern_names)
        duration = random.choice([10, 20, 30, 45, 60, 90, 120])  # 所要時間（分）
        
        tasks.append({
            'タスクID': task_id,
            '品種目': product_type,
            '割り当て可能パターン': pattern,
            '所要時間': duration
        })
    
    tasks_df = pd.DataFrame(tasks)
    
    # 4. 順序制約タブの生成
    order_constraints = []
    
    # タスク間の順序制約を生成（DAGを保証するため、前のタスクから後のタスクへのみ）
    max_constraints = int(num_tasks * (num_tasks - 1) / 2 * constraint_density)
    num_constraints_added = 0
    
    for i in range(num_tasks - 1):
        if num_constraints_added >= max_constraints:
            break
        # 各タスクから後続タスクへの制約を生成
        max_successors = min(3, num_tasks - i - 1, max_constraints - num_constraints_added)
        num_successors = random.randint(0, max_successors)
        if num_successors > 0:
            successors = random.sample(range(i + 1, num_tasks), num_successors)
            for successor in successors:
                predecessor_id = f"task-{i:04d}"
                successor_id = f"task-{successor:04d}"
                
                # 先行作業原点と後作業原点をランダムに選択
                predecessor_origin = random.choice(['終了', '開始'])
                successor_origin = random.choice(['終了', '開始'])
                
                # 時間差下限をランダムに生成（-180から120の範囲）
                time_diff_min = random.randint(-180, 120)
                
                order_constraints.append({
                    '先行作業ID': predecessor_id,
                    '後作業ID': successor_id,
                    '先行作業原点': predecessor_origin,
                    '後作業原点': successor_origin,
                    '時間差下限': time_diff_min
                })
                num_constraints_added += 1
    
    order_constraints_df = pd.DataFrame(order_constraints)
    
    # Excelファイルに保存
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        resources_df.to_excel(writer, sheet_name='リソース', index=False)
        tasks_df.to_excel(writer, sheet_name='task', index=False)
        allocation_df.to_excel(writer, sheet_name='割り当て情報', index=False)
        order_constraints_df.to_excel(writer, sheet_name='順序制約', index=False)
    
    print(f"Toy dataset generated successfully: {output_file}")
    print(f"\nDataset summary:")
    print(f"- Workers: {num_workers} (+ 1 dummy)")
    print(f"- Equipments: {num_equipments} (+ 1 dummy)")
    print(f"- Tasks: {num_tasks}")
    print(f"- Allocation patterns: {num_patterns}")
    print(f"- Order constraints: {len(order_constraints)}")
    print(f"- Constraint density: {len(order_constraints) / (num_tasks * (num_tasks - 1) / 2):.2%}")
    
    return {
        'resources': resources_df,
        'tasks': tasks_df,
        'allocation': allocation_df,
        'order_constraints': order_constraints_df
    }

if __name__ == "__main__":
    import sys
    
    # コマンドライン引数から設定を取得
    if len(sys.argv) > 1:
        if sys.argv[1] == '--large':
            # 大規模データセット
            data = generate_toy_dataset(
                num_workers=15,
                num_equipments=50,
                num_tasks=100,
                num_patterns=40,
                constraint_density=0.2,
                output_file='large_dataset.xlsx'
            )
        elif sys.argv[1] == '--medium':
            # 中規模データセット
            data = generate_toy_dataset(
                num_workers=10,
                num_equipments=30,
                num_tasks=50,
                num_patterns=25,
                constraint_density=0.25,
                output_file='medium_dataset.xlsx'
            )
        elif sys.argv[1] == '--help':
            print("Usage: python generate_toy_dataset.py [option]")
            print("Options:")
            print("  --small   : Small dataset (default)")
            print("  --medium  : Medium dataset (50 tasks)")
            print("  --large   : Large dataset (100 tasks)")
            print("  --custom  : Custom dataset (interactive)")
            sys.exit(0)
        elif sys.argv[1] == '--custom':
            # カスタム設定
            print("=== Custom Dataset Generation ===")
            num_workers = int(input("Number of workers (default=7): ") or 7)
            num_equipments = int(input("Number of equipments (default=23): ") or 23)
            num_tasks = int(input("Number of tasks (default=30): ") or 30)
            num_patterns = int(input("Number of allocation patterns (default=20): ") or 20)
            constraint_density = float(input("Constraint density 0-1 (default=0.3): ") or 0.3)
            output_file = input("Output filename (default=custom_dataset.xlsx): ") or 'custom_dataset.xlsx'
            
            data = generate_toy_dataset(
                num_workers=num_workers,
                num_equipments=num_equipments,
                num_tasks=num_tasks,
                num_patterns=num_patterns,
                constraint_density=constraint_density,
                output_file=output_file
            )
        else:
            # デフォルト（小規模）
            data = generate_toy_dataset()
    else:
        # デフォルト（小規模）
        data = generate_toy_dataset()
    
    # データの一部を表示
    print("\n--- Sample of Tasks ---")
    print(data['tasks'].head())
    print("\n--- Sample of Allocation Info ---")
    print(data['allocation'].head(10))
    print("\n--- Sample of Order Constraints ---")
    print(data['order_constraints'].head())