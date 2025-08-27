"""
追加制約対応のテスト用データセット生成（仕様変更版）
固定タスクは特定の作業者と設備の組み合わせにのみ発生
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_toy_dataset_additional(
    num_workers=7,
    num_equipments=23,
    num_tasks=30,
    num_fixed_tasks=5,
    num_patterns=20,
    num_continuous_constraints=10,
    constraint_density=0.3,
    output_file='toy_dataset_additional.xlsx'
):
    """
    追加制約対応のテスト用データセット生成（仕様変更版）
    
    Args:
        num_workers: 作業者数（ダミー除く）
        num_equipments: 設備数（ダミー除く）
        num_tasks: 通常タスク数
        num_fixed_tasks: 固定タスク数
        num_patterns: 割り当て可能パターン数
        num_continuous_constraints: 連続作業制限の数
        constraint_density: 順序制約の密度
        output_file: 出力ファイル名
    """
    
    random.seed(42)  # 再現性のため
    
    # 1. リソースの生成
    workers = [f'rsrc_{i:04d}' for i in range(num_workers)]
    workers.append('dummy-type-0001')  # ダミー作業者
    
    equipments = [f'rsrc_{i:04d}' for i in range(num_workers, num_workers + num_equipments)]
    equipments.append('dummy-type-0002')  # ダミー設備
    
    resources_data = []
    for worker in workers:
        resources_data.append({'作業者': worker, '設備': None})
    for equipment in equipments:
        resources_data.append({'作業者': None, '設備': equipment})
    
    resources_df = pd.DataFrame(resources_data)
    
    # 2. タスクの生成
    tasks = []
    for i in range(num_tasks):
        task = {
            'タスクID': f'task_{i:04d}',
            '所要時間': random.randint(30, 300),
            '割り当て可能パターン': f'procedure_node_{random.randint(0, num_patterns-1):05d}',
            '品種目': f'品種{(i % 3) + 1}'
        }
        tasks.append(task)
    tasks_df = pd.DataFrame(tasks)
    
    # 3. 固定タスクの生成（特定の作業者と設備に限定）
    # 固定タスクに使用する特定の作業者と設備を定義
    # 仕様の簡略化: rsrc_0005とrsrc_0046の組み合わせのみを使用
    FIXED_WORKER = 'rsrc_0005'
    FIXED_EQUIPMENT = 'rsrc_0046'
    
    # 作業者と設備が存在することを確認
    if FIXED_WORKER not in workers:
        FIXED_WORKER = workers[min(5, num_workers-1)]  # 5番目か最後の作業者
    if FIXED_EQUIPMENT not in equipments:
        FIXED_EQUIPMENT = equipments[min(5, num_equipments-1)]  # 5番目か最後の設備
    
    fixed_tasks = []
    # 時間帯を分散させて固定タスクを配置（全体のスケジュール期間を想定）
    base_time = 39600  # 基準時刻（例：朝9時を分で表現）
    
    for i in range(num_fixed_tasks):
        # 固定タスクの開始時刻を分散
        start_time = base_time + i * 500 + random.randint(-100, 100)
        duration = random.randint(30, 150)
        end_time = start_time + duration
        
        # 固定タスクは特定の作業者と設備の組み合わせのみ
        fixed_task = {
            'ID': f'fix_task_{i:02d}',
            '開始時刻': start_time,
            '終了時刻': end_time,
            'リソース名': f"('{FIXED_WORKER}','{FIXED_EQUIPMENT}')",  # タプル形式の文字列
            '品種名': f'品種{(i % 3) + 1}',
            '割り当て可能パターン': f'procedure_node_{random.randint(0, num_patterns-1):05d}'
        }
        fixed_tasks.append(fixed_task)
    
    fixed_tasks_df = pd.DataFrame(fixed_tasks)
    
    print(f"固定タスク設定: 作業者={FIXED_WORKER}, 設備={FIXED_EQUIPMENT}")
    print(f"固定タスク数: {num_fixed_tasks}")
    
    # 4. 割り当て情報の生成
    allocation_patterns = []
    for pattern_id in range(num_patterns):
        pattern_name = f'procedure_node_{pattern_id:05d}'
        # 各パターンに2-4の割り当て可能な組み合わせを生成
        num_allocations = random.randint(2, 4)
        for _ in range(num_allocations):
            worker = random.choice(workers[:-1])  # dummy以外から選択
            equipment = random.choice(equipments[:-1])  # dummy以外から選択
            allocation_patterns.append({
                '割り当て可能パターン': pattern_name,
                '作業者': worker,
                '設備': equipment
            })
    
    # 固定タスクのパターンも追加（特定の作業者と設備の組み合わせ）
    for _, fixed_task in fixed_tasks_df.iterrows():
        pattern = fixed_task['割り当て可能パターン']
        
        # この組み合わせが既に存在しない場合は追加
        exists = any(
            p['割り当て可能パターン'] == pattern and 
            p['作業者'] == FIXED_WORKER and 
            p['設備'] == FIXED_EQUIPMENT
            for p in allocation_patterns
        )
        if not exists:
            allocation_patterns.append({
                '割り当て可能パターン': pattern,
                '作業者': FIXED_WORKER,
                '設備': FIXED_EQUIPMENT
            })
    
    allocation_df = pd.DataFrame(allocation_patterns)
    
    # 5. 順序制約の生成
    order_constraints = []
    num_constraints = int(num_tasks * constraint_density)
    
    for _ in range(num_constraints):
        # ランダムに2つのタスクを選択
        task1_idx = random.randint(0, num_tasks - 1)
        task2_idx = random.randint(0, num_tasks - 1)
        
        if task1_idx != task2_idx:
            # 小さい方を先行タスクとする
            if task1_idx > task2_idx:
                task1_idx, task2_idx = task2_idx, task1_idx
            
            pred_task = f'task_{task1_idx:04d}'
            succ_task = f'task_{task2_idx:04d}'
            
            constraint = {
                '先行作業ID': pred_task,
                '後作業ID': succ_task,
                '先行作業原点': random.choice(['開始', '終了']),
                '後作業原点': random.choice(['開始', '終了']),
                '時間差下限': random.randint(0, 60)
            }
            order_constraints.append(constraint)
    
    # 固定タスクとの順序制約も追加（一部のタスクと固定タスクの間）
    for i in range(min(5, num_fixed_tasks)):
        if random.random() < 0.5:
            # 固定タスク→通常タスク
            constraint = {
                '先行作業ID': f'fix_task_{i:02d}',
                '後作業ID': f'task_{random.randint(0, num_tasks-1):04d}',
                '先行作業原点': '終了',
                '後作業原点': '開始',
                '時間差下限': random.randint(10, 30)
            }
        else:
            # 通常タスク→固定タスク
            constraint = {
                '先行作業ID': f'task_{random.randint(0, num_tasks-1):04d}',
                '後作業ID': f'fix_task_{i:02d}',
                '先行作業原点': '終了',
                '後作業原点': '開始',
                '時間差下限': random.randint(10, 30)
            }
        order_constraints.append(constraint)
    
    order_constraints_df = pd.DataFrame(order_constraints)
    
    # 6. 連続作業制限の生成（特定の設備に対してのみ）
    continuous_constraints = []
    
    # 固定タスクが使用する設備を重点的に制限対象にする
    constraint_equipments = [FIXED_EQUIPMENT]  # 固定タスクの設備を必ず含める
    
    # 他の設備もランダムに追加
    other_equipments = [e for e in equipments[:-1] if e != FIXED_EQUIPMENT]
    constraint_equipments.extend(random.sample(other_equipments, 
                                               min(3, len(other_equipments))))
    
    for _ in range(num_continuous_constraints):
        equipment = random.choice(constraint_equipments)
        
        # ランダムに2つのパターンを選択
        pattern1 = f'procedure_node_{random.randint(0, num_patterns-1):05d}'
        pattern2 = f'procedure_node_{random.randint(0, num_patterns-1):05d}'
        
        if pattern1 != pattern2:
            continuous_constraints.append({
                '先行パターン': pattern1,
                '後パターン': pattern2,
                '設備': equipment
            })
    
    continuous_constraints_df = pd.DataFrame(continuous_constraints)
    
    # 7. データの保存
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        resources_df.to_excel(writer, sheet_name='リソース', index=False)
        tasks_df.to_excel(writer, sheet_name='task', index=False)
        fixed_tasks_df.to_excel(writer, sheet_name='固定タスク', index=False)
        allocation_df.to_excel(writer, sheet_name='割り当て情報', index=False)
        order_constraints_df.to_excel(writer, sheet_name='順序制約', index=False)
        continuous_constraints_df.to_excel(writer, sheet_name='連続作業制限', index=False)
    
    print(f"データセットが {output_file} に保存されました")
    print(f"Summary:")
    print(f"  - Workers: {num_workers} + 1 dummy")
    print(f"  - Equipments: {num_equipments} + 1 dummy")
    print(f"  - Regular tasks: {num_tasks}")
    print(f"  - Fixed tasks: {num_fixed_tasks} (worker: {FIXED_WORKER}, equipment: {FIXED_EQUIPMENT})")
    print(f"  - Patterns: {num_patterns}")
    print(f"  - Order constraints: {len(order_constraints)}")
    print(f"  - Continuous work restrictions: {len(continuous_constraints)}")
    print(f"  - Restricted equipments: {constraint_equipments}")
    
    return {
        'resources': resources_df,
        'tasks': tasks_df,
        'fixed_tasks': fixed_tasks_df,
        'allocation': allocation_df,
        'order_constraints': order_constraints_df,
        'continuous_constraints': continuous_constraints_df
    }

if __name__ == "__main__":
    # デフォルト設定で生成
    data = generate_toy_dataset_additional()
    
    print("\n固定タスクの詳細:")
    print(data['fixed_tasks'][['ID', '開始時刻', '終了時刻', 'リソース名']].to_string())