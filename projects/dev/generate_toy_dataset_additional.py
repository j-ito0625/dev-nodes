#!/usr/bin/env python3
"""
追加制約付きのToy Dataset生成スクリプト
固定タスクと連続作業制限を含むデータセットを生成
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

def generate_toy_dataset_additional(
    num_workers=7,
    num_equipments=23,
    num_tasks=30,
    num_patterns=20,
    num_fixed_tasks=5,
    num_continuous_constraints=10,
    constraint_density=0.3,
    output_file='toy_dataset_additional.xlsx'
):
    """
    追加制約付きのToy Datasetを生成する
    
    Args:
        num_workers: 作業者数（dummy除く）
        num_equipments: 設備数（dummy除く）
        num_tasks: 通常タスク数
        num_patterns: 割り当て可能パターン数
        num_fixed_tasks: 固定タスク数
        num_continuous_constraints: 連続作業制限数
        constraint_density: 順序制約の密度（0-1）
        output_file: 出力ファイル名
    """
    
    # 1. リソースの生成
    workers = [f"rsrc-{i:04d}" for i in range(1, num_workers + 1)]
    equipments = [f"rsrc-{i:04d}" for i in range(num_workers + 1, num_workers + num_equipments + 1)]
    
    # dummy リソースを追加
    workers.append("dummy-type-0001")
    equipments.append("dummy-type-0002")
    
    resources_data = {
        '作業者': workers + [None] * len(equipments),
        '設備': [None] * len(workers) + equipments
    }
    resources_df = pd.DataFrame(resources_data)
    
    # 2. 通常タスクの生成
    tasks = []
    for i in range(num_tasks):
        task = {
            'タスクID': f'task-{i:04d}',
            '品種目': f'品種{(i % 3) + 1}',
            '割り当て可能パターン': f'procedure_node_{i % num_patterns:05d}',
            '所要時間': random.randint(10, 180)  # 10-180分
        }
        tasks.append(task)
    tasks_df = pd.DataFrame(tasks)
    
    # 3. 固定タスクの生成
    fixed_tasks = []
    # 時間帯を分散させて固定タスクを配置（全体のスケジュール期間を想定）
    base_time = 39600  # 基準時刻（例：朝9時を分で表現）
    
    for i in range(num_fixed_tasks):
        # 固定タスクの開始時刻を分散
        start_time = base_time + i * 500 + random.randint(-100, 100)
        duration = random.randint(30, 150)
        end_time = start_time + duration
        
        # ランダムに作業者と設備を選択
        worker = random.choice(workers[:-1])  # dummy以外
        equipment = random.choice(equipments[:-1])  # dummy以外
        
        fixed_task = {
            'ID': f'fix_task_{i:02d}',
            '開始時刻': start_time,
            '終了時刻': end_time,
            'リソース名': f"('{worker}','{equipment}')",  # タプル形式の文字列
            '品種名': f'品種{(i % 3) + 1}',
            '割り当て可能パターン': f'procedure_node_{random.randint(0, num_patterns-1):05d}'
        }
        fixed_tasks.append(fixed_task)
    
    fixed_tasks_df = pd.DataFrame(fixed_tasks)
    
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
    
    # 固定タスクのパターンも追加
    for _, fixed_task in fixed_tasks_df.iterrows():
        pattern = fixed_task['割り当て可能パターン']
        # リソース名をパース
        resource_str = fixed_task['リソース名']
        worker = resource_str.split("'")[1]
        equipment = resource_str.split("'")[3]
        
        # この組み合わせが既に存在しない場合は追加
        exists = any(
            p['割り当て可能パターン'] == pattern and 
            p['作業者'] == worker and 
            p['設備'] == equipment
            for p in allocation_patterns
        )
        if not exists:
            allocation_patterns.append({
                '割り当て可能パターン': pattern,
                '作業者': worker,
                '設備': equipment
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
            
            pred_task = f'task-{task1_idx:04d}'
            succ_task = f'task-{task2_idx:04d}'
            
            # ランダムに制約タイプを決定
            pred_origin = random.choice(['開始', '終了'])
            succ_origin = random.choice(['開始', '終了'])
            
            # 時間差下限をランダムに設定（負の値も許可）
            time_diff_min = random.randint(-180, 180)
            
            constraint = {
                '先行作業ID': pred_task,
                '後作業ID': succ_task,
                '先行作業原点': pred_origin,
                '後作業原点': succ_origin,
                '時間差下限': time_diff_min
            }
            order_constraints.append(constraint)
    
    # 固定タスクに関連する制約も追加（固定タスクを先行タスクとする）
    for i in range(min(3, num_fixed_tasks)):  # 最大3つの固定タスク制約
        fixed_task_id = f'fix_task_{i:02d}'
        related_task = f'task-{random.randint(0, num_tasks-1):04d}'
        
        constraint = {
            '先行作業ID': fixed_task_id,
            '後作業ID': related_task,
            '先行作業原点': '終了',
            '後作業原点': '開始',
            '時間差下限': random.randint(0, 60)  # 固定タスク後に開始
        }
        order_constraints.append(constraint)
    
    order_constraints_df = pd.DataFrame(order_constraints)
    
    # 6. 連続作業制限の生成
    continuous_constraints = []
    used_patterns = list(set(p['割り当て可能パターン'] for p in allocation_patterns))
    
    for _ in range(num_continuous_constraints):
        # ランダムに2つのパターンと1つの設備を選択
        pattern1 = random.choice(used_patterns)
        pattern2 = random.choice(used_patterns)
        equipment = random.choice(equipments[:-1])  # dummy以外
        
        # 同じパターンの連続も制限対象とすることがある
        continuous_constraint = {
            '先行パターン': pattern1,
            '後パターン': pattern2,
            '設備': equipment
        }
        continuous_constraints.append(continuous_constraint)
    
    continuous_constraints_df = pd.DataFrame(continuous_constraints)
    
    # 7. Excelファイルに保存
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        resources_df.to_excel(writer, sheet_name='リソース', index=False)
        tasks_df.to_excel(writer, sheet_name='task', index=False)
        allocation_df.to_excel(writer, sheet_name='割り当て情報', index=False)
        order_constraints_df.to_excel(writer, sheet_name='順序制約', index=False)
        fixed_tasks_df.to_excel(writer, sheet_name='固定タスク', index=False)
        continuous_constraints_df.to_excel(writer, sheet_name='連続作業制限', index=False)
    
    # 8. 生成したデータの統計情報を表示
    print(f"\n✓ Toy dataset with additional constraints generated: {output_file}")
    print(f"\n[Dataset Statistics]")
    print(f"- Workers: {num_workers} (+ 1 dummy)")
    print(f"- Equipments: {num_equipments} (+ 1 dummy)")
    print(f"- Regular Tasks: {num_tasks}")
    print(f"- Fixed Tasks: {num_fixed_tasks}")
    print(f"- Allocation Patterns: {num_patterns}")
    print(f"- Order Constraints: {len(order_constraints_df)}")
    print(f"- Continuous Work Restrictions: {num_continuous_constraints}")
    print(f"- Total Allocations: {len(allocation_df)}")
    
    # データの要約を返す
    return {
        'resources': resources_df,
        'tasks': tasks_df,
        'allocations': allocation_df,
        'order_constraints': order_constraints_df,
        'fixed_tasks': fixed_tasks_df,
        'continuous_constraints': continuous_constraints_df
    }

if __name__ == "__main__":
    # デフォルトのtoy datasetを生成
    generate_toy_dataset_additional()