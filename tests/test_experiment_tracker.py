#!/usr/bin/env python3
"""
実験追跡機能のテスト
"""

import pytest
import tempfile
import json
import pandas as pd
from pathlib import Path
import sys
import os
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from scripts.experiment_tracker import ExperimentTracker


class TestExperimentTracker:
    """ExperimentTrackerクラスのテスト"""
    
    @pytest.fixture
    def temp_tracker(self):
        """テスト用の一時的な実験追跡器を作成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = ExperimentTracker(temp_dir)
            yield tracker
    
    def test_register_experiment(self, temp_tracker):
        """実験登録のテスト"""
        experiment_data = {
            'merge_method': 'slerp',
            'models': [
                {'name': 'model1', 'weight': 0.6},
                {'name': 'model2', 'weight': 0.4}
            ],
            'config': {'alpha': 0.6}
        }
        
        exp_id = temp_tracker.register_experiment(experiment_data)
        
        # 実験が登録されたことを確認
        assert exp_id.startswith('exp_')
        assert len(temp_tracker.experiments) == 1
        
        # 登録された実験データを確認
        experiment = temp_tracker.experiments[0]
        assert experiment['merge_method'] == 'slerp'
        assert experiment['status'] == 'running'
        assert 'timestamp' in experiment
    
    def test_update_experiment(self, temp_tracker):
        """実験更新のテスト"""
        # 実験を登録
        experiment_data = {'merge_method': 'slerp'}
        exp_id = temp_tracker.register_experiment(experiment_data)
        
        # 実験を更新
        updates = {'status': 'completed', 'score': 8.5}
        temp_tracker.update_experiment(exp_id, updates)
        
        # 更新が反映されたことを確認
        experiment = temp_tracker.get_experiment(exp_id)
        assert experiment['status'] == 'completed'
        assert experiment['score'] == 8.5
        assert 'last_updated' in experiment
    
    def test_add_evaluation_results(self, temp_tracker):
        """評価結果追加のテスト"""
        # 実験を登録
        experiment_data = {'merge_method': 'slerp'}
        exp_id = temp_tracker.register_experiment(experiment_data)
        
        # 評価結果を追加
        eval_results = {
            'mt_bench_jp': {
                'overall_score': 7.5,
                'category_scores': {
                    'writing': 8.0,
                    'reasoning': 7.0
                }
            },
            'mathematical_reasoning': {
                'accuracy': 0.75
            }
        }
        
        temp_tracker.add_evaluation_results(exp_id, eval_results)
        
        # 評価結果が追加されたことを確認
        experiment = temp_tracker.get_experiment(exp_id)
        assert experiment['status'] == 'completed'
        assert 'evaluations' in experiment
        assert experiment['evaluations']['mt_bench_jp']['overall_score'] == 7.5
        assert experiment['evaluations']['mathematical_reasoning']['accuracy'] == 0.75
    
    def test_compare_experiments(self, temp_tracker):
        """実験比較のテスト"""
        # 複数の実験を登録
        experiments = [
            {
                'experiment_id': 'exp1',
                'merge_method': 'slerp',
                'models': [{'name': 'model1'}, {'name': 'model2'}],
                'evaluations': {
                    'mt_bench_jp': {'overall_score': 7.5},
                    'mathematical_reasoning': {'accuracy': 0.8}
                },
                'quantization': {
                    'quantized_size_gb': 4.2,
                    'compression_ratio': 2.5
                }
            },
            {
                'experiment_id': 'exp2', 
                'merge_method': 'evolutionary',
                'models': [{'name': 'model3'}, {'name': 'model4'}],
                'evaluations': {
                    'mt_bench_jp': {'overall_score': 8.0},
                    'mathematical_reasoning': {'accuracy': 0.85}
                },
                'quantization': {
                    'quantized_size_gb': 5.1,
                    'compression_ratio': 2.2
                }
            }
        ]
        
        for exp in experiments:
            temp_tracker.experiments.append(exp)
        
        # 比較データフレームを生成
        df = temp_tracker.compare_experiments(['exp1', 'exp2'])
        
        # データフレームの内容を確認
        assert len(df) == 2
        assert 'experiment_id' in df.columns
        assert 'method' in df.columns
        assert 'mt_bench_score' in df.columns
        assert 'math_accuracy' in df.columns
        
        # 具体的な値を確認
        exp1_row = df[df['experiment_id'] == 'exp1'].iloc[0]
        assert exp1_row['method'] == 'slerp'
        assert exp1_row['mt_bench_score'] == 7.5
        assert exp1_row['math_accuracy'] == 0.8
        
        exp2_row = df[df['experiment_id'] == 'exp2'].iloc[0]
        assert exp2_row['method'] == 'evolutionary'
        assert exp2_row['mt_bench_score'] == 8.0
        assert exp2_row['math_accuracy'] == 0.85
    
    def test_generate_leaderboard(self, temp_tracker):
        """リーダーボード生成のテスト"""
        # テスト用の実験データを追加
        experiments = [
            {
                'id': 'exp1',
                'merge_method': 'slerp',
                'models': [{'name': 'model1'}, {'name': 'model2'}],
                'evaluations': {'mt_bench_jp': {'overall_score': 7.5}}
            },
            {
                'id': 'exp2',
                'merge_method': 'evolutionary', 
                'models': [{'name': 'model3'}, {'name': 'model4'}],
                'evaluations': {'mt_bench_jp': {'overall_score': 8.5}}
            },
            {
                'id': 'exp3',
                'merge_method': 'lora',
                'models': [{'name': 'model5'}, {'name': 'model6'}],
                'evaluations': {'mt_bench_jp': {'overall_score': 6.8}}
            }
        ]
        
        temp_tracker.experiments = experiments
        
        # リーダーボードを生成
        leaderboard = temp_tracker.generate_leaderboard()
        
        # リーダーボードの検証
        assert len(leaderboard) == 3
        assert 'rank' in leaderboard.columns
        assert 'mt_bench_score' in leaderboard.columns
        
        # スコア順にソートされていることを確認
        assert leaderboard.iloc[0]['mt_bench_score'] == 8.5  # 1位
        assert leaderboard.iloc[1]['mt_bench_score'] == 7.5  # 2位
        assert leaderboard.iloc[2]['mt_bench_score'] == 6.8  # 3位
        
        # ランクが正しく設定されていることを確認
        assert leaderboard.iloc[0]['rank'] == 1
        assert leaderboard.iloc[1]['rank'] == 2
        assert leaderboard.iloc[2]['rank'] == 3
    
    def test_empty_experiments(self, temp_tracker):
        """空の実験リストのテスト"""
        # 空の状態での比較
        df = temp_tracker.compare_experiments([])
        assert df.empty
        
        # 空の状態でのリーダーボード
        leaderboard = temp_tracker.generate_leaderboard()
        assert leaderboard.empty