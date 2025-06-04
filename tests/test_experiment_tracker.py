#!/usr/bin/env python3
"""
実験追跡機能のテスト
"""

import pytest
import tempfile
import json
import pandas as pd
import shutil
import unittest.mock as mock
from pathlib import Path
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from scripts.experiment_tracker import ExperimentTracker, main


class TestExperimentTracker:
    """ExperimentTrackerクラスのテスト"""
    
    @pytest.fixture
    def temp_tracker(self):
        """テスト用の一時的な実験追跡器を作成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = ExperimentTracker(temp_dir)
            yield tracker
    
    def test_init_creates_directory(self):
        """初期化時のディレクトリ作成テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            exp_dir = Path(temp_dir) / "new_experiments"
            tracker = ExperimentTracker(str(exp_dir))
            
            assert exp_dir.exists()
            assert tracker.experiments_dir == exp_dir
            assert tracker.db_path == exp_dir / "experiments_db.json"
            assert tracker.experiments == []
    
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
        assert exp1_row['quantized_size_gb'] == 4.2
        assert exp1_row['compression_ratio'] == 2.5
        
        exp2_row = df[df['experiment_id'] == 'exp2'].iloc[0]
        assert exp2_row['method'] == 'evolutionary'
        assert exp2_row['mt_bench_score'] == 8.0
        assert exp2_row['math_accuracy'] == 0.85
        assert exp2_row['quantized_size_gb'] == 5.1
        assert exp2_row['compression_ratio'] == 2.2
    
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
    
    def test_get_experiment_not_found(self, temp_tracker):
        """存在しない実験の取得テスト"""
        result = temp_tracker.get_experiment('nonexistent_id')
        assert result is None
    
    def test_get_experiment_with_both_id_fields(self, temp_tracker):
        """idとexperiment_idの両方を持つ実験の取得テスト"""
        experiment_data = {
            'id': 'test_id',
            'experiment_id': 'old_id',
            'merge_method': 'slerp'
        }
        
        # 直接実験リストに追加
        temp_tracker.experiments.append(experiment_data)
        temp_tracker._save_experiments()
        
        # idフィールドが優先されることを確認
        result = temp_tracker.get_experiment('test_id')
        assert result is not None
        assert result['id'] == 'test_id'
    
    def test_update_experiment_with_experiment_id_field(self, temp_tracker):
        """experiment_idフィールドを使った実験更新テスト"""
        experiment_data = {
            'experiment_id': 'legacy_exp_id',
            'merge_method': 'slerp'
        }
        
        temp_tracker.experiments.append(experiment_data)
        temp_tracker._save_experiments()
        
        # experiment_idフィールドで更新
        updates = {'status': 'completed'}
        temp_tracker.update_experiment('legacy_exp_id', updates)
        
        # 更新が反映されたことを確認
        experiment = temp_tracker.get_experiment('legacy_exp_id')
        assert experiment['status'] == 'completed'
        assert 'last_updated' in experiment
    
    def test_add_evaluation_results_with_experiment_id_field(self, temp_tracker):
        """experiment_idフィールドを使った評価結果追加テスト"""
        experiment_data = {
            'experiment_id': 'legacy_exp_id',
            'merge_method': 'slerp'
        }
        
        temp_tracker.experiments.append(experiment_data)
        temp_tracker._save_experiments()
        
        eval_results = {
            'mt_bench_jp': {'overall_score': 8.0}
        }
        
        temp_tracker.add_evaluation_results('legacy_exp_id', eval_results)
        
        experiment = temp_tracker.get_experiment('legacy_exp_id')
        assert experiment['status'] == 'completed'
        assert 'evaluations' in experiment
        assert experiment['evaluations']['mt_bench_jp']['overall_score'] == 8.0
    
    def test_add_evaluation_results_update_existing(self, temp_tracker):
        """既存の評価結果の更新テスト"""
        experiment_data = {'merge_method': 'slerp'}
        exp_id = temp_tracker.register_experiment(experiment_data)
        
        # 初回評価結果追加
        initial_results = {
            'mt_bench_jp': {'overall_score': 7.0}
        }
        temp_tracker.add_evaluation_results(exp_id, initial_results)
        
        # 追加の評価結果
        additional_results = {
            'mathematical_reasoning': {'accuracy': 0.85}
        }
        temp_tracker.add_evaluation_results(exp_id, additional_results)
        
        experiment = temp_tracker.get_experiment(exp_id)
        assert 'mt_bench_jp' in experiment['evaluations']
        assert 'mathematical_reasoning' in experiment['evaluations']
        assert experiment['evaluations']['mt_bench_jp']['overall_score'] == 7.0
        assert experiment['evaluations']['mathematical_reasoning']['accuracy'] == 0.85
    
    def test_compare_experiments_missing_data(self, temp_tracker):
        """データが不完全な実験の比較テスト"""
        experiments = [
            {
                'id': 'exp1',
                'merge_method': 'slerp',
                'models': [{'name': 'model1'}]
                # 評価結果なし
            },
            {
                'id': 'exp2',
                'merge_method': 'evolutionary'
                # modelsなし
            }
        ]
        
        temp_tracker.experiments = experiments
        
        df = temp_tracker.compare_experiments(['exp1', 'exp2'])
        
        assert len(df) == 2
        assert df.iloc[0]['models'] == 'model1'
        assert df.iloc[1]['models'] == ''  # 空のmodelsリスト
    
    def test_compare_experiments_nonexistent_ids(self, temp_tracker):
        """存在しない実験IDでの比較テスト"""
        experiment_data = {'id': 'exp1', 'merge_method': 'slerp'}
        temp_tracker.experiments.append(experiment_data)
        
        # 存在するIDと存在しないIDを混在
        df = temp_tracker.compare_experiments(['exp1', 'nonexistent'])
        
        assert len(df) == 1  # 存在する実験のみ
        assert df.iloc[0]['experiment_id'] == 'exp1'
    
    def test_load_experiments_file_not_exists(self):
        """実験ファイルが存在しない場合のテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            tracker = ExperimentTracker(temp_dir)
            assert tracker.experiments == []
    
    def test_load_experiments_from_file(self):
        """ファイルからの実験データロードテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "experiments_db.json"
            
            # テストデータを作成
            test_data = [
                {'id': 'exp1', 'merge_method': 'slerp'},
                {'id': 'exp2', 'merge_method': 'evolutionary'}
            ]
            
            with open(db_path, 'w') as f:
                json.dump(test_data, f)
            
            # TrackerがデータをロードできることTRY確認
            tracker = ExperimentTracker(temp_dir)
            assert len(tracker.experiments) == 2
            assert tracker.experiments[0]['id'] == 'exp1'
            assert tracker.experiments[1]['id'] == 'exp2'
    
    def test_save_experiments(self, temp_tracker):
        """実験データの保存テスト"""
        experiment_data = {'merge_method': 'slerp'}
        exp_id = temp_tracker.register_experiment(experiment_data)
        
        # ファイルが作成されていることを確認
        assert temp_tracker.db_path.exists()
        
        # ファイルの内容を確認
        with open(temp_tracker.db_path, 'r') as f:
            saved_data = json.load(f)
        
        assert len(saved_data) == 1
        assert saved_data[0]['id'] == exp_id
        assert saved_data[0]['merge_method'] == 'slerp'
    
    def test_register_experiment_with_custom_id(self, temp_tracker):
        """カスタムIDでの実験登録テスト"""
        experiment_data = {
            'experiment_id': 'custom_exp_001',
            'merge_method': 'slerp'
        }
        
        exp_id = temp_tracker.register_experiment(experiment_data)
        
        assert exp_id == 'custom_exp_001'
        experiment = temp_tracker.get_experiment(exp_id)
        assert experiment['id'] == 'custom_exp_001'
        assert experiment['experiment_id'] == 'custom_exp_001'
    
    def test_generate_leaderboard_no_mt_bench_score(self, temp_tracker):
        """MT-Benchスコアがない場合のリーダーボード生成テスト"""
        experiments = [
            {
                'id': 'exp1',
                'merge_method': 'slerp',
                'evaluations': {'other_metric': {'score': 0.8}}
            }
        ]
        
        temp_tracker.experiments = experiments
        
        leaderboard = temp_tracker.generate_leaderboard()
        assert leaderboard.empty
    
    def test_generate_leaderboard_no_valid_experiments(self, temp_tracker):
        """有効な実験がない場合のリーダーボード生成テスト"""
        # IDフィールドがない実験
        experiments = [
            {'merge_method': 'slerp'}
        ]
        
        temp_tracker.experiments = experiments
        
        leaderboard = temp_tracker.generate_leaderboard()
        assert leaderboard.empty
    
    @pytest.fixture
    def temp_output_dir(self):
        """テスト用出力ディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_visualize_experiments_empty_data(self, temp_tracker, temp_output_dir):
        """空データでの可視化テスト"""
        temp_tracker.visualize_experiments(temp_output_dir)
        
        # ディレクトリが作成されることを確認
        assert Path(temp_output_dir).exists()
    
    def test_visualize_experiments_with_data(self, temp_tracker, temp_output_dir):
        """データありでの可視化テスト"""
        # テストデータを準備
        experiments = [
            {
                'id': 'exp1',
                'timestamp': '2024-01-01T10:00:00',
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
                'id': 'exp2',
                'timestamp': '2024-01-02T10:00:00',
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
        
        temp_tracker.experiments = experiments
        
        # 可視化関数をモック
        with mock.patch.object(temp_tracker, '_plot_mt_bench_trends') as mock_mt_bench, \
             mock.patch.object(temp_tracker, '_plot_size_vs_performance') as mock_size_perf, \
             mock.patch.object(temp_tracker, '_plot_method_comparison') as mock_method, \
             mock.patch.object(temp_tracker, '_create_interactive_dashboard') as mock_dashboard:
            
            temp_tracker.visualize_experiments(temp_output_dir)
            
            # 各可視化関数が呼ばれたことを確認
            mock_mt_bench.assert_called_once()
            mock_size_perf.assert_called_once()
            mock_method.assert_called_once()
            mock_dashboard.assert_called_once()
    
    def test_plot_mt_bench_trends(self, temp_tracker, temp_output_dir):
        """MT-Benchトレンドプロットテスト"""
        # データ準備
        df = pd.DataFrame({
            'timestamp': ['2024-01-01T10:00:00', '2024-01-02T10:00:00'],
            'method': ['slerp', 'evolutionary'],
            'mt_bench_score': [7.5, 8.0]
        })
        
        with mock.patch('matplotlib.pyplot.figure'), \
             mock.patch('matplotlib.pyplot.plot'), \
             mock.patch('matplotlib.pyplot.xlabel'), \
             mock.patch('matplotlib.pyplot.ylabel'), \
             mock.patch('matplotlib.pyplot.title'), \
             mock.patch('matplotlib.pyplot.legend'), \
             mock.patch('matplotlib.pyplot.xticks'), \
             mock.patch('matplotlib.pyplot.grid'), \
             mock.patch('matplotlib.pyplot.tight_layout'), \
             mock.patch('matplotlib.pyplot.savefig'), \
             mock.patch('matplotlib.pyplot.close'):
            
            temp_tracker._plot_mt_bench_trends(df, Path(temp_output_dir))
    
    def test_plot_size_vs_performance(self, temp_tracker, temp_output_dir):
        """サイズ vs 性能プロットテスト"""
        df = pd.DataFrame({
            'quantized_size_gb': [4.2, 5.1],
            'mt_bench_score': [7.5, 8.0],
            'compression_ratio': [2.5, 2.2],
            'method': ['slerp', 'evolutionary']
        })
        
        with mock.patch('matplotlib.pyplot.figure'), \
             mock.patch('matplotlib.pyplot.scatter'), \
             mock.patch('matplotlib.pyplot.colorbar'), \
             mock.patch('matplotlib.pyplot.annotate'), \
             mock.patch('matplotlib.pyplot.xlabel'), \
             mock.patch('matplotlib.pyplot.ylabel'), \
             mock.patch('matplotlib.pyplot.title'), \
             mock.patch('matplotlib.pyplot.grid'), \
             mock.patch('matplotlib.pyplot.tight_layout'), \
             mock.patch('matplotlib.pyplot.savefig'), \
             mock.patch('matplotlib.pyplot.close'):
            
            temp_tracker._plot_size_vs_performance(df, Path(temp_output_dir))
    
    def test_plot_method_comparison(self, temp_tracker, temp_output_dir):
        """手法比較プロットテスト"""
        df = pd.DataFrame({
            'method': ['slerp', 'evolutionary', 'slerp'],
            'mt_bench_score': [7.5, 8.0, 7.8],
            'math_accuracy': [0.8, 0.85, 0.82],
            'quantized_size_gb': [4.2, 5.1, 4.5]
        })
        
        with mock.patch('matplotlib.pyplot.subplots') as mock_subplots, \
             mock.patch('matplotlib.pyplot.tight_layout'), \
             mock.patch('matplotlib.pyplot.savefig'), \
             mock.patch('matplotlib.pyplot.close'):
            
            # モックaxes
            fig, axes = mock.Mock(), [[mock.Mock(), mock.Mock()], [mock.Mock(), mock.Mock()]]
            mock_subplots.return_value = (fig, axes)
            
            # DataFrameのgroupbyとplotメソッドをモック
            with mock.patch.object(df, 'groupby') as mock_groupby:
                mock_grouped = mock.Mock()
                mock_agg_result = mock.Mock()
                mock_agg_result.plot = mock.Mock()
                mock_grouped.agg.return_value = mock_agg_result
                mock_grouped.mean.return_value = mock.Mock()
                mock_grouped.__getitem__ = mock.Mock(return_value=mock_grouped)
                mock_groupby.return_value = mock_grouped
                
                with mock.patch.object(df, '__getitem__') as mock_getitem:
                    mock_series = mock.Mock()
                    mock_series.value_counts.return_value = mock.Mock()
                    mock_series.value_counts.return_value.plot = mock.Mock()
                    mock_getitem.return_value = mock_series
                    
                    temp_tracker._plot_method_comparison(df, Path(temp_output_dir))
    
    def test_create_interactive_dashboard(self, temp_tracker, temp_output_dir):
        """インタラクティブダッシュボード作成テスト"""
        df = pd.DataFrame({
            'method': ['slerp', 'evolutionary'],
            'mt_bench_score': [7.5, 8.0],
            'quantized_size_gb': [4.2, 5.1],
            'compression_ratio': [2.5, 2.2],
            'timestamp': ['2024-01-01T10:00:00', '2024-01-02T10:00:00']
        })
        
        with mock.patch('plotly.subplots.make_subplots') as mock_subplots, \
             mock.patch.object(temp_tracker, '_create_detailed_view'):
            
            mock_fig = mock.Mock()
            mock_subplots.return_value = mock_fig
            
            temp_tracker._create_interactive_dashboard(df, Path(temp_output_dir))
            
            # write_htmlが呼ばれることを確認
            mock_fig.write_html.assert_called_once()
    
    def test_create_detailed_view(self, temp_tracker, temp_output_dir):
        """詳細ビュー作成テスト"""
        df = pd.DataFrame({
            'experiment_id': ['exp1', 'exp2'],
            'method': ['slerp', 'evolutionary'],
            'models': ['model1, model2', 'model3, model4'],
            'timestamp': ['2024-01-01T10:00:00', '2024-01-02T10:00:00'],
            'mt_bench_score': [7.5, 8.5],
            'math_accuracy': [0.8, 0.85],
            'quantized_size_gb': [4.2, 5.1],
            'compression_ratio': [2.5, 2.2]
        })
        
        temp_tracker._create_detailed_view(df, Path(temp_output_dir))
        
        # HTMLファイルが作成されることを確認
        html_file = Path(temp_output_dir) / 'experiment_details.html'
        assert html_file.exists()
        
        # HTMLの内容を確認
        with open(html_file, 'r') as f:
            content = f.read()
            assert 'exp1' in content
            assert 'exp2' in content
            assert 'slerp' in content
            assert 'evolutionary' in content
    
    def test_empty_experiments(self, temp_tracker):
        """空の実験リストのテスト"""
        # 空の状態での比較
        df = temp_tracker.compare_experiments([])
        assert df.empty
        
        # 空の状態でのリーダーボード
        leaderboard = temp_tracker.generate_leaderboard()
        assert leaderboard.empty


class TestMainFunction:
    """main関数のテスト"""
    
    @pytest.fixture
    def temp_exp_dir(self):
        """テスト用実験ディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_main_register_action(self, temp_exp_dir):
        """register アクション テスト"""
        test_data = {
            'merge_method': 'slerp',
            'models': [{'name': 'model1'}, {'name': 'model2'}]
        }
        
        test_args = [
            'experiment_tracker.py',
            '--action', 'register',
            '--data', json.dumps(test_data),
            '--experiments-dir', temp_exp_dir
        ]
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('builtins.print') as mock_print:
            main()
            mock_print.assert_called_once()
            assert '実験を登録しました:' in mock_print.call_args[0][0]
    
    def test_main_update_action(self, temp_exp_dir):
        """update アクション テスト"""
        # まず実験を登録
        tracker = ExperimentTracker(temp_exp_dir)
        exp_id = tracker.register_experiment({'merge_method': 'slerp'})
        
        update_data = {'status': 'completed', 'score': 8.5}
        
        test_args = [
            'experiment_tracker.py',
            '--action', 'update',
            '--experiment-id', exp_id,
            '--data', json.dumps(update_data),
            '--experiments-dir', temp_exp_dir
        ]
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('builtins.print') as mock_print:
            main()
            mock_print.assert_called_once()
            assert '実験を更新しました:' in mock_print.call_args[0][0]
    
    def test_main_visualize_action(self, temp_exp_dir):
        """visualize アクション テスト"""
        # テストデータを準備
        tracker = ExperimentTracker(temp_exp_dir)
        exp_data = {
            'merge_method': 'slerp',
            'models': [{'name': 'model1'}],
            'evaluations': {'mt_bench_jp': {'overall_score': 7.5}}
        }
        tracker.register_experiment(exp_data)
        
        test_args = [
            'experiment_tracker.py',
            '--action', 'visualize',
            '--experiments-dir', temp_exp_dir,
            '--output-dir', f'{temp_exp_dir}/viz'
        ]
        
        with mock.patch('sys.argv', test_args), \
             mock.patch.object(tracker.__class__, 'visualize_experiments') as mock_viz, \
             mock.patch('builtins.print') as mock_print:
            main()
            mock_print.assert_called_once()
            assert '可視化を生成しました:' in mock_print.call_args[0][0]
    
    def test_main_leaderboard_action(self, temp_exp_dir):
        """leaderboard アクション テスト"""
        # テストデータを準備
        tracker = ExperimentTracker(temp_exp_dir)
        exp_data = {
            'merge_method': 'slerp',
            'models': [{'name': 'model1'}],
            'evaluations': {'mt_bench_jp': {'overall_score': 7.5}}
        }
        tracker.register_experiment(exp_data)
        
        test_args = [
            'experiment_tracker.py',
            '--action', 'leaderboard',
            '--experiments-dir', temp_exp_dir
        ]
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('builtins.print') as mock_print, \
             mock.patch('pandas.DataFrame.to_csv') as mock_to_csv:
            main()
            
            # print が複数回呼ばれることを確認
            assert mock_print.call_count >= 2
            
            # CSVが保存されることを確認
            mock_to_csv.assert_called_once()
    
    def test_main_register_without_data(self, temp_exp_dir):
        """データなしでの register アクション テスト"""
        test_args = [
            'experiment_tracker.py',
            '--action', 'register',
            '--experiments-dir', temp_exp_dir
        ]
        
        with mock.patch('sys.argv', test_args):
            # データがない場合は何もしない
            main()
    
    def test_main_update_without_data(self, temp_exp_dir):
        """データまたはIDなしでの update アクション テスト"""
        test_args = [
            'experiment_tracker.py',
            '--action', 'update',
            '--experiments-dir', temp_exp_dir
        ]
        
        with mock.patch('sys.argv', test_args):
            # IDまたはデータがない場合は何もしない
            main()
    
    def test_main_invalid_json(self, temp_exp_dir):
        """無効なJSON データテスト"""
        test_args = [
            'experiment_tracker.py',
            '--action', 'register',
            '--data', '{invalid json}',
            '--experiments-dir', temp_exp_dir
        ]
        
        with mock.patch('sys.argv', test_args), \
             pytest.raises(json.JSONDecodeError):
            main()
    
    def test_main_default_arguments(self, temp_exp_dir):
        """デフォルト引数での実行テスト"""
        test_args = [
            'experiment_tracker.py',
            '--action', 'visualize'
        ]
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.experiment_tracker.ExperimentTracker') as mock_tracker_class, \
             mock.patch('builtins.print'):
            
            mock_tracker = mock.Mock()
            mock_tracker_class.return_value = mock_tracker
            
            main()
            
            # デフォルトディレクトリで作成されることを確認
            mock_tracker_class.assert_called_once_with('experiments')