#!/usr/bin/env python3
"""
Webアプリケーションのテスト
"""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
import sys
import os
from unittest.mock import Mock, patch

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from web.app import LLMMergeLabApp, create_experiments_dataframe


class TestLLMMergeLabApp:
    """LLMMergeLabAppクラスのテスト"""
    
    @pytest.fixture
    def temp_app(self):
        """テスト用の一時的なアプリインスタンスを作成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # テスト用のディレクトリ構造を作成
            (temp_path / "experiments").mkdir()
            (temp_path / "configs").mkdir()
            (temp_path / "models").mkdir()
            
            # テスト用アプリを作成
            app = LLMMergeLabApp()
            app.experiments_dir = temp_path / "experiments"
            app.configs_dir = temp_path / "configs"
            app.models_dir = temp_path / "models"
            
            yield app
    
    def test_load_empty_experiments(self, temp_app):
        """空の実験リストの読み込みテスト"""
        experiments = temp_app.load_experiments()
        assert experiments == []
    
    def test_load_experiments_with_data(self, temp_app):
        """実験データありの読み込みテスト"""
        # テスト用の実験データを作成
        test_experiments = [
            {
                'id': 'test_exp_1',
                'merge_method': 'slerp',
                'status': 'completed',
                'timestamp': '2024-01-01T12:00:00'
            },
            {
                'id': 'test_exp_2',
                'merge_method': 'evolutionary',
                'status': 'running',
                'timestamp': '2024-01-02T12:00:00'
            }
        ]
        
        # 実験データベースファイルを作成
        db_path = temp_app.experiments_dir / "experiments_db.json"
        with open(db_path, 'w') as f:
            json.dump(test_experiments, f)
        
        # データが正しく読み込まれることを確認
        experiments = temp_app.load_experiments()
        assert len(experiments) == 2
        assert experiments[0]['id'] == 'test_exp_1'
        assert experiments[1]['merge_method'] == 'evolutionary'
    
    def test_load_configs(self, temp_app):
        """設定ファイルの読み込みテスト"""
        # テスト用の設定ファイルを作成
        test_config = {
            'merge_method': 'slerp',
            'output_path': 'models/test_model',
            'models': [
                {'name': 'model1', 'weight': 0.6},
                {'name': 'model2', 'weight': 0.4}
            ]
        }
        
        config_path = temp_app.configs_dir / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # 設定が正しく読み込まれることを確認
        configs = temp_app.load_configs()
        assert len(configs) == 1
        assert configs[0]['merge_method'] == 'slerp'
        assert configs[0]['filename'] == 'test_config.yaml'
    
    def test_load_invalid_config(self, temp_app):
        """無効な設定ファイルの処理テスト"""
        # 無効なYAMLファイルを作成
        invalid_config_path = temp_app.configs_dir / "invalid_config.yaml"
        with open(invalid_config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        # エラーが適切に処理されることを確認
        configs = temp_app.load_configs()
        assert configs == []  # エラーのファイルはスキップされる
    
    @patch('subprocess.run')
    def test_run_experiment_success(self, mock_subprocess, temp_app):
        """実験実行成功のテスト"""
        # subprocess.runが成功を返すようにモック
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "実験が完了しました"
        mock_subprocess.return_value.stderr = ""
        
        # 実験実行
        success, stdout, stderr = temp_app.run_experiment("test_config.yaml")
        
        # 結果を確認
        assert success == True
        assert "実験が完了しました" in stdout
        assert stderr == ""
        
        # 正しいコマンドが実行されたことを確認
        mock_subprocess.assert_called_once()
        args = mock_subprocess.call_args[0][0]
        assert "python" in args
        assert "scripts/run_experiment.py" in args
        assert "test_config.yaml" in args
    
    @patch('subprocess.run')
    def test_run_experiment_failure(self, mock_subprocess, temp_app):
        """実験実行失敗のテスト"""
        # subprocess.runが失敗を返すようにモック
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stdout = ""
        mock_subprocess.return_value.stderr = "エラーが発生しました"
        
        # 実験実行
        success, stdout, stderr = temp_app.run_experiment("test_config.yaml")
        
        # 結果を確認
        assert success == False
        assert stdout == ""
        assert "エラーが発生しました" in stderr
    
    @patch('subprocess.run')
    def test_run_experiment_with_skip_steps(self, mock_subprocess, temp_app):
        """スキップオプション付き実験実行のテスト"""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "実験が完了しました"
        mock_subprocess.return_value.stderr = ""
        
        # スキップオプション付きで実験実行
        success, stdout, stderr = temp_app.run_experiment("test_config.yaml", skip_steps=['evaluate', 'quantize'])
        
        # コマンドにスキップオプションが含まれることを確認
        args = mock_subprocess.call_args[0][0]
        assert "--skip" in args
        assert "evaluate" in args
        assert "quantize" in args


class TestUtilityFunctions:
    """ユーティリティ関数のテスト"""
    
    def test_create_experiments_dataframe_empty(self):
        """空の実験リストからDataFrame作成のテスト"""
        experiments = []
        df = create_experiments_dataframe(experiments)
        assert df.empty
    
    def test_create_experiments_dataframe_with_data(self):
        """実験データからDataFrame作成のテスト"""
        experiments = [
            {
                'id': 'exp1',
                'merge_method': 'slerp',
                'timestamp': '2024-01-01T12:00:00',
                'status': 'completed',
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
                'merge_method': 'evolutionary',
                'timestamp': '2024-01-02T12:00:00',
                'status': 'completed',
                'evaluations': {
                    'mt_bench_jp': {'overall_score': 8.0}
                }
            },
            {
                'id': 'exp3',
                'merge_method': 'lora',
                'timestamp': '2024-01-03T12:00:00',
                'status': 'running'  # 未完了の実験
            }
        ]
        
        df = create_experiments_dataframe(experiments)
        
        # 完了した実験のみが含まれることを確認
        assert len(df) == 2
        
        # カラムの存在確認
        expected_columns = ['experiment_id', 'method', 'timestamp', 'status']
        for col in expected_columns:
            assert col in df.columns
        
        # データの内容確認
        exp1_row = df[df['experiment_id'] == 'exp1'].iloc[0]
        assert exp1_row['method'] == 'slerp'
        assert exp1_row['mt_bench_score'] == 7.5
        assert exp1_row['math_accuracy'] == 0.8
        assert exp1_row['model_size_gb'] == 4.2
        
        exp2_row = df[df['experiment_id'] == 'exp2'].iloc[0]
        assert exp2_row['method'] == 'evolutionary'
        assert exp2_row['mt_bench_score'] == 8.0
        # math_accuracyとmodel_size_gbは存在しないので0になる
        assert exp2_row['math_accuracy'] == 0
        assert exp2_row['compression_ratio'] == 0
    
    def test_create_experiments_dataframe_partial_data(self):
        """部分的なデータからDataFrame作成のテスト"""
        experiments = [
            {
                'id': 'exp1',
                'merge_method': 'slerp',
                'status': 'completed',
                'evaluations': {
                    'mt_bench_jp': {'overall_score': 6.5}
                }
                # quantizationデータなし
            }
        ]
        
        df = create_experiments_dataframe(experiments)
        
        assert len(df) == 1
        assert df.iloc[0]['mt_bench_score'] == 6.5
        assert df.iloc[0]['model_size_gb'] == 0  # デフォルト値
        assert df.iloc[0]['compression_ratio'] == 0  # デフォルト値