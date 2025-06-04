#!/usr/bin/env python3
"""
Webアプリケーションのテスト（拡張版）
"""

import pytest
import tempfile
import json
import yaml
import shutil
import io
import pandas as pd
from pathlib import Path
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

# web.appからすべての関数をインポート
from web.app import (
    AIFusionStudioApp, create_experiments_dataframe, export_experiments_json,
    export_experiments_csv, show_experiment_card, show_modern_experiment_card,
    show_detailed_experiment_card, get_nested_value, calculate_diff,
    generate_comprehensive_report, cleanup_failed_experiments,
    optimize_experiment_data, run_experiment_flow,
    run_experiment_with_realtime_logs, show_experiment_comparison,
    show_dashboard, show_new_experiment, show_experiment_results,
    show_config_management, show_guide, show_data_management, main
)


class TestAIFusionStudioApp:
    """AIFusionStudioAppクラスのテスト"""
    
    @pytest.fixture
    def temp_app(self):
        """テスト用の一時的なアプリインスタンスを作成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # テスト用のディレクトリ構造を作成
            (temp_path / "experiments").mkdir()
            (temp_path / "configs").mkdir()
            (temp_path / "models").mkdir()
            (temp_path / "evaluations").mkdir()
            
            # テスト用アプリを作成
            app = AIFusionStudioApp()
            app.experiments_dir = temp_path / "experiments"
            app.configs_dir = temp_path / "configs"
            app.models_dir = temp_path / "models"
            app.evaluations_dir = temp_path / "evaluations"
            
            yield app
    
    def test_init_default(self):
        """デフォルト初期化テスト"""
        app = AIFusionStudioApp()
        
        assert app.experiments_dir == Path("experiments")
        assert app.configs_dir == Path("configs")
        assert app.models_dir == Path("models")
        assert app.evaluations_dir == Path("evaluations")
    
    def test_load_empty_experiments(self, temp_app):
        """空の実験リストの読み込みテスト"""
        experiments = temp_app.load_experiments()
        assert experiments == []
    
    def test_load_experiments_with_data(self, temp_app):
        """実験データありの読み込みテスト"""
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
        
        db_path = temp_app.experiments_dir / "experiments_db.json"
        with open(db_path, 'w') as f:
            json.dump(test_experiments, f)
        
        experiments = temp_app.load_experiments()
        assert len(experiments) == 2
        assert experiments[0]['id'] == 'test_exp_1'
        assert experiments[1]['merge_method'] == 'evolutionary'
    
    def test_load_experiments_file_not_exists(self, temp_app):
        """実験ファイルが存在しない場合のテスト"""
        experiments = temp_app.load_experiments()
        assert experiments == []
    
    def test_load_experiments_invalid_json(self, temp_app):
        """無効なJSONファイルの場合のテスト"""
        db_path = temp_app.experiments_dir / "experiments_db.json"
        with open(db_path, 'w') as f:
            f.write('invalid json content')
        
        experiments = temp_app.load_experiments()
        assert experiments == []
    
    def test_load_configs(self, temp_app):
        """設定ファイルの読み込みテスト"""
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
        
        configs = temp_app.load_configs()
        assert len(configs) == 1
        assert configs[0]['merge_method'] == 'slerp'
        assert configs[0]['filename'] == 'test_config.yaml'
    
    def test_load_configs_empty_dir(self, temp_app):
        """空の設定ディレクトリのテスト"""
        configs = temp_app.load_configs()
        assert configs == []
    
    def test_load_configs_invalid_yaml(self, temp_app):
        """無効なYAMLファイルの処理テスト"""
        invalid_config_path = temp_app.configs_dir / "invalid_config.yaml"
        with open(invalid_config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        configs = temp_app.load_configs()
        assert configs == []
    
    def test_get_experiment_stats_with_data(self, temp_app):
        """実験統計取得テスト（データあり）"""
        test_experiments = [
            {
                'id': 'exp1',
                'status': 'completed',
                'evaluations': {'mt_bench_jp': {'overall_score': 7.5}}
            },
            {
                'id': 'exp2',
                'status': 'running'
            },
            {
                'id': 'exp3',
                'status': 'failed'
            },
            {
                'id': 'exp4',
                'status': 'completed',
                'evaluations': {'mt_bench_jp': {'overall_score': 8.0}}
            }
        ]
        
        db_path = temp_app.experiments_dir / "experiments_db.json"
        with open(db_path, 'w') as f:
            json.dump(test_experiments, f)
        
        stats = temp_app.get_experiment_stats()
        
        assert stats['total'] == 4
        assert stats['completed'] == 2
        assert stats['running'] == 1
        assert stats['failed'] == 1
        assert stats['avg_score'] == 7.75  # (7.5 + 8.0) / 2
    
    def test_get_experiment_stats_empty(self, temp_app):
        """実験統計取得テスト（データなし）"""
        stats = temp_app.get_experiment_stats()
        
        assert stats['total'] == 0
        assert stats['completed'] == 0
        assert stats['running'] == 0
        assert stats['failed'] == 0
        assert stats['avg_score'] == 0
    
    def test_get_recent_experiments(self, temp_app):
        """最近の実験取得テスト"""
        test_experiments = [
            {
                'id': 'exp1',
                'timestamp': '2024-01-01T12:00:00'
            },
            {
                'id': 'exp2',
                'timestamp': '2024-01-03T12:00:00'
            },
            {
                'id': 'exp3',
                'timestamp': '2024-01-02T12:00:00'
            }
        ]
        
        db_path = temp_app.experiments_dir / "experiments_db.json"
        with open(db_path, 'w') as f:
            json.dump(test_experiments, f)
        
        recent = temp_app.get_recent_experiments(limit=2)
        
        assert len(recent) == 2
        assert recent[0]['id'] == 'exp2'  # 最新
        assert recent[1]['id'] == 'exp3'  # 2番目
    
    def test_get_leaderboard(self, temp_app):
        """リーダーボード取得テスト"""
        test_experiments = [
            {
                'id': 'exp1',
                'status': 'completed',
                'evaluations': {'mt_bench_jp': {'overall_score': 7.5}}
            },
            {
                'id': 'exp2',
                'status': 'completed',
                'evaluations': {'mt_bench_jp': {'overall_score': 8.5}}
            },
            {
                'id': 'exp3',
                'status': 'running'  # 未完了なので含まれない
            }
        ]
        
        db_path = temp_app.experiments_dir / "experiments_db.json"
        with open(db_path, 'w') as f:
            json.dump(test_experiments, f)
        
        leaderboard = temp_app.get_leaderboard()
        
        assert len(leaderboard) == 2
        assert leaderboard[0]['id'] == 'exp2'  # 高スコアが1位
        assert leaderboard[0]['score'] == 8.5
        assert leaderboard[1]['id'] == 'exp1'
        assert leaderboard[1]['score'] == 7.5
    
    def test_delete_experiment_success(self, temp_app):
        """実験削除成功テスト"""
        test_experiments = [
            {'id': 'exp1', 'status': 'completed'},
            {'id': 'exp2', 'status': 'running'}
        ]
        
        db_path = temp_app.experiments_dir / "experiments_db.json"
        with open(db_path, 'w') as f:
            json.dump(test_experiments, f)
        
        result = temp_app.delete_experiment('exp1')
        
        assert result == True
        
        # 実験が削除されたことを確認
        experiments = temp_app.load_experiments()
        exp_ids = [exp['id'] for exp in experiments]
        assert 'exp1' not in exp_ids
        assert 'exp2' in exp_ids
    
    def test_delete_experiment_not_found(self, temp_app):
        """存在しない実験削除テスト"""
        result = temp_app.delete_experiment('nonexistent')
        assert result == False
    
    def test_save_config_success(self, temp_app):
        """設定保存成功テスト"""
        config_data = {
            'merge_method': 'evolutionary',
            'models': [{'name': 'test_model'}]
        }
        
        result = temp_app.save_config('new_config.yaml', config_data)
        
        assert result == True
        
        # ファイルが保存されたことを確認
        config_path = temp_app.configs_dir / 'new_config.yaml'
        assert config_path.exists()
        
        with open(config_path, 'r') as f:
            saved_config = yaml.safe_load(f)
            assert saved_config['merge_method'] == 'evolutionary'
    
    def test_save_config_failure(self, temp_app):
        """設定保存失敗テスト"""
        with patch('builtins.open', side_effect=IOError("Permission denied")):
            result = temp_app.save_config('invalid_config.yaml', {})
            assert result == False
    
    def test_load_config_success(self, temp_app):
        """設定読み込み成功テスト"""
        test_config = {'merge_method': 'slerp'}
        config_path = temp_app.configs_dir / 'test_config.yaml'
        
        with open(config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        config = temp_app.load_config('test_config.yaml')
        
        assert config is not None
        assert config['merge_method'] == 'slerp'
    
    def test_load_config_not_found(self, temp_app):
        """存在しない設定読み込みテスト"""
        config = temp_app.load_config('nonexistent.yaml')
        assert config is None
    
    def test_load_config_invalid_yaml(self, temp_app):
        """無効なYAML設定読み込みテスト"""
        config_path = temp_app.configs_dir / 'invalid.yaml'
        with open(config_path, 'w') as f:
            f.write('invalid: yaml: content: [')
        
        config = temp_app.load_config('invalid.yaml')
        assert config is None
    
    @patch('subprocess.run')
    def test_run_experiment_success(self, mock_subprocess, temp_app):
        """実験実行成功テスト"""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "実験が完了しました"
        mock_subprocess.return_value.stderr = ""
        
        success, stdout, stderr = temp_app.run_experiment('test_config.yaml')
        
        assert success == True
        assert "実験が完了しました" in stdout
        assert stderr == ""
    
    @patch('subprocess.run')
    def test_run_experiment_failure(self, mock_subprocess, temp_app):
        """実験実行失敗テスト"""
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stdout = ""
        mock_subprocess.return_value.stderr = "エラーが発生しました"
        
        success, stdout, stderr = temp_app.run_experiment('test_config.yaml')
        
        assert success == False
        assert stdout == ""
        assert "エラーが発生しました" in stderr
    
    @patch('subprocess.run')
    def test_run_experiment_with_skip_steps(self, mock_subprocess, temp_app):
        """スキップオプション付き実験実行テスト"""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "実験が完了しました"
        mock_subprocess.return_value.stderr = ""
        
        success, stdout, stderr = temp_app.run_experiment(
            'test_config.yaml', 
            skip_steps=['evaluate', 'quantize']
        )
        
        assert success == True
        
        # コマンドにスキップオプションが含まれることを確認
        args = mock_subprocess.call_args[0][0]
        assert '--skip' in args
        assert 'evaluate' in args
        assert 'quantize' in args


class TestUtilityFunctions:
    """ユーティリティ関数のテスト"""
    
    def test_create_experiments_dataframe_empty(self):
        """空の実験リストからDataFrame作成テスト"""
        experiments = []
        df = create_experiments_dataframe(experiments)
        assert df.empty
    
    def test_create_experiments_dataframe_with_data(self):
        """実験データからDataFrame作成テスト"""
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
        assert exp2_row['math_accuracy'] == 0
        assert exp2_row['compression_ratio'] == 0
    
    def test_export_experiments_json(self):
        """JSON エクスポートテスト"""
        experiments = [
            {'id': 'exp1', 'method': 'slerp', 'score': 7.5},
            {'id': 'exp2', 'method': 'evolutionary', 'score': 8.0}
        ]
        
        result = export_experiments_json(experiments)
        
        assert isinstance(result, str)
        data = json.loads(result)
        assert len(data) == 2
        assert data[0]['id'] == 'exp1'
    
    def test_export_experiments_csv(self):
        """CSV エクスポートテスト"""
        experiments = [
            {'id': 'exp1', 'method': 'slerp', 'score': 7.5},
            {'id': 'exp2', 'method': 'evolutionary', 'score': 8.0}
        ]
        
        result = export_experiments_csv(experiments)
        
        assert isinstance(result, str)
        assert 'exp1,slerp,7.5' in result
        assert 'exp2,evolutionary,8.0' in result
    
    def test_get_nested_value_success(self):
        """ネストされた値取得成功テスト"""
        data = {
            'evaluations': {
                'mt_bench_jp': {
                    'overall_score': 7.5
                }
            }
        }
        
        result = get_nested_value(data, 'evaluations.mt_bench_jp.overall_score')
        assert result == 7.5
    
    def test_get_nested_value_not_found(self):
        """ネストされた値取得失敗テスト"""
        data = {'key': 'value'}
        
        result = get_nested_value(data, 'nonexistent.path', default='default')
        assert result == 'default'
    
    def test_get_nested_value_none_in_path(self):
        """パス途中にNoneがある場合のテスト"""
        data = {'level1': {'level2': None}}
        
        result = get_nested_value(data, 'level1.level2.level3', default='fallback')
        assert result == 'fallback'
    
    def test_calculate_diff_numeric(self):
        """数値の差分計算テスト"""
        assert calculate_diff(10, 5) == 5
        assert calculate_diff(3.5, 2.1) == pytest.approx(1.4, rel=1e-2)
        assert calculate_diff(5, 10) == -5
    
    def test_calculate_diff_non_numeric(self):
        """非数値の差分計算テスト"""
        assert calculate_diff('text', 5) == 0
        assert calculate_diff(10, 'text') == 0
        assert calculate_diff('a', 'b') == 0
        assert calculate_diff(None, 5) == 0
    
    @patch('streamlit.markdown')
    def test_show_experiment_card(self, mock_markdown):
        """実験カード表示テスト"""
        experiment = {
            'id': 'exp1',
            'merge_method': 'slerp',
            'timestamp': '2024-01-01T12:00:00',
            'status': 'completed',
            'evaluations': {
                'mt_bench_jp': {'overall_score': 7.5}
            }
        }
        
        show_experiment_card(experiment)
        
        mock_markdown.assert_called()
        call_args = mock_markdown.call_args[0][0]
        assert 'exp1' in call_args
        assert 'slerp' in call_args
        assert '7.5' in call_args
    
    @patch('streamlit.markdown')
    def test_show_modern_experiment_card(self, mock_markdown):
        """モダン実験カード表示テスト"""
        experiment = {
            'id': 'exp1',
            'merge_method': 'evolutionary',
            'timestamp': '2024-01-01T12:00:00',
            'status': 'running',
            'evaluations': {
                'mt_bench_jp': {'overall_score': 8.0}
            },
            'quantization': {
                'quantized_size_gb': 4.5,
                'compression_ratio': 2.2
            }
        }
        
        show_modern_experiment_card(experiment)
        
        mock_markdown.assert_called()
        call_args = mock_markdown.call_args[0][0]
        assert 'exp1' in call_args
        assert 'evolutionary' in call_args
        assert '8.0' in call_args
    
    @patch('streamlit.markdown')
    def test_show_detailed_experiment_card(self, mock_markdown):
        """詳細実験カード表示テスト"""
        experiment = {
            'id': 'exp1',
            'merge_method': 'lora',
            'timestamp': '2024-01-01T12:00:00',
            'status': 'failed',
            'error': 'Test error message',
            'evaluations': {
                'mt_bench_jp': {
                    'overall_score': 6.5,
                    'category_scores': {
                        'writing': 7.0,
                        'reasoning': 6.0
                    }
                }
            }
        }
        
        show_detailed_experiment_card(experiment)
        
        mock_markdown.assert_called()
        call_args = mock_markdown.call_args[0][0]
        assert 'exp1' in call_args
        assert 'lora' in call_args
        assert 'Test error message' in call_args
    
    def test_generate_comprehensive_report(self):
        """総合レポート生成テスト"""
        experiments = [
            {
                'id': 'exp1',
                'merge_method': 'slerp',
                'timestamp': '2024-01-01T12:00:00',
                'status': 'completed',
                'evaluations': {
                    'mt_bench_jp': {'overall_score': 7.5}
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
            }
        ]
        
        report = generate_comprehensive_report(experiments)
        
        assert isinstance(report, str)
        assert 'AI Fusion Studio' in report
        assert 'exp1' in report
        assert 'exp2' in report
        assert '7.5' in report
        assert '8.0' in report
    
    def test_cleanup_failed_experiments(self):
        """失敗実験のクリーンアップテスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AIFusionStudioApp()
            app.experiments_dir = Path(temp_dir) / "experiments"
            app.experiments_dir.mkdir()
            
            experiments = [
                {'id': 'exp1', 'status': 'completed'},
                {'id': 'exp2', 'status': 'failed'},
                {'id': 'exp3', 'status': 'failed'}
            ]
            
            db_path = app.experiments_dir / "experiments_db.json"
            with open(db_path, 'w') as f:
                json.dump(experiments, f)
            
            result = cleanup_failed_experiments(app, experiments)
            
            assert result == 2
            
            with open(db_path, 'r') as f:
                remaining = json.load(f)
            
            assert len(remaining) == 1
            assert remaining[0]['id'] == 'exp1'
    
    def test_optimize_experiment_data(self):
        """実験データ最適化テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AIFusionStudioApp()
            app.experiments_dir = Path(temp_dir) / "experiments"
            app.experiments_dir.mkdir()
            
            old_date = datetime(2023, 1, 1).isoformat()
            recent_date = datetime.now().isoformat()
            
            experiments = [
                {'id': 'old_exp', 'timestamp': old_date, 'status': 'completed'},
                {'id': 'recent_exp', 'timestamp': recent_date, 'status': 'completed'}
            ]
            
            db_path = app.experiments_dir / "experiments_db.json"
            with open(db_path, 'w') as f:
                json.dump(experiments, f)
            
            result = optimize_experiment_data(app, experiments)
            
            assert result >= 0


class TestWebAppPageFunctions:
    """Webアプリページ関数のテスト"""
    
    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    @patch('streamlit.metric')
    def test_show_dashboard(self, mock_metric, mock_columns, mock_markdown):
        """ダッシュボード表示テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AIFusionStudioApp()
            app.experiments_dir = Path(temp_dir) / "experiments"
            app.experiments_dir.mkdir()
            
            mock_columns.return_value = [Mock(), Mock(), Mock(), Mock()]
            
            show_dashboard(app)
            
            mock_markdown.assert_called()
            mock_columns.assert_called()
    
    @patch('streamlit.form')
    @patch('streamlit.selectbox')
    @patch('streamlit.text_input')
    def test_show_new_experiment(self, mock_text_input, mock_selectbox, mock_form):
        """新実験ページ表示テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AIFusionStudioApp()
            app.configs_dir = Path(temp_dir) / "configs"
            app.configs_dir.mkdir()
            
            mock_form_obj = Mock()
            mock_form.return_value.__enter__.return_value = mock_form_obj
            mock_form.return_value.__exit__.return_value = None
            
            show_new_experiment(app)
            
            mock_form.assert_called()
    
    @patch('streamlit.selectbox')
    @patch('streamlit.dataframe')
    def test_show_experiment_results(self, mock_dataframe, mock_selectbox):
        """実験結果ページ表示テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AIFusionStudioApp()
            app.experiments_dir = Path(temp_dir) / "experiments"
            app.experiments_dir.mkdir()
            
            show_experiment_results(app)
            
            mock_selectbox.assert_called()
    
    @patch('streamlit.file_uploader')
    @patch('streamlit.text_area')
    def test_show_config_management(self, mock_text_area, mock_file_uploader):
        """設定管理ページ表示テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AIFusionStudioApp()
            app.configs_dir = Path(temp_dir) / "configs"
            app.configs_dir.mkdir()
            
            show_config_management(app)
            
            mock_file_uploader.assert_called()
    
    @patch('streamlit.markdown')
    def test_show_guide(self, mock_markdown):
        """ガイドページ表示テスト"""
        show_guide()
        
        mock_markdown.assert_called()
    
    @patch('streamlit.selectbox')
    @patch('streamlit.button')
    def test_show_data_management(self, mock_button, mock_selectbox):
        """データ管理ページ表示テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AIFusionStudioApp()
            app.experiments_dir = Path(temp_dir) / "experiments"
            app.experiments_dir.mkdir()
            
            show_data_management(app)
            
            mock_selectbox.assert_called()
            mock_button.assert_called()
    
    @patch('streamlit.progress')
    @patch('streamlit.status')
    @patch('subprocess.Popen')
    def test_run_experiment_with_realtime_logs(self, mock_popen, mock_status, mock_progress):
        """リアルタイムログ付き実験実行テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AIFusionStudioApp()
            app.experiments_dir = Path(temp_dir) / "experiments"
            app.experiments_dir.mkdir()
            
            mock_process = Mock()
            mock_process.poll.return_value = 0
            mock_process.stdout.readline.side_effect = [b'log line 1\n', b'log line 2\n', b'']
            mock_popen.return_value = mock_process
            
            mock_status_obj = Mock()
            mock_status.return_value.__enter__.return_value = mock_status_obj
            mock_status.return_value.__exit__.return_value = None
            
            result = run_experiment_with_realtime_logs(app, 'test_config.yaml')
            
            mock_popen.assert_called()
            mock_status.assert_called()
            assert result == True
    
    @patch('streamlit.selectbox')
    @patch('streamlit.dataframe')
    def test_show_experiment_comparison(self, mock_dataframe, mock_selectbox):
        """実験比較表示テスト"""
        experiments = [
            {
                'id': 'exp1',
                'merge_method': 'slerp',
                'evaluations': {'mt_bench_jp': {'overall_score': 7.5}}
            },
            {
                'id': 'exp2',
                'merge_method': 'evolutionary',
                'evaluations': {'mt_bench_jp': {'overall_score': 8.0}}
            }
        ]
        
        show_experiment_comparison(experiments)
        
        mock_selectbox.assert_called()
    
    @patch('subprocess.Popen')
    @patch('streamlit.success')
    @patch('streamlit.error')
    def test_run_experiment_flow_success(self, mock_error, mock_success, mock_popen):
        """実験フロー成功テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AIFusionStudioApp()
            app.experiments_dir = Path(temp_dir) / "experiments"
            app.experiments_dir.mkdir()
            
            mock_process = Mock()
            mock_process.wait.return_value = 0
            mock_process.stdout.read.return_value = b'success output'
            mock_process.stderr.read.return_value = b''
            mock_popen.return_value = mock_process
            
            run_experiment_flow(app, 'test_config.yaml')
            
            mock_success.assert_called()
    
    @patch('subprocess.Popen')
    @patch('streamlit.success')
    @patch('streamlit.error')
    def test_run_experiment_flow_failure(self, mock_error, mock_success, mock_popen):
        """実験フロー失敗テスト"""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = AIFusionStudioApp()
            app.experiments_dir = Path(temp_dir) / "experiments"
            app.experiments_dir.mkdir()
            
            mock_process = Mock()
            mock_process.wait.return_value = 1
            mock_process.stdout.read.return_value = b'error output'
            mock_process.stderr.read.return_value = b'error message'
            mock_popen.return_value = mock_process
            
            run_experiment_flow(app, 'test_config.yaml')
            
            mock_error.assert_called()


class TestMainFunction:
    """main関数のテスト"""
    
    @patch('streamlit.sidebar.selectbox')
    @patch('web.app.show_dashboard')
    def test_main_dashboard(self, mock_show_dashboard, mock_selectbox):
        """main関数ダッシュボード表示テスト"""
        mock_selectbox.return_value = '📈 ダッシュボード'
        
        main()
        
        mock_show_dashboard.assert_called()
    
    @patch('streamlit.sidebar.selectbox')
    @patch('web.app.show_new_experiment')
    def test_main_new_experiment(self, mock_show_new_experiment, mock_selectbox):
        """main関数新実験ページ表示テスト"""
        mock_selectbox.return_value = '⚙️ 新しい実験'
        
        main()
        
        mock_show_new_experiment.assert_called()
    
    @patch('streamlit.sidebar.selectbox')
    @patch('web.app.show_experiment_results')
    def test_main_experiment_results(self, mock_show_experiment_results, mock_selectbox):
        """main関数実験結果ページ表示テスト"""
        mock_selectbox.return_value = '📊 実験結果'
        
        main()
        
        mock_show_experiment_results.assert_called()
    
    @patch('streamlit.sidebar.selectbox')
    @patch('web.app.show_config_management')
    def test_main_config_management(self, mock_show_config_management, mock_selectbox):
        """main関数設定管理ページ表示テスト"""
        mock_selectbox.return_value = '📁 設定管理'
        
        main()
        
        mock_show_config_management.assert_called()
    
    @patch('streamlit.sidebar.selectbox')
    @patch('web.app.show_data_management')
    def test_main_data_management(self, mock_show_data_management, mock_selectbox):
        """main関数データ管理ページ表示テスト"""
        mock_selectbox.return_value = '🗂️ データ管理'
        
        main()
        
        mock_show_data_management.assert_called()
    
    @patch('streamlit.sidebar.selectbox')
    @patch('web.app.show_guide')
    def test_main_guide(self, mock_show_guide, mock_selectbox):
        """main関数ガイドページ表示テスト"""
        mock_selectbox.return_value = '❓ ガイド'
        
        main()
        
        mock_show_guide.assert_called()