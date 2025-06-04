#!/usr/bin/env python3
"""
実験実行スクリプトのテスト
"""

import pytest
import tempfile
import json
import yaml
import shutil
import unittest.mock as mock
from pathlib import Path
import sys
import os
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from scripts.run_experiment import ExperimentRunner, main


class TestExperimentRunner:
    """ExperimentRunnerクラスのテスト"""
    
    @pytest.fixture
    def temp_config_file(self):
        """テスト用の設定ファイル"""
        config = {
            'merge_method': 'slerp',
            'output_path': 'models/test_merged_model',
            'models': [
                {'name': 'model1', 'weight': 0.6},
                {'name': 'model2', 'weight': 0.4}
            ],
            'evaluation': {
                'benchmarks': ['mt-bench-jp', 'jglue']
            },
            'quantization': {
                'method': 'awq',
                'bits': 4
            },
            'expected_results': {
                'mt_bench_jp': 7.5
            },
            'metadata': {
                'description': 'テスト実験',
                'use_case': 'research'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            yield f.name
        
        os.unlink(f.name)
    
    @pytest.fixture
    def temp_output_dir(self):
        """テスト用の出力ディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_init_success(self, temp_config_file):
        """正常な初期化テスト"""
        runner = ExperimentRunner(temp_config_file)
        
        assert runner.config_path == Path(temp_config_file)
        assert runner.skip_steps == []
        assert runner.config['merge_method'] == 'slerp'
        assert runner.experiment_id.startswith('slerp_')
        assert 'experiment_id' in runner.results
        assert 'start_time' in runner.results
    
    def test_init_with_skip_steps(self, temp_config_file):
        """スキップステップ指定での初期化テスト"""
        skip_steps = ['evaluate', 'quantize']
        runner = ExperimentRunner(temp_config_file, skip_steps=skip_steps)
        
        assert runner.skip_steps == skip_steps
    
    def test_init_config_load_failure(self):
        """設定ファイル読み込み失敗テスト"""
        with pytest.raises(FileNotFoundError):
            ExperimentRunner('/nonexistent/config.yaml')
    
    def test_init_invalid_yaml(self):
        """無効なYAMLファイルでの初期化テスト"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content: [}')
            invalid_config = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                ExperimentRunner(invalid_config)
        finally:
            os.unlink(invalid_config)
    
    def test_run_merge_success(self, temp_config_file):
        """マージ実行成功テスト"""
        runner = ExperimentRunner(temp_config_file)
        
        # subprocessのモック
        mock_result = mock.Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        
        with mock.patch('subprocess.run', return_value=mock_result):
            runner._run_merge()
            
            assert 'merge' in runner.results
            assert runner.results['merge']['status'] == 'completed'
            assert runner.results['merge']['output_path'] == 'models/test_merged_model'
    
    def test_run_merge_failure(self, temp_config_file):
        """マージ実行失敗テスト"""
        runner = ExperimentRunner(temp_config_file)
        
        # subprocessのモック（失敗）
        mock_result = mock.Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Merge failed"
        
        with mock.patch('subprocess.run', return_value=mock_result):
            with pytest.raises(RuntimeError, match="マージに失敗"):
                runner._run_merge()
    
    def test_run_evaluation_success(self, temp_config_file, temp_output_dir):
        """評価実行成功テスト"""
        runner = ExperimentRunner(temp_config_file)
        
        # 評価結果ファイルを準備
        eval_dir = Path(temp_output_dir) / f'evaluations/{runner.experiment_id}'
        eval_dir.mkdir(parents=True)
        
        eval_results = {
            'mt_bench_jp': {
                'overall_score': 8.5,
                'category_scores': {
                    'writing': 8.0,
                    'reasoning': 9.0
                }
            },
            'mathematical_reasoning': {
                'accuracy': 0.85
            }
        }
        
        eval_file = eval_dir / 'evaluation_results.json'
        with open(eval_file, 'w') as f:
            json.dump(eval_results, f)
        
        # subprocessのモック
        mock_result = mock.Mock()
        mock_result.returncode = 0
        
        with mock.patch('subprocess.run', return_value=mock_result), \
             mock.patch('pathlib.Path.cwd', return_value=Path(temp_output_dir)):
            
            # Pathをモックして相対パスを解決
            original_path = Path
            def mock_path_new(path_str):
                if path_str.startswith('evaluations/'):
                    return Path(temp_output_dir) / path_str
                return original_path(path_str)
            
            with mock.patch('pathlib.Path', side_effect=mock_path_new):
                runner._run_evaluation()
            
                assert 'evaluation' in runner.results
                assert runner.results['evaluation']['mt_bench_jp']['overall_score'] == 8.5
    
    def test_run_evaluation_failure(self, temp_config_file):
        """評価実行失敗テスト"""
        runner = ExperimentRunner(temp_config_file)
        
        # subprocessのモック（失敗）
        mock_result = mock.Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Evaluation failed"
        
        with mock.patch('subprocess.run', return_value=mock_result):
            # 評価失敗は警告のみで例外は発生しない
            runner._run_evaluation()
            
            # 評価結果がない場合は結果に含まれない
            assert 'evaluation' not in runner.results
    
    def test_run_evaluation_no_results_file(self, temp_config_file):
        """評価結果ファイルが存在しない場合のテスト"""
        runner = ExperimentRunner(temp_config_file)
        
        # subprocessのモック（成功）
        mock_result = mock.Mock()
        mock_result.returncode = 0
        
        with mock.patch('subprocess.run', return_value=mock_result):
            runner._run_evaluation()
            
            # ファイルが存在しないので評価結果は記録されない
            assert 'evaluation' not in runner.results
    
    def test_run_quantization_success(self, temp_config_file, temp_output_dir):
        """量子化実行成功テスト"""
        runner = ExperimentRunner(temp_config_file)
        
        # 量子化結果ファイルを準備
        quant_dir = Path(temp_output_dir) / f'models/quantized/{runner.experiment_id}'
        quant_dir.mkdir(parents=True)
        
        quant_info = {
            'method': 'awq',
            'quantized_size_gb': 2.5,
            'compression_ratio': 4.0
        }
        
        quant_file = quant_dir / 'quantization_info.json'
        with open(quant_file, 'w') as f:
            json.dump(quant_info, f)
        
        # subprocessのモック
        mock_result = mock.Mock()
        mock_result.returncode = 0
        
        with mock.patch('subprocess.run', return_value=mock_result), \
             mock.patch('pathlib.Path.cwd', return_value=Path(temp_output_dir)):
            
            # Pathをモックして相対パスを解決
            original_path = Path
            def mock_path_new(path_str):
                if path_str.startswith('models/quantized/'):
                    return Path(temp_output_dir) / path_str
                return original_path(path_str)
            
            with mock.patch('pathlib.Path', side_effect=mock_path_new):
                runner._run_quantization()
            
                assert 'quantization' in runner.results
                assert runner.results['quantization']['method'] == 'awq'
                assert runner.results['quantization']['quantized_size_gb'] == 2.5
    
    def test_run_quantization_failure(self, temp_config_file):
        """量子化実行失敗テスト"""
        runner = ExperimentRunner(temp_config_file)
        
        # subprocessのモック（失敗）
        mock_result = mock.Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Quantization failed"
        
        with mock.patch('subprocess.run', return_value=mock_result):
            # 量子化失敗は警告のみで例外は発生しない
            runner._run_quantization()
            
            # 量子化結果がない場合は結果に含まれない
            assert 'quantization' not in runner.results
    
    def test_register_experiment(self, temp_config_file):
        """実験登録テスト"""
        runner = ExperimentRunner(temp_config_file)
        runner.results['evaluation'] = {
            'mt_bench_jp': {'overall_score': 8.0}
        }
        
        with mock.patch('subprocess.run') as mock_subprocess:
            runner._register_experiment()
            
            # 2回subprocess.runが呼ばれることを確認（登録と更新）
            assert mock_subprocess.call_count == 2
            
            # 最初の呼び出し（登録）
            first_call = mock_subprocess.call_args_list[0]
            assert '--action' in first_call[0][0]
            assert 'register' in first_call[0][0]
            
            # 2回目の呼び出し（評価結果更新）
            second_call = mock_subprocess.call_args_list[1]
            assert '--action' in second_call[0][0]
            assert 'update' in second_call[0][0]
    
    def test_register_experiment_no_evaluation(self, temp_config_file):
        """評価結果なしでの実験登録テスト"""
        runner = ExperimentRunner(temp_config_file)
        
        with mock.patch('subprocess.run') as mock_subprocess:
            runner._register_experiment()
            
            # 1回のみsubprocess.runが呼ばれることを確認（登録のみ）
            assert mock_subprocess.call_count == 1
    
    def test_generate_report(self, temp_config_file, temp_output_dir):
        """レポート生成テスト"""
        runner = ExperimentRunner(temp_config_file)
        runner.results['evaluation'] = {
            'mt_bench_jp': {
                'overall_score': 8.5,
                'category_scores': {
                    'writing': 8.0,
                    'reasoning': 9.0
                }
            }
        }
        runner.results['quantization'] = {
            'method': 'awq',
            'quantized_size_gb': 2.5,
            'compression_ratio': 4.0\n        }\n        \n        with mock.patch('pathlib.Path.cwd', return_value=Path(temp_output_dir)):\n            # Pathをモックして相対パスを解決\n            original_path = Path\n            def mock_path_new(path_str):\n                if path_str.startswith('experiments/'):\n                    return Path(temp_output_dir) / path_str\n                return original_path(path_str)\n            \n            with mock.patch('pathlib.Path', side_effect=mock_path_new):\n                runner._generate_report()\n            \n                # レポートファイルの存在確認\n                report_dir = Path(temp_output_dir) / f'experiments/{runner.experiment_id}/report'\n                assert (report_dir / 'summary.md').exists()\n                assert (report_dir / 'full_results.json').exists()\n                \n                # サマリーファイルの内容確認\n                with open(report_dir / 'summary.md', 'r') as f:\n                    content = f.read()\n                    assert runner.experiment_id in content\n                    assert 'slerp' in content\n                    assert '8.5' in content  # MT-Benchスコア\n                    assert 'awq' in content  # 量子化手法\n                \n                # 詳細結果ファイルの内容確認\n                with open(report_dir / 'full_results.json', 'r') as f:\n                    results = json.load(f)\n                    assert results['experiment_id'] == runner.experiment_id\n    \n    def test_generate_report_minimal(self, temp_config_file, temp_output_dir):\n        \"\"\"最小限の結果でのレポート生成テスト\"\"\"\n        # 評価・量子化結果なしの設定\n        config = {\n            'merge_method': 'slerp',\n            'output_path': 'models/test_merged_model',\n            'models': [{'name': 'model1'}]\n        }\n        \n        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:\n            yaml.dump(config, f)\n            minimal_config = f.name\n        \n        try:\n            runner = ExperimentRunner(minimal_config)\n            \n            with mock.patch('pathlib.Path.cwd', return_value=Path(temp_output_dir)):\n                original_path = Path\n                def mock_path_new(path_str):\n                    if path_str.startswith('experiments/'):\n                        return Path(temp_output_dir) / path_str\n                    return original_path(path_str)\n                \n                with mock.patch('pathlib.Path', side_effect=mock_path_new):\n                    runner._generate_report()\n                \n                    # レポートファイルの存在確認\n                    report_dir = Path(temp_output_dir) / f'experiments/{runner.experiment_id}/report'\n                    assert (report_dir / 'summary.md').exists()\n                    \n                    # 内容確認（評価・量子化結果はなし）\n                    with open(report_dir / 'summary.md', 'r') as f:\n                        content = f.read()\n                        assert runner.experiment_id in content\n                        assert 'slerp' in content\n                        assert '評価結果' not in content\n                        assert '量子化結果' not in content\n        \n        finally:\n            os.unlink(minimal_config)\n    \n    def test_run_full_success(self, temp_config_file):\n        \"\"\"完全な実験実行成功テスト\"\"\"\n        runner = ExperimentRunner(temp_config_file)\n        \n        with mock.patch.object(runner, '_run_merge') as mock_merge, \\\n             mock.patch.object(runner, '_run_evaluation') as mock_eval, \\\n             mock.patch.object(runner, '_run_quantization') as mock_quant, \\\n             mock.patch.object(runner, '_register_experiment') as mock_register, \\\n             mock.patch.object(runner, '_generate_report') as mock_report:\n            \n            results = runner.run()\n            \n            # 全てのステップが実行されたことを確認\n            mock_merge.assert_called_once()\n            mock_eval.assert_called_once()\n            mock_quant.assert_called_once()\n            mock_register.assert_called_once()\n            mock_report.assert_called_once()\n            \n            # 結果の確認\n            assert results['status'] == 'completed'\n            assert 'end_time' in results\n    \n    def test_run_with_skip_steps(self, temp_config_file):\n        \"\"\"ステップスキップでの実験実行テスト\"\"\"\n        skip_steps = ['evaluate', 'quantize']\n        runner = ExperimentRunner(temp_config_file, skip_steps=skip_steps)\n        \n        with mock.patch.object(runner, '_run_merge') as mock_merge, \\\n             mock.patch.object(runner, '_run_evaluation') as mock_eval, \\\n             mock.patch.object(runner, '_run_quantization') as mock_quant, \\\n             mock.patch.object(runner, '_register_experiment') as mock_register, \\\n             mock.patch.object(runner, '_generate_report') as mock_report:\n            \n            runner.run()\n            \n            # スキップされたステップは呼ばれない\n            mock_merge.assert_called_once()\n            mock_eval.assert_not_called()\n            mock_quant.assert_not_called()\n            mock_register.assert_called_once()\n            mock_report.assert_called_once()\n    \n    def test_run_with_failure(self, temp_config_file):\n        \"\"\"実験実行失敗テスト\"\"\"\n        runner = ExperimentRunner(temp_config_file)\n        \n        with mock.patch.object(runner, '_run_merge', side_effect=Exception(\"Merge failed\")):\n            with pytest.raises(Exception, match=\"Merge failed\"):\n                runner.run()\n            \n            # 失敗状態が記録される\n            assert runner.results['status'] == 'failed'\n            assert 'error' in runner.results\n    \n    def test_cleanup(self, temp_config_file):\n        \"\"\"クリーンアップテスト\"\"\"\n        runner = ExperimentRunner(temp_config_file)\n        \n        # 現在は何も実装されていないが、エラーが発生しないことを確認\n        runner.cleanup()\n\n\nclass TestMainFunction:\n    \"\"\"main関数のテスト\"\"\"\n    \n    @pytest.fixture\n    def temp_config_file(self):\n        \"\"\"テスト用の設定ファイル\"\"\"\n        config = {\n            'merge_method': 'slerp',\n            'output_path': 'models/test_merged_model',\n            'models': [\n                {'name': 'model1', 'weight': 0.6},\n                {'name': 'model2', 'weight': 0.4}\n            ]\n        }\n        \n        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:\n            yaml.dump(config, f)\n            yield f.name\n        \n        os.unlink(f.name)\n    \n    def test_main_dry_run(self, temp_config_file):\n        \"\"\"ドライラン実行テスト\"\"\"\n        test_args = [\n            'run_experiment.py',\n            temp_config_file,\n            '--dry-run'\n        ]\n        \n        with mock.patch('sys.argv', test_args), \\\n             mock.patch('builtins.print') as mock_print:\n            \n            main()\n            \n            # ドライランの出力が表示されたことを確認\n            assert mock_print.call_count >= 4\n            print_calls = [call[0][0] for call in mock_print.call_args_list]\n            assert any('実験実行計画' in call for call in print_calls)\n    \n    def test_main_dry_run_with_skip(self, temp_config_file):\n        \"\"\"スキップオプション付きドライラン実行テスト\"\"\"\n        test_args = [\n            'run_experiment.py',\n            temp_config_file,\n            '--dry-run',\n            '--skip', 'evaluate', 'quantize'\n        ]\n        \n        with mock.patch('sys.argv', test_args), \\\n             mock.patch('builtins.print') as mock_print:\n            \n            main()\n            \n            print_calls = [call[0][0] for call in mock_print.call_args_list]\n            assert any('スキップ:' in call for call in print_calls)\n    \n    def test_main_success(self, temp_config_file):\n        \"\"\"main関数正常実行テスト\"\"\"\n        test_args = [\n            'run_experiment.py',\n            temp_config_file\n        ]\n        \n        mock_results = {\n            'experiment_id': 'test_exp_123',\n            'evaluation': {\n                'mt_bench_jp': {'overall_score': 8.5}\n            },\n            'quantization': {\n                'quantized_size_gb': 2.5\n            }\n        }\n        \n        with mock.patch('sys.argv', test_args), \\\n             mock.patch('scripts.run_experiment.ExperimentRunner') as mock_runner_class, \\\n             mock.patch('builtins.print') as mock_print:\n            \n            mock_runner = mock.Mock()\n            mock_runner.run.return_value = mock_results\n            mock_runner.cleanup = mock.Mock()\n            mock_runner_class.return_value = mock_runner\n            \n            main()\n            \n            # ExperimentRunnerが適切に作成されたことを確認\n            mock_runner_class.assert_called_once_with(temp_config_file, skip_steps=None)\n            \n            # run メソッドが呼ばれたことを確認\n            mock_runner.run.assert_called_once()\n            \n            # cleanup が呼ばれたことを確認\n            mock_runner.cleanup.assert_called_once()\n            \n            # 結果が出力されたことを確認\n            print_calls = [call[0][0] for call in mock_print.call_args_list]\n            assert any('実験完了' in call for call in print_calls)\n            assert any('test_exp_123' in call for call in print_calls)\n    \n    def test_main_with_skip_options(self, temp_config_file):\n        \"\"\"スキップオプション付きmain実行テスト\"\"\"\n        test_args = [\n            'run_experiment.py',\n            temp_config_file,\n            '--skip', 'evaluate', 'report'\n        ]\n        \n        mock_results = {\n            'experiment_id': 'test_exp_456'\n        }\n        \n        with mock.patch('sys.argv', test_args), \\\n             mock.patch('scripts.run_experiment.ExperimentRunner') as mock_runner_class, \\\n             mock.patch('builtins.print'):\n            \n            mock_runner = mock.Mock()\n            mock_runner.run.return_value = mock_results\n            mock_runner.cleanup = mock.Mock()\n            mock_runner_class.return_value = mock_runner\n            \n            main()\n            \n            # スキップオプションが渡されたことを確認\n            mock_runner_class.assert_called_once_with(temp_config_file, skip_steps=['evaluate', 'report'])\n    \n    def test_main_failure(self, temp_config_file):\n        \"\"\"main関数実行失敗テスト\"\"\"\n        test_args = [\n            'run_experiment.py',\n            temp_config_file\n        ]\n        \n        with mock.patch('sys.argv', test_args), \\\n             mock.patch('scripts.run_experiment.ExperimentRunner') as mock_runner_class, \\\n             pytest.raises(SystemExit) as exc_info:\n            \n            mock_runner = mock.Mock()\n            mock_runner.run.side_effect = Exception(\"Experiment failed\")\n            mock_runner.cleanup = mock.Mock()\n            mock_runner_class.return_value = mock_runner\n            \n            main()\n        \n        assert exc_info.value.code == 1\n        \n        # cleanup が例外時でも呼ばれることを確認\n        mock_runner.cleanup.assert_called_once()\n    \n    def test_main_no_evaluation_or_quantization(self, temp_config_file):\n        \"\"\"評価・量子化結果なしでのmain実行テスト\"\"\"\n        test_args = [\n            'run_experiment.py',\n            temp_config_file\n        ]\n        \n        mock_results = {\n            'experiment_id': 'test_exp_789'\n            # evaluation と quantization は含まれない\n        }\n        \n        with mock.patch('sys.argv', test_args), \\\n             mock.patch('scripts.run_experiment.ExperimentRunner') as mock_runner_class, \\\n             mock.patch('builtins.print') as mock_print:\n            \n            mock_runner = mock.Mock()\n            mock_runner.run.return_value = mock_results\n            mock_runner.cleanup = mock.Mock()\n            mock_runner_class.return_value = mock_runner\n            \n            main()\n            \n            # 実験IDは出力されるが、スコアやサイズは出力されない\n            print_calls = [call[0][0] for call in mock_print.call_args_list]\n            assert any('test_exp_789' in call for call in print_calls)\n            assert not any('MT-Benchスコア' in call for call in print_calls)\n            assert not any('量子化後サイズ' in call for call in print_calls)\n    \n    def test_main_invalid_config_file(self):\n        \"\"\"無効な設定ファイルでのmain実行テスト\"\"\"\n        test_args = [\n            'run_experiment.py',\n            '/nonexistent/config.yaml'\n        ]\n        \n        with mock.patch('sys.argv', test_args), \\\n             pytest.raises(SystemExit) as exc_info:\n            \n            main()\n        \n        assert exc_info.value.code == 1"