#!/usr/bin/env python3
"""
量子化機能のテスト
"""

import pytest
import tempfile
import json
import shutil
import unittest.mock as mock
from pathlib import Path
import sys
import os
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from scripts.quantize import ModelQuantizer, main


class MockConfig:
    """テスト用のモック設定"""
    def __init__(self):
        self.num_hidden_layers = 32
        self.hidden_size = 4096
        self.intermediate_size = 11008
        
    @classmethod
    def from_pretrained(cls, model_path):
        return cls()


class TestModelQuantizer:
    """ModelQuantizerクラスのテスト"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """テスト用の一時モデルディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        # 必要なファイルを作成
        config_path = Path(temp_dir) / 'config.json'
        config_path.write_text('{"model_type": "test"}')
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def temp_output_dir(self):
        """テスト用の出力ディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_init_success(self, temp_model_dir, temp_output_dir):
        """正常な初期化テスト"""
        with mock.patch('scripts.quantize.AutoConfig', MockConfig):
            quantizer = ModelQuantizer(temp_model_dir, temp_output_dir)
            
            assert quantizer.model_path == Path(temp_model_dir)
            assert quantizer.output_dir == Path(temp_output_dir)
            assert quantizer.model_size_gb > 0
            assert Path(temp_output_dir).exists()
    
    def test_init_model_not_found(self):
        """存在しないモデルパスでの初期化テスト"""
        with pytest.raises(ValueError, match="モデルパスが見つかりません"):
            ModelQuantizer("/nonexistent/path")
    
    def test_init_config_load_failure(self, temp_model_dir):
        """設定読み込み失敗時の初期化テスト"""
        # config.jsonを削除
        config_path = Path(temp_model_dir) / 'config.json'
        config_path.unlink()
        
        with mock.patch('scripts.quantize.AutoConfig.from_pretrained', side_effect=Exception("Config load error")):
            with pytest.raises(Exception):
                ModelQuantizer(temp_model_dir)
    
    def test_estimate_model_size(self, temp_model_dir):
        """モデルサイズ推定テスト"""
        with mock.patch('scripts.quantize.AutoConfig', MockConfig):
            quantizer = ModelQuantizer(temp_model_dir)
            
            # MockConfigの値を使って予想サイズを計算
            config = MockConfig()
            expected_params = config.num_hidden_layers * (
                4 * config.hidden_size * config.intermediate_size +
                4 * config.hidden_size * config.hidden_size
            )
            expected_size_gb = (expected_params * 2) / (1024**3)
            
            assert abs(quantizer.model_size_gb - expected_size_gb) < 0.01
    
    def test_quantize_awq_import_error(self, temp_model_dir):
        """AWQライブラリが利用できない場合のテスト"""
        with mock.patch('scripts.quantize.AutoConfig', MockConfig), \
             mock.patch('scripts.quantize.AutoAWQForCausalLM', None):
            
            quantizer = ModelQuantizer(temp_model_dir)
            
            with pytest.raises(ImportError, match="AWQライブラリがインストールされていません"):
                quantizer.quantize_awq()
    
    def test_quantize_awq_success(self, temp_model_dir, temp_output_dir):
        """AWQ量子化成功テスト"""
        # モックオブジェクト
        mock_model = mock.Mock()
        mock_model.quantize = mock.Mock()
        mock_model.save_quantized = mock.Mock()
        
        mock_tokenizer = mock.Mock()
        mock_tokenizer.save_pretrained = mock.Mock()
        
        mock_awq_class = mock.Mock()
        mock_awq_class.from_pretrained.return_value = mock_model
        
        with mock.patch('scripts.quantize.AutoConfig', MockConfig), \
             mock.patch('scripts.quantize.AutoAWQForCausalLM', mock_awq_class), \
             mock.patch('scripts.quantize.AutoTokenizer') as mock_tokenizer_class:
            
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            quantizer = ModelQuantizer(temp_model_dir, temp_output_dir)
            
            # _prepare_calibration_dataとディレクトリサイズ計算をモック
            with mock.patch.object(quantizer, '_prepare_calibration_data') as mock_calib, \
                 mock.patch.object(quantizer, '_calculate_directory_size', return_value=2.5), \
                 mock.patch.object(quantizer, '_save_quantization_metadata'):
                
                mock_calib.return_value = ['dummy_data']
                
                result = quantizer.quantize_awq(w_bit=4, group_size=128)
                
                # メソッドが適切に呼ばれたことを確認
                mock_awq_class.from_pretrained.assert_called_once()
                mock_model.quantize.assert_called_once()
                mock_model.save_quantized.assert_called_once()
                mock_tokenizer.save_pretrained.assert_called_once()
                
                # 結果パスの確認
                assert str(result).endswith('-awq-4bit-g128')
    
    def test_quantize_awq_failure(self, temp_model_dir, temp_output_dir):
        """AWQ量子化失敗テスト"""
        mock_awq_class = mock.Mock()
        mock_awq_class.from_pretrained.side_effect = Exception("AWQ load error")
        
        with mock.patch('scripts.quantize.AutoConfig', MockConfig), \
             mock.patch('scripts.quantize.AutoAWQForCausalLM', mock_awq_class):
            
            quantizer = ModelQuantizer(temp_model_dir, temp_output_dir)
            
            with pytest.raises(Exception):
                quantizer.quantize_awq()
    
    def test_quantize_gptq_import_error(self, temp_model_dir):
        """GPTQライブラリが利用できない場合のテスト"""
        with mock.patch('scripts.quantize.AutoConfig', MockConfig), \
             mock.patch('scripts.quantize.AutoGPTQForCausalLM', None):
            
            quantizer = ModelQuantizer(temp_model_dir)
            
            with pytest.raises(ImportError, match="Auto-GPTQライブラリがインストールされていません"):
                quantizer.quantize_gptq()
    
    def test_quantize_gptq_success(self, temp_model_dir, temp_output_dir):
        """GPTQ量子化成功テスト"""
        # モックオブジェクト
        mock_model = mock.Mock()
        mock_model.quantize = mock.Mock()
        mock_model.save_quantized = mock.Mock()
        
        mock_tokenizer = mock.Mock()
        mock_tokenizer.save_pretrained = mock.Mock()
        
        mock_gptq_class = mock.Mock()
        mock_gptq_class.from_pretrained.return_value = mock_model
        
        mock_config_class = mock.Mock()
        
        with mock.patch('scripts.quantize.AutoConfig', MockConfig), \
             mock.patch('scripts.quantize.AutoGPTQForCausalLM', mock_gptq_class), \
             mock.patch('scripts.quantize.BaseQuantizeConfig', mock_config_class), \
             mock.patch('scripts.quantize.AutoTokenizer') as mock_tokenizer_class:
            
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            quantizer = ModelQuantizer(temp_model_dir, temp_output_dir)
            
            # _prepare_calibration_dataとディレクトリサイズ計算をモック
            with mock.patch.object(quantizer, '_prepare_calibration_data') as mock_calib, \
                 mock.patch.object(quantizer, '_calculate_directory_size', return_value=2.0), \
                 mock.patch.object(quantizer, '_save_quantization_metadata'):
                
                mock_calib.return_value = ['dummy_data']
                
                result = quantizer.quantize_gptq(bits=4, group_size=128)
                
                # メソッドが適切に呼ばれたことを確認
                mock_gptq_class.from_pretrained.assert_called_once()
                mock_model.quantize.assert_called_once()
                mock_model.save_quantized.assert_called_once()
                mock_tokenizer.save_pretrained.assert_called_once()
                
                # 結果パスの確認
                assert str(result).endswith('-gptq-4bit-g128')
    
    def test_quantize_gptq_failure(self, temp_model_dir, temp_output_dir):
        """GPTQ量子化失敗テスト"""
        mock_gptq_class = mock.Mock()
        mock_gptq_class.from_pretrained.side_effect = Exception("GPTQ load error")
        
        mock_config_class = mock.Mock()
        
        with mock.patch('scripts.quantize.AutoConfig', MockConfig), \
             mock.patch('scripts.quantize.AutoGPTQForCausalLM', mock_gptq_class), \
             mock.patch('scripts.quantize.BaseQuantizeConfig', mock_config_class):
            
            quantizer = ModelQuantizer(temp_model_dir, temp_output_dir)
            
            with pytest.raises(Exception):
                quantizer.quantize_gptq()
    
    def test_convert_to_gguf_success(self, temp_model_dir, temp_output_dir):
        """GGUF変換成功テスト"""
        with mock.patch('scripts.quantize.AutoConfig', MockConfig):
            quantizer = ModelQuantizer(temp_model_dir, temp_output_dir)
            
            # subprocessのモック
            mock_result = mock.Mock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            
            with mock.patch('subprocess.run', return_value=mock_result), \
                 mock.patch('shutil.rmtree'), \
                 mock.patch.object(quantizer, '_save_quantization_metadata'), \
                 mock.patch.object(Path, 'stat') as mock_stat:
                
                # ファイルサイズのモック
                mock_stat_result = mock.Mock()
                mock_stat_result.st_size = 2.0 * (1024**3)  # 2GB
                mock_stat.return_value = mock_stat_result
                
                result = quantizer.convert_to_gguf(quantization="q4_k_m")
                
                # 結果パスの確認
                assert str(result).endswith('.gguf')
                assert 'q4_k_m' in str(result)
    
    def test_convert_to_gguf_conversion_failure(self, temp_model_dir, temp_output_dir):
        """GGUF変換失敗テスト（変換ステップ）"""
        with mock.patch('scripts.quantize.AutoConfig', MockConfig):
            quantizer = ModelQuantizer(temp_model_dir, temp_output_dir)
            
            # 最初のsubprocess（変換）が失敗
            mock_result = mock.Mock()
            mock_result.returncode = 1
            mock_result.stderr = "Conversion error"
            
            with mock.patch('subprocess.run', return_value=mock_result):
                with pytest.raises(RuntimeError, match="GGML変換に失敗"):
                    quantizer.convert_to_gguf()
    
    def test_convert_to_gguf_quantization_failure(self, temp_model_dir, temp_output_dir):
        """GGUF変換失敗テスト（量子化ステップ）"""
        with mock.patch('scripts.quantize.AutoConfig', MockConfig):
            quantizer = ModelQuantizer(temp_model_dir, temp_output_dir)
            
            def side_effect_subprocess(cmd, **kwargs):
                if "convert.py" in cmd:
                    # 変換は成功
                    result = mock.Mock()
                    result.returncode = 0
                    return result
                else:
                    # 量子化は失敗
                    result = mock.Mock()
                    result.returncode = 1
                    result.stderr = "Quantization error"
                    return result
            
            with mock.patch('subprocess.run', side_effect=side_effect_subprocess):
                with pytest.raises(RuntimeError, match="量子化に失敗"):
                    quantizer.convert_to_gguf()
    
    def test_prepare_calibration_data(self, temp_model_dir):
        """キャリブレーションデータ準備テスト"""
        mock_tokenizer = mock.Mock()
        mock_tokenizer.return_value = {'input_ids': 'dummy_tokens'}
        
        with mock.patch('scripts.quantize.AutoConfig', MockConfig):
            quantizer = ModelQuantizer(temp_model_dir)
            
            data = quantizer._prepare_calibration_data(mock_tokenizer, num_samples=5)
            
            assert len(data) == 5
            assert mock_tokenizer.call_count == 5
            
            # より多くのサンプル数をテスト（テキストの繰り返し）
            data_large = quantizer._prepare_calibration_data(mock_tokenizer, num_samples=10)
            assert len(data_large) == 10
    
    def test_calculate_directory_size(self, temp_model_dir):
        """ディレクトリサイズ計算テスト"""
        # テストファイルを作成
        test_file1 = Path(temp_model_dir) / 'file1.txt'
        test_file2 = Path(temp_model_dir) / 'subdir' / 'file2.txt'
        
        test_file2.parent.mkdir(exist_ok=True)
        test_file1.write_text('a' * 1024)  # 1KB
        test_file2.write_text('b' * 2048)  # 2KB
        
        with mock.patch('scripts.quantize.AutoConfig', MockConfig):
            quantizer = ModelQuantizer(temp_model_dir)
            
            size_gb = quantizer._calculate_directory_size(Path(temp_model_dir))
            expected_size = (1024 + 2048) / (1024**3)
            
            assert abs(size_gb - expected_size) < 1e-10
    
    def test_save_quantization_metadata_directory(self, temp_model_dir, temp_output_dir):
        """メタデータ保存テスト（ディレクトリ）"""
        with mock.patch('scripts.quantize.AutoConfig', MockConfig):
            quantizer = ModelQuantizer(temp_model_dir, temp_output_dir)
            
            metadata = {
                'method': 'awq',
                'w_bit': 4,
                'group_size': 128,
                'compression_ratio': 2.0
            }
            
            output_path = Path(temp_output_dir) / 'test_model'
            output_path.mkdir()
            
            quantizer._save_quantization_metadata(output_path, metadata)
            
            # メタデータファイルが作成されたことを確認
            metadata_file = output_path / 'quantization_info.json'
            assert metadata_file.exists()
            
            # 内容を確認
            with open(metadata_file, 'r') as f:
                saved_metadata = json.load(f)
            
            assert saved_metadata['method'] == 'awq'
            assert saved_metadata['w_bit'] == 4
            assert 'timestamp' in saved_metadata
            assert 'original_model' in saved_metadata
    
    def test_save_quantization_metadata_file(self, temp_model_dir, temp_output_dir):
        """メタデータ保存テスト（ファイル）"""
        with mock.patch('scripts.quantize.AutoConfig', MockConfig):
            quantizer = ModelQuantizer(temp_model_dir, temp_output_dir)
            
            metadata = {
                'method': 'gguf',
                'quantization': 'q4_k_m'
            }
            
            metadata_file = Path(temp_output_dir) / 'metadata.json'
            
            quantizer._save_quantization_metadata(metadata_file, metadata)
            
            # ファイルが作成されたことを確認
            assert metadata_file.exists()
            
            # 内容を確認
            with open(metadata_file, 'r') as f:
                saved_metadata = json.load(f)
            
            assert saved_metadata['method'] == 'gguf'
            assert saved_metadata['quantization'] == 'q4_k_m'
            assert 'timestamp' in saved_metadata
    
    def test_benchmark_quantized_model_success(self, temp_model_dir, temp_output_dir):
        """量子化モデルベンチマーク成功テスト"""
        # モックモデルとトークナイザー
        mock_model = mock.Mock()
        mock_tokenizer = mock.Mock()
        
        # generateメソッドのモック
        mock_outputs = mock.Mock()
        mock_outputs.shape = [1, 60]  # バッチサイズ1、60トークン
        mock_model.generate.return_value = mock_outputs
        
        # tokenizerの戻り値
        mock_inputs = {'input_ids': mock.Mock()}
        mock_inputs['input_ids'].shape = [1, 10]  # 入力は10トークン
        mock_tokenizer.return_value = mock_inputs
        
        with mock.patch('scripts.quantize.AutoConfig', MockConfig), \
             mock.patch('scripts.quantize.AutoModelForCausalLM') as mock_model_class, \
             mock.patch('scripts.quantize.AutoTokenizer') as mock_tokenizer_class, \
             mock.patch('torch.no_grad'), \
             mock.patch('time.time', side_effect=[0, 1, 1, 2, 2, 3]):  # 各プロンプトに1秒ずつ
            
            mock_model_class.from_pretrained.return_value = mock_model
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            
            quantizer = ModelQuantizer(temp_model_dir, temp_output_dir)
            
            model_path = Path(temp_output_dir) / 'test_model'
            model_path.mkdir()
            
            results = quantizer.benchmark_quantized_model(model_path)
            
            # 結果の確認
            assert 'model_path' in results
            assert 'benchmarks' in results
            assert 'inference_speed' in results['benchmarks']
            
            speed_info = results['benchmarks']['inference_speed']
            assert 'tokens_per_second' in speed_info
            assert 'total_time' in speed_info
            assert 'total_tokens' in speed_info
            
            # 計算の確認：各プロンプトで50トークン生成、3回で150トークン、3秒で50 tokens/sec
            assert speed_info['total_tokens'] == 150
            assert speed_info['total_time'] == 3
            assert abs(speed_info['tokens_per_second'] - 50.0) < 1e-10
    
    def test_benchmark_quantized_model_failure(self, temp_model_dir, temp_output_dir):
        """量子化モデルベンチマーク失敗テスト"""
        with mock.patch('scripts.quantize.AutoConfig', MockConfig), \
             mock.patch('scripts.quantize.AutoModelForCausalLM') as mock_model_class:
            
            mock_model_class.from_pretrained.side_effect = Exception("Model load error")
            
            quantizer = ModelQuantizer(temp_model_dir, temp_output_dir)
            
            model_path = Path(temp_output_dir) / 'test_model'
            
            results = quantizer.benchmark_quantized_model(model_path)
            
            # エラーが記録されていることを確認
            assert 'benchmarks' in results
            assert 'error' in results['benchmarks']
            assert 'Model load error' in results['benchmarks']['error']


class TestMainFunction:
    """main関数のテスト"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """テスト用の一時モデルディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        # 必要なファイルを作成
        config_path = Path(temp_dir) / 'config.json'
        config_path.write_text('{"model_type": "test"}')
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def temp_output_dir(self):
        """テスト用の出力ディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_main_awq_success(self, temp_model_dir, temp_output_dir):
        """main関数AWQ量子化成功テスト"""
        test_args = [
            'quantize.py',
            '--model-path', temp_model_dir,
            '--method', 'awq',
            '--bits', '4',
            '--group-size', '128',
            '--output-dir', temp_output_dir
        ]
        
        mock_quantized_path = Path(temp_output_dir) / 'test-awq-4bit-g128'
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.quantize.ModelQuantizer') as mock_quantizer_class:
            
            mock_quantizer = mock.Mock()
            mock_quantizer.quantize_awq.return_value = mock_quantized_path
            mock_quantizer_class.return_value = mock_quantizer
            
            main()
            
            # ModelQuantizerが適切に作成されたことを確認
            mock_quantizer_class.assert_called_once_with(temp_model_dir, temp_output_dir)
            
            # AWQ量子化が呼ばれたことを確認
            mock_quantizer.quantize_awq.assert_called_once_with(w_bit=4, group_size=128)
    
    def test_main_gptq_success(self, temp_model_dir, temp_output_dir):
        """main関数GPTQ量子化成功テスト"""
        test_args = [
            'quantize.py',
            '--model-path', temp_model_dir,
            '--method', 'gptq',
            '--bits', '4',
            '--group-size', '128',
            '--output-dir', temp_output_dir
        ]
        
        mock_quantized_path = Path(temp_output_dir) / 'test-gptq-4bit-g128'
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.quantize.ModelQuantizer') as mock_quantizer_class:
            
            mock_quantizer = mock.Mock()
            mock_quantizer.quantize_gptq.return_value = mock_quantized_path
            mock_quantizer_class.return_value = mock_quantizer
            
            main()
            
            # GPTQ量子化が呼ばれたことを確認
            mock_quantizer.quantize_gptq.assert_called_once_with(bits=4, group_size=128)
    
    def test_main_gguf_success(self, temp_model_dir, temp_output_dir):
        """main関数GGUF変換成功テスト"""
        test_args = [
            'quantize.py',
            '--model-path', temp_model_dir,
            '--method', 'gguf',
            '--bits', '4',
            '--output-dir', temp_output_dir
        ]
        
        mock_quantized_path = Path(temp_output_dir) / 'test-q4_k_m.gguf'
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.quantize.ModelQuantizer') as mock_quantizer_class:
            
            mock_quantizer = mock.Mock()
            mock_quantizer.convert_to_gguf.return_value = mock_quantized_path
            mock_quantizer_class.return_value = mock_quantizer
            
            main()
            
            # GGUF変換が呼ばれたことを確認
            mock_quantizer.convert_to_gguf.assert_called_once_with(quantization="q4_k_m")
    
    def test_main_all_methods(self, temp_model_dir, temp_output_dir):
        """main関数全手法実行テスト"""
        test_args = [
            'quantize.py',
            '--model-path', temp_model_dir,
            '--method', 'all',
            '--bits', '4',
            '--output-dir', temp_output_dir
        ]
        
        mock_awq_path = Path(temp_output_dir) / 'test-awq-4bit-g128'
        mock_gptq_path = Path(temp_output_dir) / 'test-gptq-4bit-g128'
        mock_gguf_path = Path(temp_output_dir) / 'test-q4_k_m.gguf'
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.quantize.ModelQuantizer') as mock_quantizer_class:
            
            mock_quantizer = mock.Mock()
            mock_quantizer.quantize_awq.return_value = mock_awq_path
            mock_quantizer.quantize_gptq.return_value = mock_gptq_path
            mock_quantizer.convert_to_gguf.return_value = mock_gguf_path
            mock_quantizer_class.return_value = mock_quantizer
            
            main()
            
            # 全ての手法が呼ばれたことを確認
            mock_quantizer.quantize_awq.assert_called_once()
            mock_quantizer.quantize_gptq.assert_called_once()
            mock_quantizer.convert_to_gguf.assert_called_once()
    
    def test_main_with_benchmark(self, temp_model_dir, temp_output_dir):
        """main関数ベンチマーク付き実行テスト"""
        test_args = [
            'quantize.py',
            '--model-path', temp_model_dir,
            '--method', 'awq',
            '--output-dir', temp_output_dir,
            '--benchmark'
        ]
        
        mock_quantized_path = Path(temp_output_dir) / 'test-awq-4bit-g128'
        mock_quantized_path.mkdir(parents=True)
        
        mock_benchmark_results = {'inference_speed': {'tokens_per_second': 50.0}}
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.quantize.ModelQuantizer') as mock_quantizer_class, \
             mock.patch('builtins.open', mock.mock_open()) as mock_file, \
             mock.patch('json.dump') as mock_json_dump:
            
            mock_quantizer = mock.Mock()
            mock_quantizer.quantize_awq.return_value = mock_quantized_path
            mock_quantizer.benchmark_quantized_model.return_value = mock_benchmark_results
            mock_quantizer_class.return_value = mock_quantizer
            
            main()
            
            # ベンチマークが実行されたことを確認
            mock_quantizer.benchmark_quantized_model.assert_called_once_with(mock_quantized_path)
            
            # ベンチマーク結果が保存されたことを確認
            mock_json_dump.assert_called_once()
    
    def test_main_quantization_error(self, temp_model_dir, temp_output_dir):
        """main関数量子化エラーテスト"""
        test_args = [
            'quantize.py',
            '--model-path', temp_model_dir,
            '--method', 'awq',
            '--output-dir', temp_output_dir
        ]
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.quantize.ModelQuantizer') as mock_quantizer_class, \
             pytest.raises(SystemExit) as exc_info:
            
            mock_quantizer = mock.Mock()
            mock_quantizer.quantize_awq.side_effect = Exception("Quantization failed")
            mock_quantizer_class.return_value = mock_quantizer
            
            main()
        
        assert exc_info.value.code == 1
    
    def test_main_unexpected_error(self, temp_model_dir):
        """main関数予期しないエラーテスト"""
        test_args = [
            'quantize.py',
            '--model-path', temp_model_dir,
            '--method', 'awq'
        ]
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.quantize.ModelQuantizer', side_effect=Exception("Unexpected error")), \
             pytest.raises(SystemExit) as exc_info:
            
            main()
        
        assert exc_info.value.code == 1
    
    def test_main_gguf_bit_mapping(self, temp_model_dir, temp_output_dir):
        """main関数GGUFビットマッピングテスト"""
        # 異なるビット数でのテスト
        test_cases = [
            (2, "q2_k"),
            (3, "q3_k_m"),
            (4, "q4_k_m"),
            (5, "q5_k_m"),
            (6, "q6_k"),
            (8, "q8_0"),
            (7, "q4_k_m"),  # 未定義のビット数はデフォルト
        ]
        
        for bits, expected_quant in test_cases:
            test_args = [
                'quantize.py',
                '--model-path', temp_model_dir,
                '--method', 'gguf',
                '--bits', str(bits),
                '--output-dir', temp_output_dir
            ]
            
            with mock.patch('sys.argv', test_args), \
                 mock.patch('scripts.quantize.ModelQuantizer') as mock_quantizer_class:
                
                mock_quantizer = mock.Mock()
                mock_quantizer.convert_to_gguf.return_value = Path(temp_output_dir) / f'test-{expected_quant}.gguf'
                mock_quantizer_class.return_value = mock_quantizer
                
                main()
                
                # 正しい量子化レベルで呼ばれたことを確認
                mock_quantizer.convert_to_gguf.assert_called_with(quantization=expected_quant)
    
    def test_main_benchmark_error(self, temp_model_dir, temp_output_dir):
        """main関数ベンチマークエラーテスト"""
        test_args = [
            'quantize.py',
            '--model-path', temp_model_dir,
            '--method', 'awq',
            '--output-dir', temp_output_dir,
            '--benchmark'
        ]
        
        mock_quantized_path = Path(temp_output_dir) / 'test-awq-4bit-g128'
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.quantize.ModelQuantizer') as mock_quantizer_class:
            
            mock_quantizer = mock.Mock()
            mock_quantizer.quantize_awq.return_value = mock_quantized_path
            mock_quantizer.benchmark_quantized_model.side_effect = Exception("Benchmark failed")
            mock_quantizer_class.return_value = mock_quantizer
            
            # ベンチマークエラーは致命的ではない
            main()
            
            # ベンチマークが試行されたことを確認
            mock_quantizer.benchmark_quantized_model.assert_called_once()
    
    def test_main_default_arguments(self, temp_model_dir):
        """main関数デフォルト引数テスト"""
        test_args = [
            'quantize.py',
            '--model-path', temp_model_dir
        ]
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.quantize.ModelQuantizer') as mock_quantizer_class:
            
            mock_quantizer = mock.Mock()
            mock_quantizer.quantize_awq.return_value = Path('dummy_path')
            mock_quantizer_class.return_value = mock_quantizer
            
            main()
            
            # デフォルト値での作成確認
            mock_quantizer_class.assert_called_once_with(temp_model_dir, 'models/quantized')
            
            # デフォルトメソッド（AWQ）、ビット数（4）、グループサイズ（128）の確認
            mock_quantizer.quantize_awq.assert_called_once_with(w_bit=4, group_size=128)