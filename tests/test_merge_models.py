#!/usr/bin/env python3
"""
モデルマージ機能のテスト
"""

import pytest
import tempfile
import yaml
import json
from pathlib import Path
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from scripts.merge_models import ModelMerger


class TestModelMerger:
    """ModelMergerクラスのテスト"""
    
    @pytest.fixture
    def temp_config(self):
        """テスト用の設定ファイルを作成"""
        config = {
            'merge_method': 'slerp',
            'output_path': 'models/test_merged_model',
            'models': [
                {'name': 'microsoft/DialoGPT-small', 'weight': 0.6},
                {'name': 'microsoft/DialoGPT-small', 'weight': 0.4}  # 同じモデルでテスト
            ],
            'alpha': 0.6
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            return f.name
    
    def test_config_loading(self, temp_config):
        """設定ファイルの読み込みテスト"""
        merger = ModelMerger(temp_config)
        
        assert merger.config['merge_method'] == 'slerp'
        assert merger.config['alpha'] == 0.6
        assert len(merger.config['models']) == 2
        
        # クリーンアップ
        os.unlink(temp_config)
    
    def test_experiment_id_generation(self, temp_config):
        """実験IDの生成テスト"""
        merger = ModelMerger(temp_config)
        
        assert merger.experiment_id.startswith('slerp_')
        assert len(merger.experiment_id) > 10
        
        # クリーンアップ
        os.unlink(temp_config)
    
    @pytest.mark.slow
    def test_validate_models_fail(self, temp_config):
        """無効なモデルの検証テスト"""
        # 存在しないモデルでテスト
        config = {
            'merge_method': 'slerp',
            'output_path': 'models/test_merged_model',
            'models': [
                {'name': 'nonexistent/model1', 'weight': 0.6},
                {'name': 'nonexistent/model2', 'weight': 0.4}
            ],
            'alpha': 0.6
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            merger = ModelMerger(config_path)
            # 存在しないモデルなので検証は失敗するはず
            assert not merger.validate_models()
        finally:
            os.unlink(config_path)
    
    def test_init_parameters(self, temp_config):
        """初期化パラメータのテスト"""
        merger = ModelMerger(temp_config)
        
        assert merger.method == 'slerp'
        assert merger.models[0]['name'] == 'microsoft/DialoGPT-small'
        assert merger.output_path == Path('models/test_merged_model')
        assert merger.device is None  # 遅延初期化
        assert merger.experiment_id.startswith('slerp_')
        
        # 遅延インポート用変数の初期化確認
        assert merger._torch is None
        assert merger._transformers is None
        assert merger._peft is None
        assert merger._numpy is None
        assert merger._tqdm is None
        
        os.unlink(temp_config)
    
    def test_lazy_import_torch(self, temp_config):
        """torchの遅延インポートテスト"""
        merger = ModelMerger(temp_config)
        
        with mock.patch('torch.cuda.is_available', return_value=False):
            torch = merger._lazy_import_torch()
            
            assert merger._torch is not None
            assert torch == merger._torch
            assert str(merger.device) == 'cpu'
            
            # 2回目の呼び出しでも同じオブジェクトが返される
            torch2 = merger._lazy_import_torch()
            assert torch == torch2
        
        os.unlink(temp_config)
    
    def test_lazy_import_torch_cuda(self, temp_config):
        """CUDA利用可能時のtorchの遅延インポートテスト"""
        merger = ModelMerger(temp_config)
        
        with mock.patch('torch.cuda.is_available', return_value=True):
            torch = merger._lazy_import_torch()
            
            assert str(merger.device) == 'cuda'
        
        os.unlink(temp_config)
    
    def test_lazy_import_transformers(self, temp_config):
        """transformersの遅延インポートテスト"""
        merger = ModelMerger(temp_config)
        
        transformers = merger._lazy_import_transformers()
        
        assert merger._transformers is not None
        assert 'AutoModelForCausalLM' in transformers
        assert 'AutoTokenizer' in transformers
        
        # 2回目の呼び出しでも同じオブジェクトが返される
        transformers2 = merger._lazy_import_transformers()
        assert transformers == transformers2
        
        os.unlink(temp_config)
    
    def test_lazy_import_peft(self, temp_config):
        """peftの遅延インポートテスト"""
        merger = ModelMerger(temp_config)
        
        peft = merger._lazy_import_peft()
        
        assert merger._peft is not None
        assert peft == merger._peft
        
        # 2回目の呼び出しでも同じオブジェクトが返される
        peft2 = merger._lazy_import_peft()
        assert peft == peft2
        
        os.unlink(temp_config)
    
    def test_lazy_import_numpy(self, temp_config):
        """numpyの遅延インポートテスト"""
        merger = ModelMerger(temp_config)
        
        np = merger._lazy_import_numpy()
        
        assert merger._numpy is not None
        assert np == merger._numpy
        
        # 2回目の呼び出しでも同じオブジェクトが返される
        np2 = merger._lazy_import_numpy()
        assert np == np2
        
        os.unlink(temp_config)
    
    def test_lazy_import_tqdm(self, temp_config):
        """tqdmの遅延インポートテスト"""
        merger = ModelMerger(temp_config)
        
        tqdm = merger._lazy_import_tqdm()
        
        assert merger._tqdm is not None
        assert tqdm == merger._tqdm
        
        # 2回目の呼び出しでも同じオブジェクトが返される
        tqdm2 = merger._lazy_import_tqdm()
        assert tqdm == tqdm2
        
        os.unlink(temp_config)
    
    def test_validate_models_success(self, temp_config):
        """モデル検証成功テスト"""
        merger = ModelMerger(temp_config)
        
        # モックモデルと設定
        mock_config = mock.Mock()
        mock_config.hidden_size = 768
        mock_config.num_hidden_layers = 12
        mock_config.vocab_size = 50257
        
        mock_model = mock.Mock()
        mock_model.config = mock_config
        
        with mock.patch.object(merger, '_lazy_import_torch') as mock_torch, \
             mock.patch.object(merger, '_lazy_import_transformers') as mock_transformers:
            
            mock_torch.return_value.cuda.empty_cache = mock.Mock()
            mock_transformers.return_value = {
                'AutoModelForCausalLM': mock.Mock()
            }
            mock_transformers.return_value['AutoModelForCausalLM'].from_pretrained.return_value = mock_model
            
            result = merger.validate_models()
            
            assert result is True
            # from_pretrainedが各モデルに対して呼ばれる
            assert mock_transformers.return_value['AutoModelForCausalLM'].from_pretrained.call_count == 2
        
        os.unlink(temp_config)
    
    def test_validate_models_load_failure(self, temp_config):
        """モデルロード失敗時の検証テスト"""
        merger = ModelMerger(temp_config)
        
        with mock.patch.object(merger, '_lazy_import_torch') as mock_torch, \
             mock.patch.object(merger, '_lazy_import_transformers') as mock_transformers:
            
            mock_transformers.return_value = {
                'AutoModelForCausalLM': mock.Mock()
            }
            mock_transformers.return_value['AutoModelForCausalLM'].from_pretrained.side_effect = Exception("Model load error")
            
            result = merger.validate_models()
            
            assert result is False
        
        os.unlink(temp_config)
    
    def test_validate_models_shape_mismatch(self, temp_config):
        """モデル形状不一致時の検証テスト"""
        merger = ModelMerger(temp_config)
        
        # 異なる形状のモデル設定
        mock_config1 = mock.Mock()
        mock_config1.hidden_size = 768
        mock_config1.num_hidden_layers = 12
        mock_config1.vocab_size = 50257
        
        mock_config2 = mock.Mock()
        mock_config2.hidden_size = 1024  # 異なるサイズ
        mock_config2.num_hidden_layers = 12
        mock_config2.vocab_size = 50257
        
        mock_model1 = mock.Mock()
        mock_model1.config = mock_config1
        
        mock_model2 = mock.Mock()
        mock_model2.config = mock_config2
        
        with mock.patch.object(merger, '_lazy_import_torch') as mock_torch, \
             mock.patch.object(merger, '_lazy_import_transformers') as mock_transformers:
            
            mock_torch.return_value.cuda.empty_cache = mock.Mock()
            mock_transformers.return_value = {
                'AutoModelForCausalLM': mock.Mock()
            }
            mock_transformers.return_value['AutoModelForCausalLM'].from_pretrained.side_effect = [mock_model1, mock_model2]
            
            result = merger.validate_models()
            
            assert result is False
        
        os.unlink(temp_config)
    
    def test_validate_models_non_slerp_method(self):
        """
non-slerp手法でのモデル検証テスト"""
        config = {
            'merge_method': 'evolutionary',  # slerpではない
            'output_path': 'models/test_merged_model',
            'models': [
                {'name': 'model1', 'weight': 0.6},
                {'name': 'model2', 'weight': 0.4}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            merger = ModelMerger(config_path)
            
            # 異なる形状でもevolutionaryではエラーにならない
            mock_config1 = mock.Mock()
            mock_config1.hidden_size = 768
            mock_config1.num_hidden_layers = 12
            mock_config1.vocab_size = 50257
            
            mock_config2 = mock.Mock()
            mock_config2.hidden_size = 1024  # 異なるサイズ
            mock_config2.num_hidden_layers = 12
            mock_config2.vocab_size = 50257
            
            mock_model1 = mock.Mock()
            mock_model1.config = mock_config1
            
            mock_model2 = mock.Mock()
            mock_model2.config = mock_config2
            
            with mock.patch.object(merger, '_lazy_import_torch') as mock_torch, \
                 mock.patch.object(merger, '_lazy_import_transformers') as mock_transformers:
                
                mock_torch.return_value.cuda.empty_cache = mock.Mock()
                mock_transformers.return_value = {
                    'AutoModelForCausalLM': mock.Mock()
                }
                mock_transformers.return_value['AutoModelForCausalLM'].from_pretrained.side_effect = [mock_model1, mock_model2]
                
                result = merger.validate_models()
                
                # evolutionary手法では形状チェックをスキップする
                assert result is True
        
        finally:
            os.unlink(config_path)
    
    @pytest.fixture
    def temp_output_dir(self):
        """テスト用出力ディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_slerp_merge(self, temp_config, temp_output_dir):
        """SLERPマージテスト"""
        # 出力パスを一時ディレクトリに変更
        with open(temp_config, 'r') as f:
            config = yaml.safe_load(f)
        config['output_path'] = temp_output_dir
        
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        merger = ModelMerger(temp_config)
        
        # モックオブジェクトを作成
        mock_param1 = mock.Mock()
        mock_param1.data = mock.Mock()
        mock_param1.data.flatten.return_value = mock.Mock()
        mock_param1.shape = (10, 10)
        
        mock_param2 = mock.Mock()
        mock_param2.data = mock.Mock()
        mock_param2.data.flatten.return_value = mock.Mock()
        
        mock_model1 = mock.Mock()
        mock_model1.named_parameters.return_value = [('layer1.weight', mock_param1)]
        mock_model1.save_pretrained = mock.Mock()
        
        mock_model2 = mock.Mock()
        mock_model2.named_parameters.return_value = [('layer1.weight', mock_param2)]
        
        mock_tokenizer1 = mock.Mock()
        mock_tokenizer1.get_vocab.return_value = {'token1': 0, 'token2': 1}
        mock_tokenizer1.save_pretrained = mock.Mock()
        
        mock_tokenizer2 = mock.Mock()
        mock_tokenizer2.get_vocab.return_value = {'token1': 0, 'token3': 2}
        
        with mock.patch.object(merger, '_lazy_import_torch') as mock_torch, \
             mock.patch.object(merger, '_lazy_import_transformers') as mock_transformers, \
             mock.patch.object(merger, '_lazy_import_tqdm') as mock_tqdm:
            
            # torchのモック
            mock_torch_obj = mock.Mock()
            mock_torch_obj.no_grad.return_value.__enter__ = mock.Mock()
            mock_torch_obj.no_grad.return_value.__exit__ = mock.Mock()
            mock_torch_obj.dot.return_value = 0.5
            mock_torch_obj.norm.return_value = 1.0
            mock_torch_obj.clamp.return_value = 0.5
            mock_torch_obj.acos.return_value = 1.0
            mock_torch_obj.sin.return_value = 0.8
            mock_torch.return_value = mock_torch_obj
            
            # transformersのモック
            mock_transformers.return_value = {
                'AutoModelForCausalLM': mock.Mock(),
                'AutoTokenizer': mock.Mock()
            }
            mock_transformers.return_value['AutoModelForCausalLM'].from_pretrained.side_effect = [mock_model1, mock_model2]
            mock_transformers.return_value['AutoTokenizer'].from_pretrained.side_effect = [mock_tokenizer1, mock_tokenizer2]
            
            # tqdmのモック
            mock_tqdm.return_value = mock.Mock()
            mock_tqdm.return_value.return_value = [('layer1.weight', mock_param1)]
            
            merger.slerp_merge()
            
            # モデルとトークナイザーの保存が呼ばれたことを確認
            mock_model1.save_pretrained.assert_called_once()
            mock_tokenizer1.save_pretrained.assert_called_once()
        
        os.unlink(temp_config)
    
    def test_evolutionary_merge(self, temp_config, temp_output_dir):
        """Evolutionaryマージテスト"""
        # 出力パスを一時ディレクトリに変更
        with open(temp_config, 'r') as f:
            config = yaml.safe_load(f)
        config['output_path'] = temp_output_dir
        config['merge_method'] = 'evolutionary'
        config['population_size'] = 5  # テスト用に小さく
        config['generations'] = 2
        config['mutation_rate'] = 0.1
        
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        merger = ModelMerger(temp_config)
        
        # モックモデルとパラメータ
        mock_param = mock.Mock()
        mock_param.data = mock.Mock()
        
        mock_model1 = mock.Mock()
        mock_model1.named_parameters.return_value = [('layer1.weight', mock_param)]
        mock_model1.save_pretrained = mock.Mock()
        
        mock_model2 = mock.Mock()
        mock_model2.named_parameters.return_value = [('layer1.weight', mock_param)]
        
        mock_tokenizer = mock.Mock()
        mock_tokenizer.save_pretrained = mock.Mock()
        
        with mock.patch.object(merger, '_lazy_import_torch') as mock_torch, \
             mock.patch.object(merger, '_lazy_import_transformers') as mock_transformers, \
             mock.patch.object(merger, '_lazy_import_numpy') as mock_np, \
             mock.patch.object(merger, '_lazy_import_tqdm') as mock_tqdm:
            
            # torchのモック
            mock_torch_obj = mock.Mock()
            mock_torch_obj.no_grad.return_value.__enter__ = mock.Mock()
            mock_torch_obj.no_grad.return_value.__exit__ = mock.Mock()
            mock_torch_obj.zeros_like.return_value = mock.Mock()
            mock_torch_obj.cuda.empty_cache = mock.Mock()
            mock_torch.return_value = mock_torch_obj
            
            # numpyのモック
            mock_np_obj = mock.Mock()
            mock_np_obj.random.dirichlet.return_value = [0.5, 0.5]
            mock_np_obj.random.random.return_value = 0.7
            mock_np_obj.sum.return_value = 1.0
            mock_np_obj.argmax.return_value = 0
            mock_np_obj.argsort.return_value = [0, 1]
            mock_np_obj.random.choice.return_value = [0, 1]
            mock_np_obj.random.normal.return_value = [0.1, 0.1]
            mock_np_obj.abs.return_value = [0.6, 0.4]
            mock_np.return_value = mock_np_obj
            
            # transformersのモック
            mock_transformers.return_value = {
                'AutoModelForCausalLM': mock.Mock(),
                'AutoTokenizer': mock.Mock()
            }
            mock_transformers.return_value['AutoModelForCausalLM'].from_pretrained.side_effect = [mock_model1, mock_model2]
            mock_transformers.return_value['AutoTokenizer'].from_pretrained.return_value = mock_tokenizer
            
            # tqdmのモック
            mock_tqdm.return_value = mock.Mock()
            mock_tqdm.return_value.return_value = [('layer1.weight', mock_param)]
            
            merger.evolutionary_merge()
            
            # モデルとトークナイザーの保存が呼ばれたことを確認
            mock_model1.save_pretrained.assert_called_once()
            mock_tokenizer.save_pretrained.assert_called_once()
        
        os.unlink(temp_config)
    
    def test_lora_merge(self, temp_config, temp_output_dir):
        """LoRAマージテスト"""
        # LoRA用の設定を作成
        config = {
            'merge_method': 'lora',
            'output_path': temp_output_dir,
            'models': [
                {'name': 'base_model', 'type': 'base'},
                {'name': 'lora_adapter1', 'type': 'lora', 'weight': 0.8},
                {'name': 'lora_adapter2', 'type': 'lora', 'weight': 0.6}
            ]
        }
        
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        merger = ModelMerger(temp_config)
        
        # モックオブジェクト
        mock_base_model = mock.Mock()
        mock_peft_model = mock.Mock()
        mock_peft_model.named_parameters.return_value = [('lora_layer.weight', mock.Mock())]
        mock_merged_model = mock.Mock()
        mock_merged_model.save_pretrained = mock.Mock()
        
        mock_tokenizer = mock.Mock()
        mock_tokenizer.save_pretrained = mock.Mock()
        
        with mock.patch.object(merger, '_lazy_import_torch') as mock_torch, \
             mock.patch.object(merger, '_lazy_import_transformers') as mock_transformers, \
             mock.patch.object(merger, '_lazy_import_peft') as mock_peft:
            
            # torchのモック
            mock_torch.return_value = mock.Mock()
            
            # transformersのモック
            mock_transformers.return_value = {
                'AutoModelForCausalLM': mock.Mock(),
                'AutoTokenizer': mock.Mock()
            }
            mock_transformers.return_value['AutoModelForCausalLM'].from_pretrained.return_value = mock_base_model
            mock_transformers.return_value['AutoTokenizer'].from_pretrained.return_value = mock_tokenizer
            
            # peftのモック
            mock_peft.return_value.from_pretrained.return_value = mock_peft_model
            mock_peft_model.merge_and_unload.return_value = mock_merged_model
            
            merger.lora_merge()
            
            # メソッドの呼び出しを確認
            mock_peft.return_value.from_pretrained.assert_called()
            mock_peft_model.merge_and_unload.assert_called_once()
            mock_merged_model.save_pretrained.assert_called_once()
            mock_tokenizer.save_pretrained.assert_called_once()
        
        os.unlink(temp_config)
    
    def test_run_success(self, temp_config, temp_output_dir):
        """正常実行テスト"""
        # 出力パスを一時ディレクトリに変更
        with open(temp_config, 'r') as f:
            config = yaml.safe_load(f)
        config['output_path'] = temp_output_dir
        
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        merger = ModelMerger(temp_config)
        
        with mock.patch.object(merger, 'validate_models', return_value=True), \
             mock.patch.object(merger, 'slerp_merge') as mock_slerp:
            
            result = merger.run()
            
            # 結果の確認
            assert result['status'] == 'completed'
            assert result['method'] == 'slerp'
            assert result['experiment_id'].startswith('slerp_')
            assert 'timestamp' in result
            
            # slerp_mergeが呼ばれたことを確認
            mock_slerp.assert_called_once()
            
            # メタデータファイルが作成されたことを確認
            metadata_path = Path(temp_output_dir) / 'merge_metadata.json'
            assert metadata_path.exists()
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                assert metadata['status'] == 'completed'
        
        os.unlink(temp_config)
    
    def test_run_validation_failure(self, temp_config):
        """検証失敗時の実行テスト"""
        merger = ModelMerger(temp_config)
        
        with mock.patch.object(merger, 'validate_models', return_value=False):
            result = merger.run()
            
            assert result['status'] == 'failed'
            assert result['error'] == 'Model validation failed'
        
        os.unlink(temp_config)
    
    def test_run_unknown_method(self, temp_config):
        """不明なマージ手法時のテスト"""
        # 不明な手法を設定
        with open(temp_config, 'r') as f:
            config = yaml.safe_load(f)
        config['merge_method'] = 'unknown_method'
        
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        merger = ModelMerger(temp_config)
        
        with mock.patch.object(merger, 'validate_models', return_value=True):
            result = merger.run()
            
            assert result['status'] == 'failed'
            assert '不明なマージ手法' in result['error']
        
        os.unlink(temp_config)
    
    def test_run_merge_exception(self, temp_config):
        """マージ中の例外処理テスト"""
        merger = ModelMerger(temp_config)
        
        with mock.patch.object(merger, 'validate_models', return_value=True), \
             mock.patch.object(merger, 'slerp_merge', side_effect=Exception("Merge error")):
            
            result = merger.run()
            
            assert result['status'] == 'failed'
            assert result['error'] == 'Merge error'
        
        os.unlink(temp_config)
    
    def test_run_evolutionary_method(self, temp_config):
        """evolutionary手法の実行テスト"""
        # evolutionary手法を設定
        with open(temp_config, 'r') as f:
            config = yaml.safe_load(f)
        config['merge_method'] = 'evolutionary'
        
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        merger = ModelMerger(temp_config)
        
        with mock.patch.object(merger, 'validate_models', return_value=True), \
             mock.patch.object(merger, 'evolutionary_merge') as mock_evolutionary:
            
            result = merger.run()
            
            assert result['status'] == 'completed'
            assert result['method'] == 'evolutionary'
            mock_evolutionary.assert_called_once()
        
        os.unlink(temp_config)
    
    def test_run_lora_method(self, temp_config):
        """lora手法の実行テスト"""
        # lora手法を設定
        with open(temp_config, 'r') as f:
            config = yaml.safe_load(f)
        config['merge_method'] = 'lora'
        
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        merger = ModelMerger(temp_config)
        
        with mock.patch.object(merger, 'validate_models', return_value=True), \
             mock.patch.object(merger, 'lora_merge') as mock_lora:
            
            result = merger.run()
            
            assert result['status'] == 'completed'
            assert result['method'] == 'lora'
            mock_lora.assert_called_once()
        
        os.unlink(temp_config)


class TestMainFunction:
    """main関数のテスト"""
    
    @pytest.fixture
    def temp_config_file(self):
        """テスト用設定ファイル"""
        config = {
            'merge_method': 'slerp',
            'output_path': 'models/test_merged_model',
            'models': [
                {'name': 'model1', 'weight': 0.6},
                {'name': 'model2', 'weight': 0.4}
            ],
            'alpha': 0.6
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            yield f.name
        
        os.unlink(f.name)
    
    def test_main_success(self, temp_config_file):
        """main関数正常実行テスト"""
        test_args = [
            'merge_models.py',
            '--config', temp_config_file,
            '--device', 'cpu'
        ]
        
        mock_result = {
            'experiment_id': 'test_exp_123',
            'status': 'completed',
            'output_path': '/test/output',
            'method': 'slerp'
        }
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.merge_models.ModelMerger') as mock_merger_class, \
             mock.patch('builtins.print') as mock_print:
            
            mock_merger = mock.Mock()
            mock_merger.run.return_value = mock_result
            mock_merger_class.return_value = mock_merger
            
            main()
            
            # ModelMergerが適切に作成されたことを確認
            mock_merger_class.assert_called_once_with(temp_config_file)
            
            # runメソッドが呼ばれたことを確認
            mock_merger.run.assert_called_once()
            
            # 結果が出力されたことを確認
            assert mock_print.call_count >= 4  # 複数のprint文
    
    def test_main_config_not_found(self):
        """設定ファイルが存在しない場合のテスト"""
        test_args = [
            'merge_models.py',
            '--config', '/nonexistent/config.yaml'
        ]
        
        with mock.patch('sys.argv', test_args), \
             pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1
    
    def test_main_merge_failure(self, temp_config_file):
        """マージ失敗時のテスト"""
        test_args = [
            'merge_models.py',
            '--config', temp_config_file
        ]
        
        mock_result = {
            'experiment_id': 'test_exp_123',
            'status': 'failed',
            'error': 'Test error message',
            'output_path': '/test/output'
        }
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.merge_models.ModelMerger') as mock_merger_class, \
             mock.patch('builtins.print') as mock_print, \
             pytest.raises(SystemExit) as exc_info:
            
            mock_merger = mock.Mock()
            mock_merger.run.return_value = mock_result
            mock_merger_class.return_value = mock_merger
            
            main()
        
        assert exc_info.value.code == 1
        # エラーメッセージが出力されたことを確認
        assert mock_print.call_count >= 4
    
    def test_main_default_device(self, temp_config_file):
        """デフォルトデバイス設定テスト"""
        test_args = [
            'merge_models.py',
            '--config', temp_config_file
            # --device を指定しない
        ]
        
        mock_result = {
            'experiment_id': 'test_exp_123',
            'status': 'completed',
            'output_path': '/test/output',
            'method': 'slerp'
        }
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.merge_models.ModelMerger') as mock_merger_class, \
             mock.patch('builtins.print'):
            
            mock_merger = mock.Mock()
            mock_merger.run.return_value = mock_result
            mock_merger_class.return_value = mock_merger
            
            main()
            
            # デフォルトでModelMergerが作成されたことを確誋
            mock_merger_class.assert_called_once_with(temp_config_file)
    
    def test_config_file_loading_error(self):
        """設定ファイル読み込みエラーテスト"""
        # 無効なYAMLファイルを作成
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content: [}')
            invalid_config = f.name
        
        test_args = [
            'merge_models.py',
            '--config', invalid_config
        ]
        
        try:
            with mock.patch('sys.argv', test_args), \
                 pytest.raises(yaml.YAMLError):
                main()
        finally:
            os.unlink(invalid_config)