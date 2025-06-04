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