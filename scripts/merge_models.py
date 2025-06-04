#!/usr/bin/env python3
"""
LLMモデルマージング実行スクリプト
複数の手法（Slerp, Evolutionary, LoRA）をサポート
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelMerger:
    """複数のマージ手法を統合したモデルマージャー"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.method = self.config['merge_method']
        self.models = self.config['models']
        self.output_path = Path(self.config['output_path'])
        self.device = None  # 遅延初期化
        
        # 実験IDを生成
        self.experiment_id = f"{self.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 遅延インポート用の変数
        self._torch = None
        self._transformers = None
        self._peft = None
        self._numpy = None
        self._tqdm = None
    
    def _lazy_import_torch(self):
        """torchを遅延インポート"""
        if self._torch is None:
            import torch
            self._torch = torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._torch
    
    def _lazy_import_transformers(self):
        """transformersを遅延インポート"""
        if self._transformers is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._transformers = {
                'AutoModelForCausalLM': AutoModelForCausalLM,
                'AutoTokenizer': AutoTokenizer
            }
        return self._transformers
    
    def _lazy_import_peft(self):
        """peftを遅延インポート"""
        if self._peft is None:
            from peft import PeftModel
            self._peft = PeftModel
        return self._peft
    
    def _lazy_import_numpy(self):
        """numpyを遅延インポート"""
        if self._numpy is None:
            import numpy as np
            self._numpy = np
        return self._numpy
    
    def _lazy_import_tqdm(self):
        """tqdmを遅延インポート"""
        if self._tqdm is None:
            from tqdm import tqdm
            self._tqdm = tqdm
        return self._tqdm
        
    def validate_models(self) -> bool:
        """モデルの互換性をチェック"""
        logger.info("モデルの互換性をチェック中...")
        
        torch = self._lazy_import_torch()
        transformers = self._lazy_import_transformers()
        AutoModelForCausalLM = transformers['AutoModelForCausalLM']
        
        configs = []
        for model_spec in self.models:
            model_name = model_spec['name']
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16,
                    device_map="cpu"
                )
                config = model.config
                configs.append({
                    'name': model_name,
                    'hidden_size': config.hidden_size,
                    'num_layers': config.num_hidden_layers,
                    'vocab_size': config.vocab_size
                })
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"モデル {model_name} の読み込みに失敗: {e}")
                return False
        
        # 形状チェック
        if self.method in ['slerp', 'linear']:
            base_config = configs[0]
            for cfg in configs[1:]:
                if (cfg['hidden_size'] != base_config['hidden_size'] or 
                    cfg['num_layers'] != base_config['num_layers']):
                    logger.error(f"モデル形状が不一致: {cfg['name']} vs {base_config['name']}")
                    return False
        
        logger.info("すべてのモデルの互換性確認完了")
        return True
    
    def slerp_merge(self) -> None:
        """Spherical Linear Interpolation (SLERP) マージ"""
        logger.info("SLERP マージを開始...")
        
        torch = self._lazy_import_torch()
        transformers = self._lazy_import_transformers()
        tqdm = self._lazy_import_tqdm()
        
        AutoModelForCausalLM = transformers['AutoModelForCausalLM']
        AutoTokenizer = transformers['AutoTokenizer']
        
        # モデルをロード
        model1 = AutoModelForCausalLM.from_pretrained(
            self.models[0]['name'],
            torch_dtype=torch.float16,
            device_map=self.device
        )
        model2 = AutoModelForCausalLM.from_pretrained(
            self.models[1]['name'],
            torch_dtype=torch.float16,
            device_map=self.device
        )
        
        alpha = self.config.get('alpha', 0.5)
        
        # SLERP実装
        with torch.no_grad():
            for name, param1 in tqdm(model1.named_parameters(), desc="Merging parameters"):
                if name in dict(model2.named_parameters()):
                    param2 = dict(model2.named_parameters())[name]
                    
                    # パラメータを平坦化
                    p1_flat = param1.data.flatten()
                    p2_flat = param2.data.flatten()
                    
                    # コサイン類似度を計算
                    dot_product = torch.dot(p1_flat, p2_flat)
                    norm1 = torch.norm(p1_flat)
                    norm2 = torch.norm(p2_flat)
                    
                    if norm1 > 0 and norm2 > 0:
                        cos_theta = dot_product / (norm1 * norm2)
                        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
                        theta = torch.acos(cos_theta)
                        
                        if theta > 0.01:  # 非常に小さい角度の場合は線形補間
                            sin_theta = torch.sin(theta)
                            w1 = torch.sin((1 - alpha) * theta) / sin_theta
                            w2 = torch.sin(alpha * theta) / sin_theta
                            param1.data = (w1 * p1_flat + w2 * p2_flat).reshape(param1.shape)
                        else:
                            param1.data = ((1 - alpha) * p1_flat + alpha * p2_flat).reshape(param1.shape)
        
        # トークナイザーも統合
        tokenizer1 = AutoTokenizer.from_pretrained(self.models[0]['name'])
        tokenizer2 = AutoTokenizer.from_pretrained(self.models[1]['name'])
        
        # 語彙を統合（重複を除く）
        merged_vocab = dict(tokenizer1.get_vocab())
        for token, idx in tokenizer2.get_vocab().items():
            if token not in merged_vocab:
                merged_vocab[token] = len(merged_vocab)
        
        # 保存
        self.output_path.mkdir(parents=True, exist_ok=True)
        model1.save_pretrained(self.output_path)
        tokenizer1.save_pretrained(self.output_path)
        
        logger.info(f"SLERP マージ完了: {self.output_path}")
    
    def evolutionary_merge(self) -> None:
        """進化的アルゴリズムによるマージ（Sakana AI方式）"""
        logger.info("Evolutionary マージを開始...")
        
        torch = self._lazy_import_torch()
        transformers = self._lazy_import_transformers()
        np = self._lazy_import_numpy()
        tqdm = self._lazy_import_tqdm()
        
        AutoModelForCausalLM = transformers['AutoModelForCausalLM']
        AutoTokenizer = transformers['AutoTokenizer']
        
        # 設定
        population_size = self.config.get('population_size', 20)
        generations = self.config.get('generations', 10)
        mutation_rate = self.config.get('mutation_rate', 0.1)
        
        # モデルをロード
        models = []
        for model_spec in self.models:
            model = AutoModelForCausalLM.from_pretrained(
                model_spec['name'],
                torch_dtype=torch.float16,
                device_map="cpu"
            )
            models.append(model)
        
        # 初期集団を生成（ランダムな重み配分）
        population = []
        for _ in range(population_size):
            weights = np.random.dirichlet(np.ones(len(models)))
            population.append(weights)
        
        # 進化ループ
        best_weights = None
        best_score = -float('inf')
        
        for gen in range(generations):
            logger.info(f"世代 {gen+1}/{generations}")
            
            # 各個体を評価（ここではダミー評価）
            scores = []
            for weights in population:
                # 実際の評価はコストが高いため、ここではランダムスコア
                # 本来はマージしたモデルを評価タスクで測定
                score = np.random.random() + np.sum(weights * weights)
                scores.append(score)
            
            # 最良個体を記録
            best_idx = np.argmax(scores)
            if scores[best_idx] > best_score:
                best_score = scores[best_idx]
                best_weights = population[best_idx]
            
            # 選択と交叉
            new_population = []
            
            # エリート戦略（上位20%を保持）
            elite_count = int(population_size * 0.2)
            elite_indices = np.argsort(scores)[-elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # 残りは交叉と突然変異で生成
            while len(new_population) < population_size:
                # トーナメント選択
                parent1 = population[np.argmax([scores[i] for i in np.random.choice(population_size, 3)])]
                parent2 = population[np.argmax([scores[i] for i in np.random.choice(population_size, 3)])]
                
                # 交叉
                child = (parent1 + parent2) / 2
                child = child / np.sum(child)  # 正規化
                
                # 突然変異
                if np.random.random() < mutation_rate:
                    mutation = np.random.normal(0, 0.1, len(models))
                    child = child + mutation
                    child = np.abs(child)
                    child = child / np.sum(child)  # 正規化
                
                new_population.append(child)
            
            population = new_population
        
        logger.info(f"最適な重み配分: {best_weights}")
        
        # 最良の重みでマージ
        merged_model = models[0]
        with torch.no_grad():
            for name, param in tqdm(merged_model.named_parameters(), desc="Final merge"):
                param.data = torch.zeros_like(param.data)
                for i, model in enumerate(models):
                    if name in dict(model.named_parameters()):
                        param.data += best_weights[i] * dict(model.named_parameters())[name].data
        
        # トークナイザーは最初のモデルのものを使用
        tokenizer = AutoTokenizer.from_pretrained(self.models[0]['name'])
        
        # 保存
        self.output_path.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(self.output_path)
        tokenizer.save_pretrained(self.output_path)
        
        # メモリ解放
        del models
        torch.cuda.empty_cache()
        
        logger.info(f"Evolutionary マージ完了: {self.output_path}")
    
    def lora_merge(self) -> None:
        """LoRA統合"""
        logger.info("LoRA 統合を開始...")
        
        torch = self._lazy_import_torch()
        transformers = self._lazy_import_transformers()
        PeftModel = self._lazy_import_peft()
        
        AutoModelForCausalLM = transformers['AutoModelForCausalLM']
        AutoTokenizer = transformers['AutoTokenizer']
        
        # ベースモデルをロード
        base_model_name = self.models[0]['name']
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        
        # LoRAアダプターを適用
        for model_spec in self.models[1:]:
            if model_spec.get('type') == 'lora':
                logger.info(f"LoRAアダプター適用: {model_spec['name']}")
                base_model = PeftModel.from_pretrained(base_model, model_spec['name'])
                
                # 重み配分を適用（オプション）
                if 'weight' in model_spec:
                    weight = model_spec['weight']
                    # LoRAの重みをスケーリング
                    for name, param in base_model.named_parameters():
                        if 'lora' in name:
                            param.data *= weight
        
        # マージしたモデルを保存
        merged_model = base_model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(self.output_path)
        tokenizer.save_pretrained(self.output_path)
        
        logger.info(f"LoRA 統合完了: {self.output_path}")
    
    def run(self) -> Dict:
        """マージを実行"""
        result = {
            'experiment_id': self.experiment_id,
            'method': self.method,
            'models': self.models,
            'output_path': str(self.output_path),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # モデル検証
            if not self.validate_models():
                result['status'] = 'failed'
                result['error'] = 'Model validation failed'
                return result
            
            # マージ実行
            if self.method == 'slerp':
                self.slerp_merge()
            elif self.method == 'evolutionary':
                self.evolutionary_merge()
            elif self.method == 'lora':
                self.lora_merge()
            else:
                raise ValueError(f"不明なマージ手法: {self.method}")
            
            result['status'] = 'completed'
            
            # メタデータを保存
            metadata_path = self.output_path / 'merge_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"マージ中にエラーが発生: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
        
        return result


def main():
    parser = argparse.ArgumentParser(description='LLMモデルマージング')
    parser.add_argument('--config', required=True, help='マージ設定ファイル（YAML）')
    parser.add_argument('--device', default='auto', help='使用デバイス')
    
    args = parser.parse_args()
    
    # 設定ファイルの存在確認
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        sys.exit(1)
    
    # マージ実行
    merger = ModelMerger(str(config_path))
    result = merger.run()
    
    # 結果を表示
    print("\n" + "="*60)
    print("マージ結果:")
    print(f"実験ID: {result['experiment_id']}")
    print(f"状態: {result['status']}")
    print(f"出力先: {result['output_path']}")
    
    if result['status'] == 'failed':
        print(f"エラー: {result.get('error', 'Unknown error')}")
        sys.exit(1)
    else:
        print("✅ マージが正常に完了しました")
        print("="*60)


if __name__ == '__main__':
    main()