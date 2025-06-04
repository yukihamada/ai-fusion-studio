#!/usr/bin/env python3
"""
モデル量子化スクリプト
AWQ, GPTQ, GGUF形式への変換をサポート
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
try:
    from awq import AutoAWQForCausalLM
except ImportError:
    AutoAWQForCausalLM = None

try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
except ImportError:
    AutoGPTQForCausalLM = None
    BaseQuantizeConfig = None
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelQuantizer:
    """モデル量子化クラス"""
    
    def __init__(self, model_path: str, output_dir: str = "models/quantized"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ValueError(f"モデルパスが見つかりません: {model_path}")
            
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # モデル情報を取得
            self.config = AutoConfig.from_pretrained(model_path)
            self.model_size_gb = self._estimate_model_size()
            
            logger.info(f"モデル: {model_path}")
            logger.info(f"推定サイズ: {self.model_size_gb:.2f} GB")
        except Exception as e:
            logger.error(f"モデル設定の読み込みに失敗しました: {e}")
            raise
    
    def _estimate_model_size(self) -> float:
        """モデルサイズを推定（GB）"""
        # パラメータ数から推定
        num_params = self.config.num_hidden_layers * (
            4 * self.config.hidden_size * self.config.intermediate_size +
            4 * self.config.hidden_size * self.config.hidden_size
        )
        # FP16での推定サイズ
        size_bytes = num_params * 2
        return size_bytes / (1024**3)
    
    def quantize_awq(self, w_bit: int = 4, group_size: int = 128) -> Path:
        """AWQ (Activation-aware Weight Quantization) による量子化"""
        if AutoAWQForCausalLM is None:
            raise ImportError("AWQライブラリがインストールされていません。'pip install autoawq'を実行してください。")
            
        logger.info(f"AWQ量子化を開始 (w_bit={w_bit}, group_size={group_size})")
        
        output_path = self.output_dir / f"{self.model_path.name}-awq-{w_bit}bit-g{group_size}"
        
        try:
            # モデルをロード
            model = AutoAWQForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # キャリブレーションデータを準備
            calibration_data = self._prepare_calibration_data(tokenizer)
            
            # 量子化設定
            quant_config = {
                "zero_point": True,
                "q_group_size": group_size,
                "w_bit": w_bit,
                "version": "GEMM"
            }
            
            # 量子化実行
            logger.info("モデルを量子化中...")
            model.quantize(
                tokenizer,
                quant_config=quant_config,
                calib_data=calibration_data
            )
            
            # 保存
            logger.info(f"量子化モデルを保存中: {output_path}")
            model.save_quantized(output_path)
            tokenizer.save_pretrained(output_path)
            
            # 量子化後のサイズを計算
            quantized_size = self._calculate_directory_size(output_path)
            compression_ratio = self.model_size_gb / quantized_size
            
            logger.info(f"AWQ量子化完了: {output_path}")
            logger.info(f"量子化後サイズ: {quantized_size:.2f} GB (圧縮率: {compression_ratio:.2f}x)")
            
            # メタデータを保存
            self._save_quantization_metadata(output_path, {
                'method': 'awq',
                'w_bit': w_bit,
                'group_size': group_size,
                'original_size_gb': self.model_size_gb,
                'quantized_size_gb': quantized_size,
                'compression_ratio': compression_ratio
            })
            
            return output_path
            
        except Exception as e:
            logger.error(f"AWQ量子化に失敗: {e}")
            raise
    
    def quantize_gptq(self, bits: int = 4, group_size: int = 128, 
                     act_order: bool = True) -> Path:
        """GPTQ (Gradient-based Post-training Quantization) による量子化"""
        if AutoGPTQForCausalLM is None or BaseQuantizeConfig is None:
            raise ImportError("Auto-GPTQライブラリがインストールされていません。'pip install auto-gptq'を実行してください。")
            
        logger.info(f"GPTQ量子化を開始 (bits={bits}, group_size={group_size})")
        
        output_path = self.output_dir / f"{self.model_path.name}-gptq-{bits}bit-g{group_size}"
        
        try:
            # 量子化設定
            quantize_config = BaseQuantizeConfig(
                bits=bits,
                group_size=group_size,
                damp_percent=0.01,
                desc_act=act_order,
                static_groups=False,
                sym=True,
                true_sequential=True,
                model_name_or_path=None,
                model_file_base_name="model"
            )
            
            # モデルをロード
            model = AutoGPTQForCausalLM.from_pretrained(
                self.model_path,
                quantize_config=quantize_config,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # キャリブレーションデータを準備
            calibration_data = self._prepare_calibration_data(tokenizer)
            
            # 量子化実行
            logger.info("モデルを量子化中...")
            model.quantize(
                calibration_data,
                batch_size=1,
                use_triton=False
            )
            
            # 保存
            logger.info(f"量子化モデルを保存中: {output_path}")
            model.save_quantized(output_path)
            tokenizer.save_pretrained(output_path)
            
            # 量子化後のサイズを計算
            quantized_size = self._calculate_directory_size(output_path)
            compression_ratio = self.model_size_gb / quantized_size
            
            logger.info(f"GPTQ量子化完了: {output_path}")
            logger.info(f"量子化後サイズ: {quantized_size:.2f} GB (圧縮率: {compression_ratio:.2f}x)")
            
            # メタデータを保存
            self._save_quantization_metadata(output_path, {
                'method': 'gptq',
                'bits': bits,
                'group_size': group_size,
                'act_order': act_order,
                'original_size_gb': self.model_size_gb,
                'quantized_size_gb': quantized_size,
                'compression_ratio': compression_ratio
            })
            
            return output_path
            
        except Exception as e:
            logger.error(f"GPTQ量子化に失敗: {e}")
            raise
    
    def convert_to_gguf(self, quantization: str = "q4_k_m") -> Path:
        """GGUF形式への変換（llama.cpp用）"""
        logger.info(f"GGUF変換を開始 (quantization={quantization})")
        
        output_path = self.output_dir / f"{self.model_path.name}-{quantization}.gguf"
        
        try:
            # 一時ディレクトリにモデルをコピー
            temp_dir = self.output_dir / "temp_gguf_conversion"
            temp_dir.mkdir(exist_ok=True)
            
            # llama.cppの変換スクリプトを使用
            # 注: llama.cppがインストールされている必要があります
            convert_script = "convert.py"  # llama.cppのconvert.pyパス
            quantize_exe = "quantize"      # llama.cppのquantizeバイナリパス
            
            # Step 1: HuggingFace形式をGGML形式に変換
            logger.info("HuggingFace形式をGGML形式に変換中...")
            ggml_path = temp_dir / "model.ggml"
            
            cmd = [
                "python", convert_script,
                str(self.model_path),
                "--outfile", str(ggml_path),
                "--outtype", "f16"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"GGML変換に失敗: {result.stderr}")
            
            # Step 2: 量子化
            logger.info(f"量子化中 ({quantization})...")
            cmd = [
                quantize_exe,
                str(ggml_path),
                str(output_path),
                quantization
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"量子化に失敗: {result.stderr}")
            
            # クリーンアップ
            shutil.rmtree(temp_dir)
            
            # 量子化後のサイズを確認
            quantized_size = output_path.stat().st_size / (1024**3)
            compression_ratio = self.model_size_gb / quantized_size
            
            logger.info(f"GGUF変換完了: {output_path}")
            logger.info(f"量子化後サイズ: {quantized_size:.2f} GB (圧縮率: {compression_ratio:.2f}x)")
            
            # メタデータを保存
            self._save_quantization_metadata(output_path.parent / f"{output_path.stem}_metadata.json", {
                'method': 'gguf',
                'quantization': quantization,
                'original_size_gb': self.model_size_gb,
                'quantized_size_gb': quantized_size,
                'compression_ratio': compression_ratio
            })
            
            return output_path
            
        except Exception as e:
            logger.error(f"GGUF変換に失敗: {e}")
            logger.info("llama.cppがインストールされていることを確認してください")
            raise
    
    def _prepare_calibration_data(self, tokenizer, num_samples: int = 128) -> list:
        """量子化用のキャリブレーションデータを準備"""
        logger.info("キャリブレーションデータを準備中...")
        
        # 日本語のサンプルテキスト
        calibration_texts = [
            "人工知能の発展により、私たちの生活は大きく変化しています。",
            "機械学習モデルの性能向上には、良質なデータセットが不可欠です。",
            "自然言語処理技術は、翻訳や要約などの様々なタスクで活用されています。",
            "深層学習の進歩により、画像認識の精度が飛躍的に向上しました。",
            "量子コンピュータは、従来のコンピュータでは解けない問題を解決する可能性があります。",
            "持続可能な社会の実現には、再生可能エネルギーの活用が重要です。",
            "医療分野では、AIを活用した診断支援システムが開発されています。",
            "教育現場でも、個別最適化された学習システムの導入が進んでいます。",
        ]
        
        # より多くのサンプルが必要な場合は繰り返す
        calibration_data = []
        for i in range(num_samples):
            text = calibration_texts[i % len(calibration_texts)]
            tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            calibration_data.append(tokens)
        
        return calibration_data
    
    def _calculate_directory_size(self, directory: Path) -> float:
        """ディレクトリのサイズを計算（GB）"""
        total_size = 0
        for path in directory.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size / (1024**3)
    
    def _save_quantization_metadata(self, path: Path, metadata: Dict) -> None:
        """量子化のメタデータを保存"""
        metadata['timestamp'] = datetime.now().isoformat()
        metadata['original_model'] = str(self.model_path)
        
        if path.is_dir():
            metadata_path = path / "quantization_info.json"
        else:
            metadata_path = path
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def benchmark_quantized_model(self, quantized_path: Path) -> Dict:
        """量子化モデルのベンチマーク"""
        logger.info(f"量子化モデルのベンチマークを実行: {quantized_path}")
        
        results = {
            'model_path': str(quantized_path),
            'benchmarks': {}
        }
        
        # 推論速度のベンチマーク
        try:
            model = AutoModelForCausalLM.from_pretrained(
                quantized_path,
                device_map="auto",
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(quantized_path)
            
            # テストプロンプト
            test_prompts = [
                "こんにちは、今日は",
                "人工知能について説明してください。",
                "東京の天気は"
            ]
            
            import time
            total_time = 0
            total_tokens = 0
            
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=False
                    )
                end_time = time.time()
                
                generation_time = end_time - start_time
                num_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
                
                total_time += generation_time
                total_tokens += num_tokens
            
            # トークン/秒を計算
            tokens_per_second = total_tokens / total_time
            
            results['benchmarks']['inference_speed'] = {
                'tokens_per_second': tokens_per_second,
                'total_time': total_time,
                'total_tokens': total_tokens
            }
            
            logger.info(f"推論速度: {tokens_per_second:.2f} tokens/秒")
            
        except Exception as e:
            logger.error(f"ベンチマークに失敗: {e}")
            results['benchmarks']['error'] = str(e)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='LLMモデル量子化')
    parser.add_argument('--model-path', required=True, help='量子化するモデルのパス')
    parser.add_argument('--method', choices=['awq', 'gptq', 'gguf', 'all'], 
                       default='awq', help='量子化手法')
    parser.add_argument('--bits', type=int, default=4, help='量子化ビット数')
    parser.add_argument('--group-size', type=int, default=128, 
                       help='量子化グループサイズ')
    parser.add_argument('--output-dir', default='models/quantized', 
                       help='出力ディレクトリ')
    parser.add_argument('--benchmark', action='store_true', 
                       help='量子化後にベンチマークを実行')
    
    args = parser.parse_args()
    
    try:
        # 量子化実行
        quantizer = ModelQuantizer(args.model_path, args.output_dir)
        
        quantized_paths = []
        errors = []
        
        if args.method in ['awq', 'all']:
            try:
                path = quantizer.quantize_awq(w_bit=args.bits, group_size=args.group_size)
                quantized_paths.append(path)
                logger.info(f"AWQ量子化成功: {path}")
            except Exception as e:
                logger.error(f"AWQ量子化エラー: {e}")
                errors.append(f"AWQ: {str(e)}")
        
        if args.method in ['gptq', 'all']:
            try:
                path = quantizer.quantize_gptq(bits=args.bits, group_size=args.group_size)
                quantized_paths.append(path)
                logger.info(f"GPTQ量子化成功: {path}")
            except Exception as e:
                logger.error(f"GPTQ量子化エラー: {e}")
                errors.append(f"GPTQ: {str(e)}")
        
        if args.method in ['gguf', 'all']:
            try:
                # GGUFの量子化レベルマッピング
                gguf_quant_map = {
                    2: "q2_k",
                    3: "q3_k_m",
                    4: "q4_k_m",
                    5: "q5_k_m",
                    6: "q6_k",
                    8: "q8_0"
                }
                quant_level = gguf_quant_map.get(args.bits, "q4_k_m")
                path = quantizer.convert_to_gguf(quantization=quant_level)
                quantized_paths.append(path)
                logger.info(f"GGUF変換成功: {path}")
            except Exception as e:
                logger.error(f"GGUF変換エラー: {e}")
                errors.append(f"GGUF: {str(e)}")
        
        # ベンチマーク実行
        if args.benchmark and quantized_paths:
            for path in quantized_paths:
                try:
                    results = quantizer.benchmark_quantized_model(path)
                    # ベンチマーク結果を保存
                    benchmark_path = path / "benchmark_results.json" if path.is_dir() else path.with_suffix('.benchmark.json')
                    with open(benchmark_path, 'w') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    logger.info(f"ベンチマーク結果保存: {benchmark_path}")
                except Exception as e:
                    logger.error(f"ベンチマークエラー ({path}): {e}")
        
        # 結果サマリー
        if quantized_paths:
            logger.info(f"\n量子化完了: {len(quantized_paths)}個のモデル")
            for path in quantized_paths:
                logger.info(f"  - {path}")
        else:
            logger.error("量子化されたモデルがありません")
            
        if errors:
            logger.error(f"\nエラー: {len(errors)}個")
            for error in errors:
                logger.error(f"  - {error}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()