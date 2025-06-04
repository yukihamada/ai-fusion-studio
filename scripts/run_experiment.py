#!/usr/bin/env python3
"""
エンドツーエンドの実験実行スクリプト
マージ → 評価 → 量子化 → レポート生成
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """実験の自動実行クラス"""
    
    def __init__(self, config_path: str, skip_steps: Optional[list] = None):
        self.config_path = Path(config_path)
        self.skip_steps = skip_steps or []
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.experiment_id = f"{self.config['merge_method']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = {
            'experiment_id': self.experiment_id,
            'config_path': str(config_path),
            'start_time': datetime.now().isoformat()
        }
    
    def run(self) -> Dict:
        """実験を実行"""
        logger.info(f"実験を開始: {self.experiment_id}")
        logger.info(f"設定ファイル: {self.config_path}")
        
        try:
            # 1. モデルマージ
            if 'merge' not in self.skip_steps:
                self._run_merge()
            
            # 2. 評価
            if 'evaluate' not in self.skip_steps:
                self._run_evaluation()
            
            # 3. 量子化
            if 'quantize' not in self.skip_steps:
                self._run_quantization()
            
            # 4. 実験追跡に登録
            if 'track' not in self.skip_steps:
                self._register_experiment()
            
            # 5. レポート生成
            if 'report' not in self.skip_steps:
                self._generate_report()
            
            self.results['status'] = 'completed'
            self.results['end_time'] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"実験が失敗しました: {e}")
            self.results['status'] = 'failed'
            self.results['error'] = str(e)
            raise
        
        return self.results
    
    def _run_merge(self) -> None:
        """モデルマージを実行"""
        logger.info("=== モデルマージを開始 ===")
        
        cmd = [
            'python', 'scripts/merge_models.py',
            '--config', str(self.config_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"マージに失敗: {result.stderr}")
        
        logger.info("マージ完了")
        self.results['merge'] = {
            'status': 'completed',
            'output_path': self.config['output_path']
        }
    
    def _run_evaluation(self) -> None:
        """モデル評価を実行"""
        logger.info("=== モデル評価を開始 ===")
        
        model_path = self.config['output_path']
        benchmarks = self.config.get('evaluation', {}).get('benchmarks', ['mt-bench-jp'])
        
        cmd = [
            'python', 'scripts/evaluate.py',
            '--model-path', model_path,
            '--benchmarks'] + benchmarks + [
            '--output-dir', f'evaluations/{self.experiment_id}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"評価に一部失敗: {result.stderr}")
        
        # 評価結果を読み込み
        eval_results_path = Path(f'evaluations/{self.experiment_id}/evaluation_results.json')
        if eval_results_path.exists():
            with open(eval_results_path, 'r') as f:
                eval_results = json.load(f)
                self.results['evaluation'] = eval_results
                
                # 主要スコアを記録
                if 'mt_bench_jp' in eval_results:
                    score = eval_results['mt_bench_jp']['overall_score']
                    logger.info(f"MT-Bench スコア: {score:.2f}/10")
    
    def _run_quantization(self) -> None:
        """モデル量子化を実行"""
        logger.info("=== モデル量子化を開始 ===")
        
        model_path = self.config['output_path']
        quant_config = self.config.get('quantization', {})
        method = quant_config.get('method', 'awq')
        bits = quant_config.get('bits', 4)
        
        cmd = [
            'python', 'scripts/quantize.py',
            '--model-path', model_path,
            '--method', method,
            '--bits', str(bits),
            '--output-dir', f'models/quantized/{self.experiment_id}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"量子化に失敗: {result.stderr}")
        else:
            # 量子化結果を記録
            quant_info_path = Path(f'models/quantized/{self.experiment_id}') / 'quantization_info.json'
            if quant_info_path.exists():
                with open(quant_info_path, 'r') as f:
                    quant_info = json.load(f)
                    self.results['quantization'] = quant_info
                    logger.info(f"量子化後サイズ: {quant_info.get('quantized_size_gb', 'N/A'):.2f} GB")
    
    def _register_experiment(self) -> None:
        """実験を追跡システムに登録"""
        logger.info("=== 実験を追跡システムに登録 ===")
        
        # 実験データを準備
        experiment_data = {
            'experiment_id': self.experiment_id,
            'merge_method': self.config['merge_method'],
            'models': self.config['models'],
            'config': self.config,
            'results': self.results
        }
        
        cmd = [
            'python', 'scripts/experiment_tracker.py',
            '--action', 'register',
            '--data', json.dumps(experiment_data)
        ]
        
        subprocess.run(cmd)
        
        # 評価結果があれば追加
        if 'evaluation' in self.results:
            cmd = [
                'python', 'scripts/experiment_tracker.py',
                '--action', 'update',
                '--experiment-id', self.experiment_id,
                '--data', json.dumps({'evaluations': self.results['evaluation']})
            ]
            subprocess.run(cmd)
    
    def _generate_report(self) -> None:
        """実験レポートを生成"""
        logger.info("=== 実験レポートを生成 ===")
        
        report_dir = Path(f'experiments/{self.experiment_id}/report')
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # サマリーレポート
        report_content = f"""
# 実験レポート: {self.experiment_id}

## 実験概要
- **実施日時**: {self.results['start_time']}
- **マージ手法**: {self.config['merge_method']}
- **使用モデル**: {', '.join([m['name'] for m in self.config['models']])}

## 結果サマリー
"""
        
        # 評価結果
        if 'evaluation' in self.results:
            report_content += "\n### 評価結果\n"
            if 'mt_bench_jp' in self.results['evaluation']:
                score = self.results['evaluation']['mt_bench_jp']['overall_score']
                report_content += f"- **Japanese MT-Bench**: {score:.2f}/10\n"
                
                # カテゴリ別スコア
                categories = self.results['evaluation']['mt_bench_jp'].get('category_scores', {})
                if categories:
                    report_content += "\n#### カテゴリ別スコア\n"
                    for cat, score in categories.items():
                        report_content += f"- {cat}: {score:.2f}\n"
        
        # 量子化結果
        if 'quantization' in self.results:
            report_content += "\n### 量子化結果\n"
            quant = self.results['quantization']
            report_content += f"- **量子化手法**: {quant.get('method', 'N/A')}\n"
            report_content += f"- **量子化後サイズ**: {quant.get('quantized_size_gb', 'N/A'):.2f} GB\n"
            report_content += f"- **圧縮率**: {quant.get('compression_ratio', 'N/A'):.2f}x\n"
        
        # 期待値との比較
        if 'expected_results' in self.config:
            report_content += "\n### 期待値との比較\n"
            expected = self.config['expected_results']
            
            if 'mt_bench_jp' in expected and 'evaluation' in self.results:
                actual = self.results['evaluation'].get('mt_bench_jp', {}).get('overall_score', 0)
                expected_score = expected['mt_bench_jp']
                diff = actual - expected_score
                report_content += f"- MT-Bench: {actual:.2f} (期待値: {expected_score}, 差分: {diff:+.2f})\n"
        
        # メタデータ
        if 'metadata' in self.config:
            report_content += "\n### 実験メタデータ\n"
            meta = self.config['metadata']
            report_content += f"- **説明**: {meta.get('description', 'N/A')}\n"
            if 'use_case' in meta:
                report_content += f"- **用途**: {meta['use_case']}\n"
        
        # レポート保存
        with open(report_dir / 'summary.md', 'w') as f:
            f.write(report_content)
        
        # 詳細結果も保存
        with open(report_dir / 'full_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"レポートを生成: {report_dir}")
    
    def cleanup(self) -> None:
        """一時ファイルのクリーンアップ"""
        # 必要に応じて実装
        pass


def main():
    parser = argparse.ArgumentParser(description='LLMマージ実験の自動実行')
    parser.add_argument('config', help='実験設定ファイル（YAML）')
    parser.add_argument('--skip', nargs='+', 
                       choices=['merge', 'evaluate', 'quantize', 'track', 'report'],
                       help='スキップするステップ')
    parser.add_argument('--dry-run', action='store_true', 
                       help='実行計画のみ表示')
    
    args = parser.parse_args()
    
    if args.dry_run:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        print("=== 実験実行計画 ===")
        print(f"設定ファイル: {args.config}")
        print(f"マージ手法: {config['merge_method']}")
        print(f"使用モデル: {[m['name'] for m in config['models']]}")
        print(f"出力先: {config['output_path']}")
        
        if args.skip:
            print(f"スキップ: {args.skip}")
        
        return
    
    # 実験実行
    runner = ExperimentRunner(args.config, skip_steps=args.skip)
    
    try:
        results = runner.run()
        
        print("\n=== 実験完了 ===")
        print(f"実験ID: {results['experiment_id']}")
        
        if 'evaluation' in results and 'mt_bench_jp' in results['evaluation']:
            score = results['evaluation']['mt_bench_jp']['overall_score']
            print(f"MT-Benchスコア: {score:.2f}/10")
        
        if 'quantization' in results:
            size = results['quantization'].get('quantized_size_gb', 'N/A')
            print(f"量子化後サイズ: {size:.2f} GB")
        
    except Exception as e:
        logger.error(f"実験が失敗しました: {e}")
        sys.exit(1)
    finally:
        runner.cleanup()


if __name__ == '__main__':
    main()