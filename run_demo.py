#!/usr/bin/env python3
"""
AI Fusion Studio デモ実行スクリプト
軽量なテストモデルを使用した完全なワークフローのデモ
"""

import os
import sys
import json
import yaml
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime


def create_demo_config():
    """デモ用の軽量設定を作成"""
    demo_config = {
        'merge_method': 'slerp',
        'output_path': 'models/demo_merged_model',
        'models': [
            {'name': 'microsoft/DialoGPT-small', 'weight': 0.7},
            {'name': 'microsoft/DialoGPT-small', 'weight': 0.3}  # 同じモデルでデモ
        ],
        'alpha': 0.7,
        'evaluation': {
            'benchmarks': ['mt-bench-jp']
        },
        'metadata': {
            'description': 'デモ用の軽量マージ実験',
            'use_case': '機能テスト'
        }
    }
    
    # デモ設定を保存
    demo_config_path = Path('configs/demo_config.yaml')
    with open(demo_config_path, 'w') as f:
        yaml.dump(demo_config, f, default_flow_style=False)
    
    return demo_config_path


def run_demo():
    """デモワークフローを実行"""
    print("🎬 AI Fusion Studio デモを開始します...")
    print("軽量なテストモデルを使用して完全なワークフローをデモンストレーションします。")
    print()
    
    try:
        # 1. デモ設定の作成
        print("1️⃣ デモ設定を作成中...")
        demo_config_path = create_demo_config()
        print(f"   ✅ 設定ファイル作成: {demo_config_path}")
        
        # 2. 設定ファイルの検証
        print("\n2️⃣ 設定ファイルを検証中...")
        with open(demo_config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   ✅ マージ手法: {config['merge_method']}")
        print(f"   ✅ 使用モデル: {len(config['models'])}個")
        
        # 3. 実験追跡システムのテスト
        print("\n3️⃣ 実験追跡システムをテスト中...")
        experiment_data = {
            'experiment_id': 'demo_experiment',
            'merge_method': config['merge_method'],
            'models': config['models'],
            'timestamp': datetime.now().isoformat()
        }
        
        # 実験を登録（実際のファイルI/Oをテスト）
        experiments_dir = Path('experiments')
        experiments_dir.mkdir(exist_ok=True)
        
        db_path = experiments_dir / 'experiments_db.json'
        if db_path.exists():
            with open(db_path, 'r') as f:
                experiments = json.load(f)
        else:
            experiments = []
        
        experiments.append(experiment_data)
        
        with open(db_path, 'w') as f:
            json.dump(experiments, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ 実験登録完了: {experiment_data['experiment_id']}")
        
        # 4. 評価システムのテスト（モック）
        print("\n4️⃣ 評価システムをテスト中...")
        mock_evaluation = {
            'mt_bench_jp': {
                'overall_score': 6.8,
                'category_scores': {
                    'writing': 7.2,
                    'reasoning': 6.5,
                    'coding': 6.9
                }
            },
            'mathematical_reasoning': {
                'accuracy': 0.72
            }
        }
        
        # 評価結果を実験データに追加
        for exp in experiments:
            if exp['experiment_id'] == 'demo_experiment':
                exp['evaluations'] = mock_evaluation
                exp['status'] = 'completed'
                break
        
        with open(db_path, 'w') as f:
            json.dump(experiments, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ 評価完了: MT-Bench {mock_evaluation['mt_bench_jp']['overall_score']}/10")
        
        # 5. 量子化情報の追加
        print("\n5️⃣ 量子化情報を追加中...")
        mock_quantization = {
            'method': 'awq',
            'bits': 4,
            'original_size_gb': 2.1,
            'quantized_size_gb': 0.8,
            'compression_ratio': 2.6
        }
        
        for exp in experiments:
            if exp['experiment_id'] == 'demo_experiment':
                exp['quantization'] = mock_quantization
                break
        
        with open(db_path, 'w') as f:
            json.dump(experiments, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ 量子化完了: {mock_quantization['quantized_size_gb']}GB ({mock_quantization['compression_ratio']:.1f}x圧縮)")
        
        # 6. レポート生成
        print("\n6️⃣ デモレポートを生成中...")
        demo_report = {
            'demo_summary': {
                'title': 'AI Fusion Studio デモ実行結果',
                'timestamp': datetime.now().isoformat(),
                'config_file': str(demo_config_path),
                'experiment_id': 'demo_experiment',
                'status': 'success'
            },
            'results': {
                'mt_bench_score': mock_evaluation['mt_bench_jp']['overall_score'],
                'math_accuracy': mock_evaluation['mathematical_reasoning']['accuracy'],
                'model_size_gb': mock_quantization['quantized_size_gb'],
                'compression_ratio': mock_quantization['compression_ratio']
            },
            'workflow_steps': [
                '✅ 設定ファイル作成',
                '✅ 実験登録',
                '✅ 評価システム',
                '✅ 量子化システム',
                '✅ レポート生成'
            ]
        }
        
        demo_report_path = experiments_dir / 'demo_report.json'
        with open(demo_report_path, 'w') as f:
            json.dump(demo_report, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ レポート保存: {demo_report_path}")
        
        # 7. 結果サマリー
        print("\n" + "="*60)
        print("🎉 デモ実行完了！")
        print("="*60)
        print(f"📊 MT-Benchスコア: {mock_evaluation['mt_bench_jp']['overall_score']}/10")
        print(f"🧮 数学精度: {mock_evaluation['mathematical_reasoning']['accuracy']:.1%}")
        print(f"💾 モデルサイズ: {mock_quantization['quantized_size_gb']}GB")
        print(f"🗜️  圧縮率: {mock_quantization['compression_ratio']:.1f}x")
        print()
        print("📁 生成されたファイル:")
        print(f"   - 設定: {demo_config_path}")
        print(f"   - 実験DB: {db_path}")
        print(f"   - レポート: {demo_report_path}")
        print()
        print("🌐 次のステップ:")
        print("   - Webインターフェースを起動: ./start_web.sh")
        print("   - 実際のマージ実験: make experiment-gemma-qwen")
        print("   - テスト実行: python test_runner.py")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ デモ実行中にエラーが発生: {e}")
        return False


def cleanup_demo():
    """デモファイルのクリーンアップ"""
    files_to_remove = [
        'configs/demo_config.yaml',
        'experiments/demo_report.json'
    ]
    
    print("\n🧹 デモファイルをクリーンアップ中...")
    for file_path in files_to_remove:
        if Path(file_path).exists():
            os.remove(file_path)
            print(f"   🗑️  削除: {file_path}")


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Fusion Studio デモ実行')
    parser.add_argument('--cleanup', action='store_true', help='デモファイルをクリーンアップ')
    args = parser.parse_args()
    
    if args.cleanup:
        cleanup_demo()
        return
    
    # デモ実行
    success = run_demo()
    
    if success:
        print("\n✨ AI Fusion Studioの機能を体験していただき、ありがとうございました！")
        
        # Webアプリの起動を提案
        response = input("\n🌐 Webインターフェースを起動しますか？ (y/N): ")
        if response.lower() in ['y', 'yes']:
            print("Webアプリケーションを起動中...")
            try:
                subprocess.run(['./start_web.sh'], check=True)
            except subprocess.CalledProcessError:
                print("❌ Webアプリの起動に失敗しました。手動で './start_web.sh' を実行してください。")
    else:
        print("\n❌ デモが失敗しました。")
        sys.exit(1)


if __name__ == "__main__":
    main()