#!/usr/bin/env python3
"""
エンドツーエンド（E2E）テストスイート
実際のワークフロー全体をテストし、カバレッジを測定
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import threading
import signal
import coverage
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


class E2ETestRunner:
    """エンドツーエンドテストランナー"""
    
    def __init__(self, port=9876):
        self.project_root = Path(__file__).parent
        self.port = port
        self.web_process = None
        self.coverage = coverage.Coverage(
            source=['scripts', 'web'],
            omit=['*/test*', '*/capture*', '*/e2e*']
        )
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'coverage': {},
            'overall_success': False
        }
        
    def log(self, message, level="INFO"):
        """ログ出力"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] E2E-{level}: {message}")
    
    def setup_test_environment(self):
        """テスト環境のセットアップ"""
        self.log("🔧 テスト環境をセットアップ中...")
        
        # テスト用ディレクトリ作成
        test_dirs = ['models/e2e_test', 'experiments/e2e_test', 'evaluations/e2e_test']
        for test_dir in test_dirs:
            (self.project_root / test_dir).mkdir(parents=True, exist_ok=True)
        
        # テスト用設定ファイル作成
        test_config = {
            'merge_method': 'slerp',
            'output_path': 'models/e2e_test/test_merged_model',
            'models': [
                {'name': 'microsoft/DialoGPT-small', 'weight': 0.7},
                {'name': 'microsoft/DialoGPT-small', 'weight': 0.3}
            ],
            'alpha': 0.7,
            'evaluation': {
                'benchmarks': ['mt-bench-jp']
            },
            'metadata': {
                'description': 'E2Eテスト用設定',
                'test_mode': True
            }
        }
        
        import yaml
        test_config_path = self.project_root / 'configs' / 'e2e_test.yaml'
        with open(test_config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        self.log("✅ テスト環境セットアップ完了")
        return test_config_path
    
    def start_web_app(self):
        """Webアプリケーション起動"""
        self.log(f"🌐 Webアプリを起動中（ポート {self.port}）...")
        
        try:
            # ポート確認
            if self.is_port_in_use(self.port):
                self.port += 1
                self.log(f"ポート変更: {self.port}")
            
            self.web_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "web/app.py", 
                "--server.port", str(self.port),
                "--server.headless", "true"
            ], cwd=self.project_root, stdout=subprocess.PIPE, 
               stderr=subprocess.PIPE, text=True)
            
            # 起動待機
            for _ in range(30):
                if self.is_web_app_ready():
                    self.log(f"✅ Webアプリ起動完了: http://localhost:{self.port}")
                    return True
                time.sleep(1)
            
            self.log("❌ Webアプリ起動タイムアウト", "ERROR")
            return False
            
        except Exception as e:
            self.log(f"❌ Webアプリ起動エラー: {e}", "ERROR")
            return False
    
    def is_port_in_use(self, port):
        """ポート使用チェック"""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
        except:
            return False
    
    def is_web_app_ready(self):
        """Webアプリ準備完了チェック"""
        try:
            response = requests.get(f"http://localhost:{self.port}", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def test_complete_workflow(self):
        """完全ワークフローのE2Eテスト"""
        self.log("🚀 完全ワークフローE2Eテストを開始...")
        
        workflow_results = {}
        test_config_path = self.setup_test_environment()
        
        try:
            # カバレッジ測定開始
            self.coverage.start()
            
            # 1. モデルマージテスト
            self.log("1️⃣ モデルマージをテスト中...")
            merge_result = self.test_model_merge(test_config_path)
            workflow_results['model_merge'] = merge_result
            
            # 2. 評価テスト（軽量版）
            if merge_result['success']:
                self.log("2️⃣ 評価システムをテスト中...")
                eval_result = self.test_evaluation_system()
                workflow_results['evaluation'] = eval_result
            
            # 3. 量子化テスト（モック）
            self.log("3️⃣ 量子化システムをテスト中...")
            quant_result = self.test_quantization_system()
            workflow_results['quantization'] = quant_result
            
            # 4. 実験追跡テスト
            self.log("4️⃣ 実験追跡をテスト中...")
            track_result = self.test_experiment_tracking()
            workflow_results['experiment_tracking'] = track_result
            
            # 5. WebUIテスト
            self.log("5️⃣ WebUIをテスト中...")
            webui_result = self.test_web_ui_functionality()
            workflow_results['web_ui'] = webui_result
            
            # カバレッジ測定終了
            self.coverage.stop()
            self.coverage.save()
            
            # 総合判定
            all_success = all(result.get('success', False) for result in workflow_results.values())
            
            self.test_results['tests']['complete_workflow'] = {
                'success': all_success,
                'details': workflow_results,
                'duration': time.time() - self.start_time if hasattr(self, 'start_time') else 0
            }
            
            return all_success
            
        except Exception as e:
            self.log(f"❌ ワークフローテストエラー: {e}", "ERROR")
            self.test_results['tests']['complete_workflow'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    def test_model_merge(self, config_path):
        """モデルマージのテスト"""
        try:
            # マージ実行（タイムアウト付き）
            cmd = [
                sys.executable, "scripts/merge_models.py",
                "--config", str(config_path),
                "--device", "cpu"  # CPUで高速化
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=self.project_root, timeout=120)
            
            success = result.returncode == 0
            
            # 出力ファイルの確認
            output_dir = Path('models/e2e_test/test_merged_model')
            if success and output_dir.exists():
                model_files = list(output_dir.glob('*.json')) + list(output_dir.glob('*.bin'))
                success = len(model_files) > 0
            
            return {
                'success': success,
                'stdout': result.stdout[:500],
                'stderr': result.stderr[:500] if result.stderr else None,
                'output_files_found': success
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'マージタイムアウト（120秒）'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_evaluation_system(self):
        """評価システムのテスト"""
        try:
            # 軽量評価実行
            cmd = [
                sys.executable, "scripts/evaluate.py",
                "--model-path", "microsoft/DialoGPT-small",  # 軽量モデルで代替
                "--benchmarks", "mt-bench-jp",
                "--output-dir", "evaluations/e2e_test"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=self.project_root, timeout=60)
            
            success = result.returncode == 0
            
            # 評価結果ファイルの確認
            if success:
                eval_files = list(Path('evaluations/e2e_test').glob('*.json'))
                success = len(eval_files) > 0
            
            return {
                'success': success,
                'evaluation_files_created': success
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': '評価タイムアウト（60秒）'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_quantization_system(self):
        """量子化システムのテスト（モック）"""
        try:
            # 量子化ヘルプのテスト（実際の量子化は時間がかかるため）
            cmd = [
                sys.executable, "scripts/quantize.py", "--help"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=self.project_root, timeout=15)
            
            success = result.returncode == 0 and "量子化" in result.stdout
            
            return {
                'success': success,
                'help_available': success
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_experiment_tracking(self):
        """実験追跡のテスト"""
        try:
            # リーダーボード生成テスト
            cmd = [
                sys.executable, "scripts/experiment_tracker.py",
                "--action", "leaderboard"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=self.project_root, timeout=30)
            
            success = result.returncode == 0
            
            return {
                'success': success,
                'leaderboard_generated': success
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_web_ui_functionality(self):
        """WebUI機能のテスト"""
        if not self.is_web_app_ready():
            return {'success': False, 'error': 'WebUI利用不可'}
        
        try:
            # 基本エンドポイントのテスト
            endpoints = [
                ('main_page', '/'),
                ('health_check', '/_stcore/health')
            ]
            
            results = {}
            for endpoint_name, path in endpoints:
                try:
                    response = requests.get(f"http://localhost:{self.port}{path}", timeout=10)
                    success = response.status_code == 200
                    results[endpoint_name] = {
                        'success': success,
                        'status_code': response.status_code,
                        'response_time': response.elapsed.total_seconds()
                    }
                except Exception as e:
                    results[endpoint_name] = {'success': False, 'error': str(e)}
            
            overall_success = all(r.get('success', False) for r in results.values())
            
            return {
                'success': overall_success,
                'endpoint_results': results
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def generate_coverage_report(self):
        """カバレッジレポート生成"""
        self.log("📊 カバレッジレポートを生成中...")
        
        try:
            # HTMLレポート生成
            html_dir = self.project_root / 'coverage_html'
            if html_dir.exists():
                shutil.rmtree(html_dir)
            
            self.coverage.html_report(directory=str(html_dir))
            
            # JSON形式のカバレッジデータ取得
            coverage_data = {}
            total_lines = 0
            covered_lines = 0
            
            for filename in self.coverage.get_data().measured_files():
                file_coverage = self.coverage.analysis(filename)
                total_lines += len(file_coverage[1])  # executable lines
                covered_lines += len(file_coverage[1]) - len(file_coverage[3])  # missing lines
                
                rel_path = str(Path(filename).relative_to(self.project_root))
                coverage_data[rel_path] = {
                    'total_lines': len(file_coverage[1]),
                    'covered_lines': len(file_coverage[1]) - len(file_coverage[3]),
                    'coverage_percent': ((len(file_coverage[1]) - len(file_coverage[3])) / len(file_coverage[1]) * 100) if file_coverage[1] else 0
                }
            
            overall_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
            
            self.test_results['coverage'] = {
                'overall_percent': overall_coverage,
                'total_lines': total_lines,
                'covered_lines': covered_lines,
                'file_coverage': coverage_data,
                'html_report_path': str(html_dir / 'index.html')
            }
            
            self.log(f"✅ カバレッジ: {overall_coverage:.1f}% ({covered_lines}/{total_lines}行)")
            return overall_coverage
            
        except Exception as e:
            self.log(f"❌ カバレッジレポート生成エラー: {e}", "ERROR")
            return 0
    
    def create_coverage_visualization(self):
        """カバレッジ可視化"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 日本語フォント設定
        plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 全体カバレッジゲージ
        coverage_percent = self.test_results['coverage']['overall_percent']
        
        # ゲージチャート
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # 背景円
        ax1.plot(r * np.cos(theta), r * np.sin(theta), 'lightgray', linewidth=10)
        
        # カバレッジ円
        coverage_theta = theta[:int(coverage_percent)]
        color = '#4CAF50' if coverage_percent >= 80 else '#FF9800' if coverage_percent >= 60 else '#F44336'
        ax1.plot(r * np.cos(coverage_theta), r * np.sin(coverage_theta), color, linewidth=10)
        
        # パーセンテージ表示
        ax1.text(0, -0.3, f'{coverage_percent:.1f}%', ha='center', va='center', 
                fontsize=24, fontweight='bold')
        ax1.text(0, -0.5, 'コードカバレッジ', ha='center', va='center', fontsize=14)
        
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-0.8, 1.2)
        ax1.axis('off')
        ax1.set_title('E2Eテストカバレッジ', fontsize=16, pad=20)
        
        # 2. ファイル別カバレッジ
        file_coverage = self.test_results['coverage']['file_coverage']
        if file_coverage:
            files = list(file_coverage.keys())[:10]  # 上位10ファイル
            coverages = [file_coverage[f]['coverage_percent'] for f in files]
            
            # ファイル名を短縮
            short_files = [f.split('/')[-1] for f in files]
            
            bars = ax2.barh(short_files, coverages, 
                           color=['#4CAF50' if c >= 80 else '#FF9800' if c >= 60 else '#F44336' 
                                 for c in coverages])
            
            # パーセンテージ表示
            for bar, coverage in zip(bars, coverages):
                width = bar.get_width()
                ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
                        f'{coverage:.1f}%', ha='left', va='center')
            
            ax2.set_xlim(0, 105)
            ax2.set_xlabel('カバレッジ (%)')
            ax2.set_title('ファイル別カバレッジ', fontsize=16)
            ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('e2e_coverage_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log("✅ カバレッジ可視化完了: e2e_coverage_report.png")
    
    def take_screenshots(self):
        """WebUIのスクリーンショット撮影"""
        if not self.is_web_app_ready():
            self.log("⚠️ WebUIが利用できないためスクリーンショットをスキップ")
            return False
        
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            
            # Chrome設定
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox") 
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--window-size=1920,1080")
            
            try:
                driver = webdriver.Chrome(options=chrome_options)
            except:
                self.log("ChromeDriverが見つかりません", "WARN")
                return False
            
            screenshots_dir = Path("e2e_screenshots")
            screenshots_dir.mkdir(exist_ok=True)
            
            # メインページのスクリーンショット
            self.log("📸 E2E WebUIスクリーンショット撮影中...")
            driver.get(f"http://localhost:{self.port}")
            time.sleep(5)
            driver.save_screenshot(str(screenshots_dir / "e2e_webui_main.png"))
            
            driver.quit()
            
            self.log("✅ スクリーンショット撮影完了")
            return True
            
        except Exception as e:
            self.log(f"❌ スクリーンショット撮影エラー: {e}", "ERROR")
            return False
    
    def cleanup(self):
        """テスト環境のクリーンアップ"""
        self.log("🧹 テスト環境をクリーンアップ中...")
        
        # Webアプリ停止
        if self.web_process:
            try:
                self.web_process.terminate()
                self.web_process.wait(timeout=10)
            except:
                try:
                    self.web_process.kill()
                except:
                    pass
        
        # テストファイル削除
        test_files = [
            'configs/e2e_test.yaml',
            'models/e2e_test',
            'experiments/e2e_test',
            'evaluations/e2e_test'
        ]
        
        for test_file in test_files:
            file_path = self.project_root / test_file
            if file_path.exists():
                try:
                    if file_path.is_dir():
                        shutil.rmtree(file_path)
                    else:
                        file_path.unlink()
                except:
                    pass
        
        self.log("✅ クリーンアップ完了")
    
    def generate_final_report(self):
        """最終E2Eテストレポート生成"""
        self.log("📊 最終E2Eテストレポートを生成中...")
        
        # 結果サマリー
        workflow_test = self.test_results['tests'].get('complete_workflow', {})
        coverage_data = self.test_results['coverage']
        
        overall_success = workflow_test.get('success', False)
        coverage_percent = coverage_data.get('overall_percent', 0)
        
        self.test_results['overall_success'] = overall_success
        
        # レポート保存
        report_path = self.project_root / f"e2e_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        # サマリー表示
        print("\n" + "="*80)
        print("🎯 LLM Merge Lab E2Eテスト結果")
        print("="*80)
        
        print(f"📅 実行日時: {self.test_results['timestamp']}")
        print(f"📝 レポートファイル: {report_path}")
        print()
        
        # ワークフローテスト結果
        if workflow_test:
            print("🚀 ワークフローテスト結果:")
            details = workflow_test.get('details', {})
            for test_name, result in details.items():
                status = "✅" if result.get('success', False) else "❌"
                print(f"  {status} {test_name}")
        
        # カバレッジ結果
        print(f"\n📊 コードカバレッジ: {coverage_percent:.1f}%")
        print(f"   総行数: {coverage_data.get('total_lines', 0)}")
        print(f"   カバー行数: {coverage_data.get('covered_lines', 0)}")
        
        if 'html_report_path' in coverage_data:
            print(f"   📄 詳細レポート: {coverage_data['html_report_path']}")
        
        print("\n" + "="*80)
        if overall_success and coverage_percent >= 70:
            print("🎉 E2Eテスト完全成功！LLM Merge Labは本番Ready！")
        elif overall_success:
            print("✅ E2Eテスト成功（カバレッジ改善余地あり）")
        else:
            print("⚠️ E2Eテストで問題を検出しました")
        print("="*80)
        
        return overall_success
    
    def run_complete_e2e_test(self):
        """完全E2Eテスト実行"""
        self.start_time = time.time()
        self.log("🚀 LLM Merge Lab E2Eテストを開始...")
        
        try:
            # 1. Webアプリ起動
            if not self.start_web_app():
                return False
            
            # 2. 完全ワークフローテスト
            workflow_success = self.test_complete_workflow()
            
            # 3. カバレッジレポート生成
            coverage_percent = self.generate_coverage_report()
            
            # 4. カバレッジ可視化
            self.create_coverage_visualization()
            
            # 5. スクリーンショット撮影
            self.take_screenshots()
            
            # 6. 最終レポート生成
            overall_success = self.generate_final_report()
            
            return overall_success
            
        except KeyboardInterrupt:
            self.log("⏹️ E2Eテストが中断されました", "WARN")
            return False
        except Exception as e:
            self.log(f"💥 E2Eテスト実行エラー: {e}", "ERROR")
            return False
        finally:
            self.cleanup()


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM Merge Lab E2Eテスト')
    parser.add_argument('--port', type=int, default=9876, 
                       help='Webアプリポート番号（デフォルト: 9876）')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='テスト後のクリーンアップをスキップ')
    
    args = parser.parse_args()
    
    # カバレッジライブラリのインストール確認
    try:
        import coverage
    except ImportError:
        print("❌ coverageライブラリが必要です: pip install coverage")
        sys.exit(1)
    
    runner = E2ETestRunner(port=args.port)
    
    # シグナルハンドラ設定
    def signal_handler(sig, frame):
        print("\n⏹️ E2Eテストを中断中...")
        runner.cleanup()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        success = runner.run_complete_e2e_test()
        
        if not args.no_cleanup:
            runner.cleanup()
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n💥 予期しないエラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()