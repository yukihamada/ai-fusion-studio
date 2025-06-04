#!/usr/bin/env python3
"""
完全自動テストスイート
全機能を包括的にテストし、生成されたモデルの品質も評価
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import signal


class AutoTestSuite:
    """完全自動テストスイート"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'overall_status': 'running'
        }
        self.web_process = None
        self.web_port = 8501
    
    def log(self, message, level="INFO"):
        """ログ出力"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def run_unit_tests(self):
        """ユニットテスト実行"""
        self.log("🧪 ユニットテストを開始...")
        
        start_time = time.time()
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/", "-v", "--tb=short",
                "--json-report", "--json-report-file=test_results.json"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=120)
            
            duration = time.time() - start_time
            success = result.returncode == 0
            
            # JSONレポート読み込み
            json_path = self.project_root / "test_results.json"
            details = {}
            if json_path.exists():
                try:
                    with open(json_path, 'r') as f:
                        pytest_report = json.load(f)
                        summary = pytest_report.get('summary', {})
                        details = {
                            'total': summary.get('total', 0),
                            'passed': summary.get('passed', 0),
                            'failed': summary.get('failed', 0),
                            'skipped': summary.get('skipped', 0)
                        }
                except (json.JSONDecodeError, KeyError) as e:
                    self.log(f"JSONレポート読み込みエラー: {e}", "WARNING")
                    # 出力から解析を試みる
                    import re
                    if result.stdout:
                        # パターンマッチングで結果を抽出
                        passed_match = re.search(r'(\d+) passed', result.stdout)
                        failed_match = re.search(r'(\d+) failed', result.stdout)
                        skipped_match = re.search(r'(\d+) skipped', result.stdout)
                        
                        details = {
                            'passed': int(passed_match.group(1)) if passed_match else 0,
                            'failed': int(failed_match.group(1)) if failed_match else 0,
                            'skipped': int(skipped_match.group(1)) if skipped_match else 0
                        }
                        details['total'] = details['passed'] + details['failed'] + details['skipped']
            
            self.test_results['tests']['unit_tests'] = {
                'status': 'passed' if success else 'failed',
                'duration': duration,
                'details': details,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            self.log(f"✅ ユニットテスト完了 ({duration:.1f}s)")
            return success
            
        except subprocess.TimeoutExpired:
            self.log("❌ ユニットテストがタイムアウト", "ERROR")
            self.test_results['tests']['unit_tests'] = {
                'status': 'timeout',
                'duration': 120,
                'error': 'Test timeout after 120 seconds'
            }
            return False
        except Exception as e:
            self.log(f"❌ ユニットテスト実行エラー: {e}", "ERROR")
            self.test_results['tests']['unit_tests'] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def test_script_functionality(self):
        """スクリプト機能テスト"""
        self.log("🔧 スクリプト機能をテスト中...")
        
        script_tests = {
            'merge_models_help': ['python', 'scripts/merge_models.py', '--help'],
            'evaluate_help': ['python', 'scripts/evaluate.py', '--help'],
            'experiment_tracker_help': ['python', 'scripts/experiment_tracker.py', '--help'],
            'quantize_help': ['python', 'scripts/quantize.py', '--help']
        }
        
        results = {}
        for test_name, cmd in script_tests.items():
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, 
                                     cwd=self.project_root, timeout=30)
                success = result.returncode == 0
                results[test_name] = {
                    'status': 'passed' if success else 'failed',
                    'stdout': result.stdout[:500],  # 最初の500文字のみ
                    'stderr': result.stderr[:500] if result.stderr else None
                }
                
                status = "✅" if success else "❌"
                self.log(f"{status} {test_name}")
                
            except subprocess.TimeoutExpired:
                results[test_name] = {'status': 'timeout'}
                self.log(f"⏰ {test_name} タイムアウト")
            except Exception as e:
                results[test_name] = {'status': 'error', 'error': str(e)}
                self.log(f"❌ {test_name} エラー: {e}")
        
        self.test_results['tests']['script_functionality'] = results
        all_passed = all(r['status'] == 'passed' for r in results.values())
        
        self.log(f"✅ スクリプト機能テスト完了" if all_passed else "⚠️ 一部スクリプトテストが失敗")
        return all_passed
    
    def start_web_app(self):
        """Webアプリケーション起動"""
        self.log("🌐 Webアプリケーションを起動中...")
        
        try:
            # ポート確認
            if self.is_port_in_use(self.web_port):
                self.web_port = 8502
            
            self.web_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "web/app.py", "--server.port", str(self.web_port),
                "--server.headless", "true"
            ], cwd=self.project_root, stdout=subprocess.PIPE, 
               stderr=subprocess.PIPE, text=True)
            
            # 起動待機
            for _ in range(30):  # 30秒待機
                if self.is_web_app_ready():
                    self.log(f"✅ Webアプリ起動完了: http://localhost:{self.web_port}")
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
            response = requests.get(f"http://localhost:{self.web_port}", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def test_web_app_endpoints(self):
        """Webアプリエンドポイントテスト"""
        if not self.is_web_app_ready():
            self.log("❌ Webアプリが利用できません", "ERROR")
            return False
        
        self.log("🌐 Webアプリエンドポイントをテスト中...")
        
        endpoints_to_test = [
            ('main_page', '/'),
            ('health_check', '/_stcore/health')
        ]
        
        results = {}
        for endpoint_name, path in endpoints_to_test:
            try:
                response = requests.get(f"http://localhost:{self.web_port}{path}", timeout=10)
                success = response.status_code == 200
                
                results[endpoint_name] = {
                    'status': 'passed' if success else 'failed',
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds()
                }
                
                status = "✅" if success else "❌"
                self.log(f"{status} {endpoint_name} ({response.status_code})")
                
            except Exception as e:
                results[endpoint_name] = {
                    'status': 'error',
                    'error': str(e)
                }
                self.log(f"❌ {endpoint_name} エラー: {e}")
        
        self.test_results['tests']['web_app_endpoints'] = results
        
        all_passed = all(r['status'] == 'passed' for r in results.values())
        self.log(f"✅ Webアプリテスト完了" if all_passed else "⚠️ 一部Webアプリテストが失敗")
        return all_passed
    
    def test_demo_workflow(self):
        """デモワークフローテスト"""
        self.log("🎬 デモワークフローをテスト中...")
        
        try:
            result = subprocess.run([
                sys.executable, "run_demo.py"
            ], input="n\n", capture_output=True, text=True, 
               cwd=self.project_root, timeout=60)
            
            success = result.returncode == 0 and "🎉 デモ実行完了！" in result.stdout
            
            self.test_results['tests']['demo_workflow'] = {
                'status': 'passed' if success else 'failed',
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            if success:
                self.log("✅ デモワークフロー完了")
                
                # 生成されたファイルの確認
                expected_files = [
                    'configs/demo_config.yaml',
                    'experiments/experiments_db.json',
                    'experiments/demo_report.json'
                ]
                
                files_check = {}
                for file_path in expected_files:
                    full_path = self.project_root / file_path
                    exists = full_path.exists()
                    files_check[file_path] = exists
                    
                    status = "✅" if exists else "❌"
                    self.log(f"  {status} {file_path}")
                
                self.test_results['tests']['demo_workflow']['files_generated'] = files_check
                
                return success and all(files_check.values())
            else:
                self.log("❌ デモワークフロー失敗", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("❌ デモワークフロー タイムアウト", "ERROR")
            self.test_results['tests']['demo_workflow'] = {
                'status': 'timeout',
                'error': 'Demo workflow timeout'
            }
            return False
        except Exception as e:
            self.log(f"❌ デモワークフロー エラー: {e}", "ERROR")
            self.test_results['tests']['demo_workflow'] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def test_experiment_tracking(self):
        """実験追跡システムテスト"""
        self.log("📊 実験追跡システムをテスト中...")
        
        try:
            # リーダーボード生成テスト
            result = subprocess.run([
                sys.executable, "scripts/experiment_tracker.py",
                "--action", "leaderboard"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=30)
            
            success = result.returncode == 0
            
            self.test_results['tests']['experiment_tracking'] = {
                'status': 'passed' if success else 'failed',
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            if success:
                self.log("✅ 実験追跡システム テスト完了")
            else:
                self.log("❌ 実験追跡システム テスト失敗", "ERROR")
            
            return success
            
        except Exception as e:
            self.log(f"❌ 実験追跡システム エラー: {e}", "ERROR")
            self.test_results['tests']['experiment_tracking'] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def test_model_quality(self):
        """生成モデルの品質テスト"""
        self.log("🤖 生成モデルの品質をテスト中...")
        
        # デモで生成された実験データを使用
        experiments_db_path = self.project_root / "experiments" / "experiments_db.json"
        
        if not experiments_db_path.exists():
            self.log("❌ 実験データベースが見つかりません", "ERROR")
            return False
        
        try:
            with open(experiments_db_path, 'r') as f:
                experiments = json.load(f)
            
            quality_tests = {
                'has_experiments': len(experiments) > 0,
                'has_completed_experiments': any(exp.get('status') == 'completed' for exp in experiments),
                'has_evaluation_results': False,
                'has_quantization_info': False,
                'reasonable_scores': False
            }
            
            for exp in experiments:
                if exp.get('status') == 'completed':
                    # 評価結果の存在確認
                    if 'evaluations' in exp:
                        quality_tests['has_evaluation_results'] = True
                        
                        # スコアの妥当性チェック
                        if 'mt_bench_jp' in exp['evaluations']:
                            score = exp['evaluations']['mt_bench_jp'].get('overall_score', 0)
                            if 0 <= score <= 10:
                                quality_tests['reasonable_scores'] = True
                    
                    # 量子化情報の確認
                    if 'quantization' in exp:
                        quality_tests['has_quantization_info'] = True
            
            all_passed = all(quality_tests.values())
            
            self.test_results['tests']['model_quality'] = {
                'status': 'passed' if all_passed else 'failed',
                'details': quality_tests,
                'experiments_count': len(experiments)
            }
            
            if all_passed:
                self.log("✅ モデル品質テスト完了")
            else:
                self.log("⚠️ モデル品質テストで問題を検出")
                for test, result in quality_tests.items():
                    if not result:
                        self.log(f"  ❌ {test}")
            
            return all_passed
            
        except Exception as e:
            self.log(f"❌ モデル品質テスト エラー: {e}", "ERROR")
            self.test_results['tests']['model_quality'] = {
                'status': 'error',
                'error': str(e)
            }
            return False
    
    def test_performance_benchmarks(self):
        """パフォーマンスベンチマーク"""
        self.log("⚡ パフォーマンスベンチマークを実行中...")
        
        benchmarks = {}
        
        # スクリプト起動時間
        scripts_to_benchmark = [
            'scripts/merge_models.py',
            'scripts/evaluate.py',
            'scripts/experiment_tracker.py'
        ]
        
        for script in scripts_to_benchmark:
            try:
                start_time = time.time()
                result = subprocess.run([
                    sys.executable, script, '--help'
                ], capture_output=True, timeout=15)
                end_time = time.time()
                
                benchmarks[script] = {
                    'startup_time': end_time - start_time,
                    'success': result.returncode == 0
                }
                
            except subprocess.TimeoutExpired:
                benchmarks[script] = {
                    'startup_time': 15.0,
                    'success': False,
                    'timeout': True
                }
        
        # メモリ使用量チェック（簡易版）
        try:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            benchmarks['memory_usage_mb'] = memory_usage
        except:
            benchmarks['memory_usage_mb'] = 'unknown'
        
        self.test_results['tests']['performance_benchmarks'] = benchmarks
        
        # パフォーマンス基準チェック（すべて10秒以内）
        performance_ok = all(
            b.get('startup_time', 0) < 10.0 and b.get('success', False)
            for b in benchmarks.values() if isinstance(b, dict)
        )
        
        self.log(f"✅ パフォーマンステスト完了" if performance_ok else "⚠️ パフォーマンス改善が必要")
        return performance_ok
    
    def cleanup(self):
        """テスト環境のクリーンアップ"""
        self.log("🧹 テスト環境をクリーンアップ中...")
        
        # Webアプリを停止
        if self.web_process:
            try:
                self.web_process.terminate()
                self.web_process.wait(timeout=10)
                self.log("✅ Webアプリを停止しました")
            except:
                try:
                    self.web_process.kill()
                    self.log("⚠️ Webアプリを強制終了しました")
                except:
                    pass
        
        # 一時ファイルの削除
        temp_files = [
            'test_results.json',
            'configs/demo_config.yaml'
        ]
        
        for temp_file in temp_files:
            file_path = self.project_root / temp_file
            if file_path.exists():
                try:
                    file_path.unlink()
                    self.log(f"🗑️ 削除: {temp_file}")
                except:
                    pass
    
    def generate_comprehensive_report(self):
        """包括的なテストレポート生成"""
        self.log("📊 包括的テストレポートを生成中...")
        
        self.test_results['overall_status'] = 'completed'
        
        # 総合判定
        test_statuses = []
        for test_name, test_result in self.test_results['tests'].items():
            if isinstance(test_result, dict):
                status = test_result.get('status', 'unknown')
                test_statuses.append(status == 'passed')
        
        overall_success = all(test_statuses) if test_statuses else False
        self.test_results['overall_success'] = overall_success
        
        # レポート保存
        report_path = self.project_root / f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        # サマリー表示
        print("\n" + "="*80)
        print("🎯 AI Fusion Studio 包括的テストレポート")
        print("="*80)
        
        print(f"📅 実行日時: {self.test_results['timestamp']}")
        print(f"📝 レポートファイル: {report_path}")
        print()
        
        # テスト結果サマリー
        for test_name, test_result in self.test_results['tests'].items():
            if isinstance(test_result, dict):
                status = test_result.get('status', 'unknown')
                status_icon = {
                    'passed': '✅',
                    'failed': '❌', 
                    'error': '💥',
                    'timeout': '⏰',
                    'unknown': '❓'
                }.get(status, '❓')
                
                print(f"{status_icon} {test_name}: {status}")
                
                # 詳細情報があれば表示
                if 'duration' in test_result:
                    print(f"   ⏱️ 実行時間: {test_result['duration']:.2f}秒")
                
                if 'details' in test_result:
                    details = test_result['details']
                    if isinstance(details, dict):
                        for key, value in details.items():
                            print(f"   📊 {key}: {value}")
        
        print("\n" + "="*80)
        if overall_success:
            print("🎉 すべてのテストが成功しました！")
            print("🚀 AI Fusion Studioは本番環境で使用する準備ができています。")
        else:
            print("⚠️ 一部のテストが失敗しました。")
            print("🔧 問題を確認して修正してください。")
        print("="*80)
        
        return overall_success
    
    def run_all_tests(self):
        """すべてのテストを実行"""
        self.log("🚀 AI Fusion Studio 包括的自動テストを開始...")
        
        try:
            # テスト実行順序
            test_sequence = [
                ("ユニットテスト", self.run_unit_tests),
                ("スクリプト機能", self.test_script_functionality),
                ("デモワークフロー", self.test_demo_workflow),
                ("実験追跡システム", self.test_experiment_tracking),
                ("モデル品質", self.test_model_quality),
                ("Webアプリ起動", self.start_web_app),
                ("Webアプリエンドポイント", self.test_web_app_endpoints),
                ("パフォーマンスベンチマーク", self.test_performance_benchmarks)
            ]
            
            for test_name, test_func in test_sequence:
                self.log(f"▶️ {test_name} を実行中...")
                try:
                    result = test_func()
                    status = "✅ 成功" if result else "❌ 失敗"
                    self.log(f"{status}: {test_name}")
                except Exception as e:
                    self.log(f"💥 エラー: {test_name} - {e}", "ERROR")
            
            # 最終レポート生成
            overall_success = self.generate_comprehensive_report()
            
            return overall_success
            
        except KeyboardInterrupt:
            self.log("⏹️ テストが中断されました", "WARN")
            return False
        finally:
            self.cleanup()


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Fusion Studio 包括的自動テスト')
    parser.add_argument('--quick', action='store_true', help='クイックテスト（重いテストをスキップ）')
    parser.add_argument('--web-only', action='store_true', help='Webアプリテストのみ')
    args = parser.parse_args()
    
    suite = AutoTestSuite()
    
    # シグナルハンドラ設定（Ctrl+C対応）
    def signal_handler(sig, frame):
        print("\n⏹️ テストを中断中...")
        suite.cleanup()
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if args.web_only:
            # Webアプリテストのみ
            suite.start_web_app()
            success = suite.test_web_app_endpoints()
        elif args.quick:
            # クイックテスト
            success = (suite.test_script_functionality() and 
                      suite.test_demo_workflow() and
                      suite.start_web_app() and
                      suite.test_web_app_endpoints())
        else:
            # 完全テスト
            success = suite.run_all_tests()
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n💥 予期しないエラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()