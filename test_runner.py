#!/usr/bin/env python3
"""
自動テスト実行スクリプト
全てのテストを実行し、結果をレポートとして出力
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime


class TestRunner:
    """テスト実行管理クラス"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'duration': 0,
            'test_files': {}
        }
    
    def _parse_pytest_output(self, output):
        """pytestの出力からテスト結果を解析"""
        import re
        
        # 結果サマリーのパターン (例: "3 failed, 19 passed, 1 deselected")
        summary_pattern = r'(\d+) failed, (\d+) passed|(\d+) passed'
        match = re.search(summary_pattern, output)
        
        if match:
            if match.group(1) and match.group(2):  # failed and passed
                self.test_results['failed'] = int(match.group(1))
                self.test_results['passed'] = int(match.group(2))
            elif match.group(3):  # only passed
                self.test_results['passed'] = int(match.group(3))
                self.test_results['failed'] = 0
            
            self.test_results['total_tests'] = self.test_results['passed'] + self.test_results['failed']
        
        # スキップされたテストを探す
        skip_pattern = r'(\d+) skipped'
        skip_match = re.search(skip_pattern, output)
        if skip_match:
            self.test_results['skipped'] = int(skip_match.group(1))
    
    def run_unit_tests(self):
        """ユニットテストを実行"""
        print("🧪 ユニットテストを実行中...")
        
        start_time = time.time()
        
        # pytestを実行
        cmd = [
            sys.executable, "-m", "pytest", 
            "tests/", 
            "-v", 
            "--tb=short",
            "--json-report",
            "--json-report-file=test_results.json",
            "-m", "not slow"  # 重いテストは除外
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            self.test_results['duration'] = time.time() - start_time
            self.test_results['exit_code'] = result.returncode
            self.test_results['stdout'] = result.stdout
            self.test_results['stderr'] = result.stderr
            
            # JSONレポートがあれば読み込み
            json_report_path = self.project_root / "test_results.json"
            if json_report_path.exists():
                try:
                    with open(json_report_path, 'r') as f:
                        pytest_report = json.load(f)
                        summary = pytest_report.get('summary', {})
                        self.test_results['total_tests'] = summary.get('total', 0)
                        self.test_results['passed'] = summary.get('passed', 0)
                        self.test_results['failed'] = summary.get('failed', 0)
                        self.test_results['skipped'] = summary.get('skipped', 0)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"⚠️ JSONレポート読み込みエラー: {e}")
                    # stdout/stderrから情報を抽出
                    self._parse_pytest_output(result.stdout)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ テスト実行エラー: {e}")
            return False
    
    def run_integration_tests(self):
        """統合テストを実行（軽量版）"""
        print("🔗 統合テストを実行中...")
        
        # 基本的な統合チェック
        integration_results = {
            'config_loading': self.test_config_loading(),
            'script_imports': self.test_script_imports(),
            'web_app_startup': self.test_web_app_startup()
        }
        
        self.test_results['integration'] = integration_results
        
        # すべての統合テストが成功したかチェック
        return all(integration_results.values())
    
    def test_config_loading(self):
        """設定ファイルの読み込みテスト"""
        try:
            import yaml
            config_files = list((self.project_root / "configs").glob("*.yaml"))
            
            for config_file in config_files:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    # 必須フィールドをチェック
                    if 'merge_method' not in config or 'models' not in config:
                        return False
            
            return len(config_files) > 0
            
        except Exception as e:
            print(f"設定ファイル読み込みエラー: {e}")
            return False
    
    def test_script_imports(self):
        """スクリプトのインポートテスト"""
        scripts_to_test = [
            'scripts.merge_models',
            'scripts.evaluate', 
            'scripts.quantize',
            'scripts.experiment_tracker'
        ]
        
        for script in scripts_to_test:
            try:
                __import__(script)
            except Exception as e:
                print(f"インポートエラー {script}: {e}")
                return False
        
        return True
    
    def test_web_app_startup(self):
        """Webアプリの起動テスト"""
        try:
            # Streamlitのインポートテスト
            import streamlit as st
            
            # アプリクラスのインポートテスト
            from web.app import LLMMergeLabApp
            
            # インスタンス作成テスト
            app = LLMMergeLabApp()
            
            return True
            
        except Exception as e:
            print(f"Webアプリ起動テスト失敗: {e}")
            return False
    
    def run_performance_tests(self):
        """パフォーマンステスト（軽量版）"""
        print("⚡ パフォーマンステストを実行中...")
        
        performance_results = {}
        
        # スクリプトの起動時間測定
        scripts_to_test = [
            ('scripts/merge_models.py', ['--help']),
            ('scripts/evaluate.py', ['--help']),
            ('scripts/experiment_tracker.py', ['--help'])
        ]
        
        for script, args in scripts_to_test:
            start_time = time.time()
            try:
                result = subprocess.run(
                    [sys.executable, script] + args, 
                    capture_output=True, 
                    cwd=self.project_root,
                    timeout=10
                )
                end_time = time.time()
                
                performance_results[script] = {
                    'startup_time': end_time - start_time,
                    'success': result.returncode == 0
                }
                
            except subprocess.TimeoutExpired:
                performance_results[script] = {
                    'startup_time': 10.0,
                    'success': False,
                    'error': 'timeout'
                }
        
        self.test_results['performance'] = performance_results
        
        # すべてのスクリプトが5秒以内に起動することを確認
        return all(
            result['startup_time'] < 5.0 and result['success'] 
            for result in performance_results.values()
        )
    
    def generate_report(self):
        """テストレポートを生成"""
        print("\n" + "="*60)
        print("📊 テスト結果レポート")
        print("="*60)
        
        # 基本統計
        total = self.test_results['total_tests']
        passed = self.test_results['passed']
        failed = self.test_results['failed']
        skipped = self.test_results['skipped']
        duration = self.test_results['duration']
        
        print(f"🧪 ユニットテスト:")
        print(f"   総数: {total}")
        print(f"   ✅ 成功: {passed}")
        print(f"   ❌ 失敗: {failed}")
        print(f"   ⏭️  スキップ: {skipped}")
        print(f"   ⏱️  実行時間: {duration:.2f}秒")
        
        if failed > 0:
            success_rate = (passed / total) * 100 if total > 0 else 0
            print(f"   📈 成功率: {success_rate:.1f}%")
        
        # 統合テスト結果
        if 'integration' in self.test_results:
            print(f"\n🔗 統合テスト:")
            for test_name, result in self.test_results['integration'].items():
                status = "✅" if result else "❌"
                print(f"   {status} {test_name}")
        
        # パフォーマンステスト結果
        if 'performance' in self.test_results:
            print(f"\n⚡ パフォーマンステスト:")
            for script, result in self.test_results['performance'].items():
                status = "✅" if result['success'] else "❌"
                time_str = f"{result['startup_time']:.2f}s"
                print(f"   {status} {script}: {time_str}")
        
        # 総合判定
        unit_success = failed == 0 and total > 0
        integration_success = all(self.test_results.get('integration', {}).values())
        performance_success = all(
            r['success'] for r in self.test_results.get('performance', {}).values()
        )
        
        overall_success = unit_success and integration_success and performance_success
        
        print(f"\n{'='*60}")
        if overall_success:
            print("🎉 すべてのテストが成功しました！")
        else:
            print("⚠️  一部のテストが失敗しました。")
        print(f"{'='*60}\n")
        
        # レポートファイルを保存
        report_path = self.project_root / "test_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 詳細レポート: {report_path}")
        
        return overall_success
    
    def run_all_tests(self):
        """すべてのテストを実行"""
        print("🚀 LLM Merge Lab 自動テストを開始します...")
        print(f"📅 実行日時: {self.test_results['timestamp']}")
        print()
        
        # テスト実行
        unit_success = self.run_unit_tests()
        integration_success = self.run_integration_tests()
        performance_success = self.run_performance_tests()
        
        # レポート生成
        overall_success = self.generate_report()
        
        return overall_success


def main():
    """メイン関数"""
    runner = TestRunner()
    
    try:
        success = runner.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  テストが中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 予期しないエラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()