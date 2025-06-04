#!/usr/bin/env python3
"""
継続的インテグレーション (CI) スクリプト
コード変更時の自動テスト＆デプロイメント
"""

import os
import sys
import json
import time
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class CIEventHandler(FileSystemEventHandler):
    """ファイル変更監視イベントハンドラー"""
    
    def __init__(self, ci_system):
        self.ci_system = ci_system
        self.last_trigger = 0
        self.debounce_seconds = 5  # 5秒間のデバウンス
    
    def on_modified(self, event):
        """ファイル変更時の処理"""
        if event.is_directory:
            return
        
        # Python、YAML、設定ファイルのみ監視
        if not any(event.src_path.endswith(ext) for ext in ['.py', '.yaml', '.yml', '.json']):
            return
        
        # __pycache__やテスト結果ファイルは無視
        if any(ignore in event.src_path for ignore in ['__pycache__', 'test_results', '.pytest_cache']):
            return
        
        current_time = time.time()
        if current_time - self.last_trigger < self.debounce_seconds:
            return
        
        self.last_trigger = current_time
        print(f"\n📝 ファイル変更検出: {event.src_path}")
        
        # 非同期でCI実行
        import threading
        thread = threading.Thread(target=self.ci_system.run_incremental_ci, args=(event.src_path,))
        thread.daemon = True
        thread.start()


class ContinuousIntegration:
    """継続的インテグレーションシステム"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.ci_results = []
        self.file_hashes = {}
        self.load_file_hashes()
    
    def log(self, message, level="INFO"):
        """ログ出力"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] CI-{level}: {message}")
    
    def get_file_hash(self, file_path):
        """ファイルのハッシュを計算"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return None
    
    def load_file_hashes(self):
        """前回のファイルハッシュを読み込み"""
        hash_file = self.project_root / '.ci_hashes.json'
        if hash_file.exists():
            try:
                with open(hash_file, 'r') as f:
                    self.file_hashes = json.load(f)
            except:
                self.file_hashes = {}
    
    def save_file_hashes(self):
        """ファイルハッシュを保存"""
        hash_file = self.project_root / '.ci_hashes.json'
        try:
            with open(hash_file, 'w') as f:
                json.dump(self.file_hashes, f, indent=2)
        except Exception as e:
            self.log(f"ハッシュ保存エラー: {e}", "ERROR")
    
    def get_changed_files(self):
        """変更されたファイルを検出"""
        changed_files = []
        
        # Pythonファイルをチェック
        for py_file in self.project_root.rglob('*.py'):
            if any(ignore in str(py_file) for ignore in ['__pycache__', '.pytest_cache', 'test_results']):
                continue
            
            current_hash = self.get_file_hash(py_file)
            file_key = str(py_file.relative_to(self.project_root))
            
            if current_hash and current_hash != self.file_hashes.get(file_key):
                changed_files.append(py_file)
                self.file_hashes[file_key] = current_hash
        
        # 設定ファイルをチェック
        for config_file in self.project_root.rglob('*.yaml'):
            current_hash = self.get_file_hash(config_file)
            file_key = str(config_file.relative_to(self.project_root))
            
            if current_hash and current_hash != self.file_hashes.get(file_key):
                changed_files.append(config_file)
                self.file_hashes[file_key] = current_hash
        
        return changed_files
    
    def run_quick_tests(self, changed_file=None):
        """クイックテスト実行"""
        self.log("⚡ クイックテストを実行中...")
        
        start_time = time.time()
        
        tests_to_run = []
        
        if changed_file:
            # 変更ファイルに応じてテストを選択
            file_path = str(changed_file)
            
            if 'scripts/' in file_path:
                tests_to_run.append('script_functionality')
            elif 'web/' in file_path:
                tests_to_run.append('web_app')
            elif 'configs/' in file_path:
                tests_to_run.append('demo_workflow')
            else:
                tests_to_run = ['unit_tests', 'script_functionality']
        else:
            tests_to_run = ['unit_tests', 'script_functionality', 'demo_workflow']
        
        results = {}
        overall_success = True
        
        for test in tests_to_run:
            try:
                if test == 'unit_tests':
                    result = self.run_unit_tests_quick()
                elif test == 'script_functionality':
                    result = self.test_script_help()
                elif test == 'demo_workflow':
                    result = self.test_demo_config()
                elif test == 'web_app':
                    result = self.test_web_imports()
                else:
                    result = True
                
                results[test] = result
                overall_success = overall_success and result
                
                status = "✅" if result else "❌"
                self.log(f"{status} {test}")
                
            except Exception as e:
                self.log(f"❌ {test} エラー: {e}", "ERROR")
                results[test] = False
                overall_success = False
        
        duration = time.time() - start_time
        
        ci_result = {
            'timestamp': datetime.now().isoformat(),
            'type': 'quick_test',
            'changed_file': str(changed_file) if changed_file else None,
            'duration': duration,
            'results': results,
            'overall_success': overall_success
        }
        
        self.ci_results.append(ci_result)
        self.save_ci_results()
        
        status = "✅ 成功" if overall_success else "❌ 失敗"
        self.log(f"{status}: クイックテスト完了 ({duration:.1f}s)")
        
        return overall_success
    
    def run_unit_tests_quick(self):
        """クイック単体テスト"""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/", "-x", "--tb=no", "-q"
            ], capture_output=True, text=True, cwd=self.project_root, timeout=60)
            
            return result.returncode == 0
        except:
            return False
    
    def test_script_help(self):
        """スクリプトヘルプテスト"""
        scripts = [
            'scripts/merge_models.py',
            'scripts/evaluate.py',
            'scripts/experiment_tracker.py'
        ]
        
        for script in scripts:
            try:
                result = subprocess.run([
                    sys.executable, script, '--help'
                ], capture_output=True, timeout=10)
                
                if result.returncode != 0:
                    return False
            except:
                return False
        
        return True
    
    def test_demo_config(self):
        """デモ設定テスト"""
        try:
            import yaml
            config_file = self.project_root / 'configs' / 'demo_config.yaml'
            
            if not config_file.exists():
                return False
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # 必須フィールドの確認
            required_fields = ['merge_method', 'models', 'output_path']
            return all(field in config for field in required_fields)
        except:
            return False
    
    def test_web_imports(self):
        """Webアプリインポートテスト"""
        try:
            import importlib.util
            
            # web/app.pyのインポートテスト
            spec = importlib.util.spec_from_file_location(
                "web_app", self.project_root / "web" / "app.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            return True
        except:
            return False
    
    def run_full_ci(self):
        """完全CI実行"""
        self.log("🚀 完全CI実行を開始...")
        
        start_time = time.time()
        
        # 変更ファイル検出
        changed_files = self.get_changed_files()
        if changed_files:
            self.log(f"📝 {len(changed_files)}個のファイル変更を検出")
            for file_path in changed_files:
                self.log(f"  - {file_path.relative_to(self.project_root)}")
        else:
            self.log("📝 ファイル変更なし")
        
        # テスト実行
        test_success = self.run_quick_tests()
        
        # ハッシュ保存
        self.save_file_hashes()
        
        duration = time.time() - start_time
        
        ci_result = {
            'timestamp': datetime.now().isoformat(),
            'type': 'full_ci',
            'changed_files': [str(f.relative_to(self.project_root)) for f in changed_files],
            'duration': duration,
            'test_success': test_success,
            'overall_success': test_success
        }
        
        self.ci_results.append(ci_result)
        self.save_ci_results()
        
        status = "✅ 成功" if test_success else "❌ 失敗"
        self.log(f"{status}: 完全CI完了 ({duration:.1f}s)")
        
        return test_success
    
    def run_incremental_ci(self, changed_file_path):
        """インクリメンタルCI実行"""
        self.log(f"🔄 インクリメンタルCI実行: {Path(changed_file_path).name}")
        
        start_time = time.time()
        
        # 変更ファイルに応じたテスト実行
        test_success = self.run_quick_tests(Path(changed_file_path))
        
        duration = time.time() - start_time
        
        ci_result = {
            'timestamp': datetime.now().isoformat(),
            'type': 'incremental_ci',
            'trigger_file': changed_file_path,
            'duration': duration,
            'test_success': test_success
        }
        
        self.ci_results.append(ci_result)
        self.save_ci_results()
        
        if test_success:
            self.log("✅ インクリメンタルCI成功")
        else:
            self.log("❌ インクリメンタルCI失敗 - 修正が必要です", "ERROR")
        
        return test_success
    
    def save_ci_results(self):
        """CI結果を保存"""
        results_file = self.project_root / 'ci_results.json'
        try:
            # 最新100件のみ保持
            recent_results = self.ci_results[-100:]
            with open(results_file, 'w') as f:
                json.dump(recent_results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.log(f"CI結果保存エラー: {e}", "ERROR")
    
    def generate_ci_report(self):
        """CI実行レポート生成"""
        if not self.ci_results:
            self.log("CI実行履歴がありません")
            return
        
        print("\n" + "="*60)
        print("📊 CI実行レポート")
        print("="*60)
        
        # 最近の実行結果
        recent_results = self.ci_results[-10:]  # 最新10件
        
        total_runs = len(recent_results)
        successful_runs = len([r for r in recent_results if r.get('overall_success', False)])
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        
        print(f"📈 成功率: {success_rate:.1f}% ({successful_runs}/{total_runs})")
        print(f"⏱️ 平均実行時間: {sum(r.get('duration', 0) for r in recent_results) / total_runs:.1f}s")
        print()
        
        print("🕒 最近の実行履歴:")
        for result in reversed(recent_results[-5:]):  # 最新5件
            timestamp = result['timestamp'][:19].replace('T', ' ')
            ci_type = result['type']
            success = result.get('overall_success', False)
            duration = result.get('duration', 0)
            
            status = "✅" if success else "❌"
            print(f"  {status} {timestamp} [{ci_type}] {duration:.1f}s")
        
        print("="*60)
    
    def start_file_watching(self):
        """ファイル監視開始"""
        self.log("👀 ファイル変更監視を開始...")
        
        event_handler = CIEventHandler(self)
        observer = Observer()
        
        # 監視対象ディレクトリ
        watch_dirs = ['scripts', 'web', 'configs', 'tests']
        
        for watch_dir in watch_dirs:
            dir_path = self.project_root / watch_dir
            if dir_path.exists():
                observer.schedule(event_handler, str(dir_path), recursive=True)
                self.log(f"📁 監視対象: {watch_dir}/")
        
        observer.start()
        
        try:
            self.log("🔄 ファイル監視中... (Ctrl+Cで停止)")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.log("⏹️ ファイル監視を停止")
        finally:
            observer.stop()
            observer.join()


def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM Merge Lab 継続的インテグレーション')
    parser.add_argument('--watch', action='store_true', help='ファイル変更監視モード')
    parser.add_argument('--full', action='store_true', help='完全CI実行')
    parser.add_argument('--quick', action='store_true', help='クイックテスト実行')
    parser.add_argument('--report', action='store_true', help='CI実行レポート表示')
    
    args = parser.parse_args()
    
    ci_system = ContinuousIntegration()
    
    try:
        if args.watch:
            # ファイル監視モード
            ci_system.start_file_watching()
        elif args.full:
            # 完全CI実行
            success = ci_system.run_full_ci()
            sys.exit(0 if success else 1)
        elif args.quick:
            # クイックテスト
            success = ci_system.run_quick_tests()
            sys.exit(0 if success else 1)
        elif args.report:
            # レポート表示
            ci_system.generate_ci_report()
        else:
            # デフォルト: クイックテスト
            success = ci_system.run_quick_tests()
            if success:
                print("\n✅ CI実行成功 - コードの品質が保たれています")
            else:
                print("\n❌ CI実行失敗 - 修正が必要です")
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print("\n⏹️ CI実行が中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 CI実行エラー: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()