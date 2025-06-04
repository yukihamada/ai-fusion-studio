#!/usr/bin/env python3
"""
Webアプリのスクリーンショットを自動撮影
"""

import time
import subprocess
import os
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
import sys

def capture_web_screenshots():
    """Webアプリのスクリーンショットを撮影"""
    
    # Streamlitアプリを起動
    print("🌐 Webアプリを起動中...")
    web_process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", 
        "web/app.py", 
        "--server.port", "8503",
        "--server.headless", "true"
    ])
    
    # 起動待機
    time.sleep(10)
    
    try:
        # Seleniumの設定
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # ヘッドレスモード
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # ChromeDriverを起動
        try:
            driver = webdriver.Chrome(options=chrome_options)
        except:
            # ChromeDriverが見つからない場合はSafariを試す
            print("ChromeDriverが見つかりません。Safariを使用します...")
            driver = webdriver.Safari()
            driver.set_window_size(1920, 1080)
        
        # 保存先ディレクトリ
        screenshots_dir = Path("docs/images")
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. ダッシュボード画面
        print("📸 ダッシュボード画面を撮影中...")
        driver.get("http://localhost:8503")
        time.sleep(5)
        driver.save_screenshot(str(screenshots_dir / "dashboard.png"))
        
        # 2. 新しい実験画面
        print("📸 新しい実験画面を撮影中...")
        # サイドバーの「新しい実験」をクリック
        try:
            new_exp_button = driver.find_element(By.XPATH, "//span[contains(text(), '🚀 新しい実験')]")
            new_exp_button.click()
            time.sleep(3)
        except:
            print("新しい実験ボタンが見つかりません")
        
        driver.save_screenshot(str(screenshots_dir / "new_experiment.png"))
        
        # 3. 実験結果画面
        print("📸 実験結果画面を撮影中...")
        try:
            results_button = driver.find_element(By.XPATH, "//span[contains(text(), '📈 実験結果')]")
            results_button.click()
            time.sleep(3)
        except:
            print("実験結果ボタンが見つかりません")
        
        driver.save_screenshot(str(screenshots_dir / "experiment_results.png"))
        
        # 4. 設定管理画面
        print("📸 設定管理画面を撮影中...")
        try:
            config_button = driver.find_element(By.XPATH, "//span[contains(text(), '⚙️ 設定管理')]")
            config_button.click()
            time.sleep(3)
        except:
            print("設定管理ボタンが見つかりません")
        
        driver.save_screenshot(str(screenshots_dir / "config_management.png"))
        
        print("✅ スクリーンショット撮影完了！")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
    
    finally:
        # クリーンアップ
        if 'driver' in locals():
            driver.quit()
        web_process.terminate()
        web_process.wait()

def create_mock_screenshots():
    """モックスクリーンショットを生成（Seleniumが使えない場合）"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
    
    # 日本語フォント設定
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    screenshots_dir = Path("docs/images")
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. ダッシュボード画面のモック
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # ヘッダー
    ax.add_patch(FancyBboxPatch((5, 85), 90, 10, boxstyle="round,pad=0.1", 
                                facecolor='#667eea', edgecolor='none'))
    ax.text(50, 90, '🤖 LLM Merge Lab', fontsize=24, color='white', 
            ha='center', va='center', fontweight='bold')
    ax.text(50, 87, '最強の日本語LLMを作る実験プラットフォーム', fontsize=14, 
            color='white', ha='center', va='center')
    
    # メトリクスカード
    metrics = [
        ('📊 総実験数', '15'),
        ('✅ 完了済み', '12'),
        ('🏆 最高スコア', '8.5'),
        ('📈 成功率', '80%')
    ]
    
    for i, (label, value) in enumerate(metrics):
        x = 10 + i * 22
        ax.add_patch(FancyBboxPatch((x, 65), 18, 12, boxstyle="round,pad=0.5",
                                    facecolor='#f8f9fa', edgecolor='#e9ecef'))
        ax.text(x + 9, 73, label, fontsize=12, ha='center')
        ax.text(x + 9, 69, value, fontsize=20, ha='center', fontweight='bold')
    
    # グラフエリア
    ax.add_patch(FancyBboxPatch((10, 20), 40, 35, boxstyle="round,pad=0.5",
                                facecolor='white', edgecolor='#e9ecef'))
    ax.text(30, 52, 'MT-Benchスコア分布', fontsize=14, ha='center', fontweight='bold')
    
    # 棒グラフ（簡易版）
    for i in range(5):
        height = np.random.randint(20, 35)
        ax.add_patch(patches.Rectangle((15 + i*6, 25), 4, height, 
                                      facecolor='#667eea', alpha=0.7))
    
    ax.add_patch(FancyBboxPatch((55, 20), 40, 35, boxstyle="round,pad=0.5",
                                facecolor='white', edgecolor='#e9ecef'))
    ax.text(75, 52, '手法別性能比較', fontsize=14, ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(screenshots_dir / 'dashboard.png', dpi=150, bbox_inches='tight', 
                facecolor='#f0f2f6')
    plt.close()
    
    # 2. 新しい実験画面のモック
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # ヘッダー
    ax.text(50, 92, '🚀 新しい実験を開始', fontsize=20, ha='center', fontweight='bold')
    
    # 推奨設定カード
    cards = [
        {
            'title': 'Gemma × Qwen SLERP',
            'desc': 'コミュニティで8.5/10のスコアを記録',
            'time': '30分',
            'difficulty': '初級'
        },
        {
            'title': 'EvoLLM-JP再現',
            'desc': '日本語×数理の進化的マージ',
            'time': '60分',
            'difficulty': '中級'
        },
        {
            'title': 'Gemma + Swallow LoRA',
            'desc': '簡単実装で日本語の自然さを向上',
            'time': '20分',
            'difficulty': '初級'
        }
    ]
    
    for i, card in enumerate(cards):
        y = 70 - i * 25
        ax.add_patch(FancyBboxPatch((10, y), 80, 18, boxstyle="round,pad=0.5",
                                    facecolor='white', edgecolor='#e9ecef'))
        ax.text(15, y + 14, card['title'], fontsize=16, fontweight='bold')
        ax.text(15, y + 10, card['desc'], fontsize=12, color='#666')
        ax.text(15, y + 6, f"⏱️ {card['time']} | 📊 {card['difficulty']}", fontsize=10)
        
        # 実行ボタン
        ax.add_patch(FancyBboxPatch((75, y + 6), 12, 6, boxstyle="round,pad=0.3",
                                    facecolor='#667eea', edgecolor='none'))
        ax.text(81, y + 9, '▶️ 実行', fontsize=10, color='white', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(screenshots_dir / 'new_experiment.png', dpi=150, bbox_inches='tight',
                facecolor='#f0f2f6')
    plt.close()
    
    # 3. 実験結果画面のモック
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # ヘッダー
    ax.text(50, 92, '📈 実験結果詳細', fontsize=20, ha='center', fontweight='bold')
    
    # 実験カード
    experiments = [
        {
            'id': 'slerp_20250604_120000',
            'method': 'SLERP',
            'score': '8.52',
            'status': '完了'
        },
        {
            'id': 'evolutionary_20250603',
            'method': 'Evolutionary',
            'score': '8.31',
            'status': '完了'
        }
    ]
    
    for i, exp in enumerate(experiments):
        y = 70 - i * 30
        ax.add_patch(FancyBboxPatch((10, y), 80, 25, boxstyle="round,pad=0.5",
                                    facecolor='white', edgecolor='#e9ecef'))
        ax.text(15, y + 20, f"📊 {exp['id']}", fontsize=14, fontweight='bold')
        ax.text(15, y + 15, f"手法: {exp['method']} | ステータス: {exp['status']}", fontsize=12)
        
        # スコア表示
        ax.add_patch(FancyBboxPatch((70, y + 12), 15, 10, boxstyle="round,pad=0.3",
                                    facecolor='#d4edda', edgecolor='none'))
        ax.text(77.5, y + 17, f"{exp['score']}/10", fontsize=14, ha='center', va='center')
        
        # 詳細情報
        ax.text(15, y + 8, "カテゴリ別: Writing 8.0 | Reasoning 7.5 | Coding 8.2", 
                fontsize=10, color='#666')
        ax.text(15, y + 4, "量子化: 4.5GB → 1.2GB (3.75x圧縮)", fontsize=10, color='#666')
    
    plt.tight_layout()
    plt.savefig(screenshots_dir / 'experiment_results.png', dpi=150, bbox_inches='tight',
                facecolor='#f0f2f6')
    plt.close()
    
    print("✅ モックスクリーンショット生成完了！")

if __name__ == "__main__":
    try:
        # Seleniumでの撮影を試みる
        capture_web_screenshots()
    except Exception as e:
        print(f"Seleniumでの撮影に失敗: {e}")
        print("モックスクリーンショットを生成します...")
        create_mock_screenshots()