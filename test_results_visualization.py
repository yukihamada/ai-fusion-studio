#!/usr/bin/env python3
"""
テスト結果の可視化スクリプト
グラフとチャートでテスト結果を表示
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path
from datetime import datetime

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_test_summary_chart():
    """テストサマリーチャート作成"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # テスト結果円グラフ
    test_results = {
        '成功': 5,
        '失敗': 3,
        'エラー': 0
    }
    
    colors = ['#4CAF50', '#F44336', '#FF9800']
    wedges, texts, autotexts = ax1.pie(
        test_results.values(), 
        labels=test_results.keys(),
        colors=colors,
        autopct='%1.0f%%',
        startangle=90
    )
    
    ax1.set_title('テスト結果サマリー\n(総テスト数: 8)', fontsize=14, pad=20)
    
    # 成功率ゲージ
    success_rate = 62.5
    ax2.clear()
    
    # ゲージ背景
    ax2.add_patch(Rectangle((0, 0), 100, 20, facecolor='#E0E0E0'))
    # 成功率バー
    color = '#4CAF50' if success_rate >= 80 else '#FF9800' if success_rate >= 60 else '#F44336'
    ax2.add_patch(Rectangle((0, 0), success_rate, 20, facecolor=color))
    
    ax2.text(50, 10, f'{success_rate}%', ha='center', va='center', 
             fontsize=20, fontweight='bold', color='white' if success_rate > 50 else 'black')
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 20)
    ax2.axis('off')
    ax2.set_title('総合成功率', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('test_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_chart():
    """パフォーマンスチャート作成"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scripts = ['merge_models.py', 'evaluate.py', 'experiment_tracker.py', 'quantize.py']
    times = [17, 8, 3, 7]
    threshold = 10
    
    # バーの色（閾値超えは赤）
    colors = ['#F44336' if t > threshold else '#4CAF50' for t in times]
    
    bars = ax.barh(scripts, times, color=colors)
    
    # 閾値ライン
    ax.axvline(x=threshold, color='orange', linestyle='--', linewidth=2, label='基準値 (10秒)')
    
    # 値をバーの上に表示
    for bar, time in zip(bars, times):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{time}s', ha='left', va='center')
    
    ax.set_xlabel('起動時間 (秒)', fontsize=12)
    ax.set_title('スクリプト起動パフォーマンス', fontsize=14, pad=20)
    ax.legend()
    ax.set_xlim(0, 20)
    
    plt.tight_layout()
    plt.savefig('performance_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_quality_chart():
    """モデル品質チャート作成"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # MT-Benchカテゴリ別スコア
    categories = ['Writing', 'Reasoning', 'Coding', 'Overall']
    scores = [7.2, 6.5, 6.9, 6.8]
    
    bars = ax1.bar(categories, scores, color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0'])
    
    # スコアをバーの上に表示
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{score}', ha='center', va='bottom')
    
    ax1.set_ylim(0, 10)
    ax1.set_ylabel('スコア (0-10)', fontsize=12)
    ax1.set_title('MT-Bench カテゴリ別スコア', fontsize=14, pad=20)
    ax1.grid(axis='y', alpha=0.3)
    
    # モデル圧縮効果
    sizes = ['圧縮前\n(2.1GB)', '圧縮後\n(0.8GB)']
    values = [2.1, 0.8]
    
    bars2 = ax2.bar(sizes, values, color=['#F44336', '#4CAF50'], width=0.6)
    
    # 圧縮率を表示
    ax2.text(0.5, 1.5, '2.6x\n圧縮', ha='center', va='center', 
             fontsize=16, fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax2.set_ylabel('サイズ (GB)', fontsize=12)
    ax2.set_title('モデル圧縮効果', fontsize=14, pad=20)
    ax2.set_ylim(0, 2.5)
    
    plt.tight_layout()
    plt.savefig('model_quality_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_system_health_dashboard():
    """システムヘルスダッシュボード作成"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    components = ['コアエンジン', 'Web UI', '評価システム', '実験追跡', 'CI/CD']
    reliability = [95, 100, 90, 70, 80]
    status = ['good', 'good', 'good', 'warning', 'warning']
    
    # 色マッピング
    color_map = {
        'good': '#4CAF50',
        'warning': '#FF9800',
        'error': '#F44336'
    }
    colors = [color_map[s] for s in status]
    
    # 横棒グラフ
    bars = ax.barh(components, reliability, color=colors)
    
    # パーセンテージ表示
    for bar, rel in zip(bars, reliability):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{rel}%', ha='left', va='center', fontweight='bold')
    
    # 状態アイコン
    for i, (comp, stat) in enumerate(zip(components, status)):
        icon = '✅' if stat == 'good' else '⚠️'
        ax.text(-2, i, icon, ha='right', va='center', fontsize=16)
    
    ax.set_xlim(-5, 105)
    ax.set_xlabel('信頼性 (%)', fontsize=12)
    ax.set_title('システムヘルスステータス', fontsize=16, pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # 凡例
    good_patch = mpatches.Patch(color='#4CAF50', label='正常')
    warning_patch = mpatches.Patch(color='#FF9800', label='注意')
    error_patch = mpatches.Patch(color='#F44336', label='エラー')
    ax.legend(handles=[good_patch, warning_patch, error_patch], loc='lower right')
    
    plt.tight_layout()
    plt.savefig('system_health_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_test_timeline():
    """テストタイムライン作成"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # テストデータ
    tests = [
        ('ユニットテスト', 0, 40, 'error'),
        ('スクリプト機能', 40, 35, 'success'),
        ('デモワークフロー', 75, 0.5, 'success'),
        ('実験追跡', 75.5, 3, 'failed'),
        ('モデル品質', 78.5, 0.5, 'success'),
        ('Webアプリ起動', 79, 2, 'success'),
        ('Webエンドポイント', 81, 0.5, 'success'),
        ('パフォーマンス', 81.5, 25, 'failed')
    ]
    
    color_map = {
        'success': '#4CAF50',
        'failed': '#F44336',
        'error': '#FF5722'
    }
    
    # ガントチャート風表示
    for i, (name, start, duration, status) in enumerate(tests):
        ax.barh(i, duration, left=start, height=0.8, 
                color=color_map[status], alpha=0.8, edgecolor='black')
        ax.text(start + duration/2, i, name, ha='center', va='center', 
                fontweight='bold', fontsize=10)
    
    ax.set_ylim(-0.5, len(tests) - 0.5)
    ax.set_xlim(0, 110)
    ax.set_xlabel('経過時間 (秒)', fontsize=12)
    ax.set_title('テスト実行タイムライン', fontsize=16, pad=20)
    ax.set_yticks([])
    ax.grid(axis='x', alpha=0.3)
    
    # 凡例
    success_patch = mpatches.Patch(color='#4CAF50', label='成功')
    failed_patch = mpatches.Patch(color='#F44336', label='失敗')
    error_patch = mpatches.Patch(color='#FF5722', label='エラー')
    ax.legend(handles=[success_patch, failed_patch, error_patch], 
              loc='upper right', bbox_to_anchor=(1.1, 1))
    
    # 合計時間表示
    ax.text(55, -1.5, f'総実行時間: 106.5秒', ha='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
    
    plt.tight_layout()
    plt.savefig('test_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """メイン実行"""
    print("📊 テスト結果の可視化を開始...")
    
    # 各種チャートを生成
    create_test_summary_chart()
    print("✅ テストサマリーチャート生成完了: test_summary.png")
    
    create_performance_chart()
    print("✅ パフォーマンスチャート生成完了: performance_chart.png")
    
    create_model_quality_chart()
    print("✅ モデル品質チャート生成完了: model_quality_chart.png")
    
    create_system_health_dashboard()
    print("✅ システムヘルスダッシュボード生成完了: system_health_dashboard.png")
    
    create_test_timeline()
    print("✅ テストタイムライン生成完了: test_timeline.png")
    
    print("\n🎉 すべてのグラフ生成が完了しました！")
    print("生成されたファイル:")
    print("  - test_summary.png")
    print("  - performance_chart.png")
    print("  - model_quality_chart.png")
    print("  - system_health_dashboard.png")
    print("  - test_timeline.png")

if __name__ == "__main__":
    main()