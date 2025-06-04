#!/usr/bin/env python3
"""
実験追跡・可視化システム
マージ実験の結果を追跡し、比較・分析を行う
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meirio', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ExperimentTracker:
    """実験追跡・管理クラス"""
    
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # 実験データベース
        self.db_path = self.experiments_dir / "experiments_db.json"
        self.experiments = self._load_experiments()
    
    def _load_experiments(self) -> List[Dict]:
        """既存の実験データをロード"""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                return json.load(f)
        return []
    
    def _save_experiments(self) -> None:
        """実験データを保存"""
        with open(self.db_path, 'w') as f:
            json.dump(self.experiments, f, indent=2, ensure_ascii=False)
    
    def register_experiment(self, experiment_data: Dict) -> str:
        """新しい実験を登録"""
        # 実験IDを生成
        experiment_id = experiment_data.get('experiment_id', 
                                           f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # 実験データを拡張
        experiment = {
            'id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'running',
            **experiment_data
        }
        
        self.experiments.append(experiment)
        self._save_experiments()
        
        logger.info(f"実験を登録: {experiment_id}")
        return experiment_id
    
    def update_experiment(self, experiment_id: str, updates: Dict) -> None:
        """実験データを更新"""
        for exp in self.experiments:
            if exp.get('id', exp.get('experiment_id')) == experiment_id:
                exp.update(updates)
                exp['last_updated'] = datetime.now().isoformat()
                break
        
        self._save_experiments()
        logger.info(f"実験を更新: {experiment_id}")
    
    def add_evaluation_results(self, experiment_id: str, eval_results: Dict) -> None:
        """評価結果を追加"""
        for exp in self.experiments:
            if exp.get('id', exp.get('experiment_id')) == experiment_id:
                if 'evaluations' not in exp:
                    exp['evaluations'] = {}
                exp['evaluations'].update(eval_results)
                exp['status'] = 'completed'
                break
        
        self._save_experiments()
        logger.info(f"評価結果を追加: {experiment_id}")
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """特定の実験データを取得"""
        for exp in self.experiments:
            exp_id = exp.get('id', exp.get('experiment_id'))
            if exp_id and exp_id == experiment_id:
                return exp
        return None
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """複数の実験を比較"""
        comparison_data = []
        
        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                row = {
                    'experiment_id': exp_id,
                    'method': exp.get('merge_method', 'unknown'),
                    'models': ', '.join([m['name'] for m in exp.get('models', [])]),
                    'timestamp': exp.get('timestamp', '')
                }
                
                # 評価結果を追加
                if 'evaluations' in exp:
                    if 'mt_bench_jp' in exp['evaluations']:
                        row['mt_bench_score'] = exp['evaluations']['mt_bench_jp'].get('overall_score', 0)
                    if 'mathematical_reasoning' in exp['evaluations']:
                        row['math_accuracy'] = exp['evaluations']['mathematical_reasoning'].get('accuracy', 0)
                
                # 量子化情報を追加
                if 'quantization' in exp:
                    row['quantized_size_gb'] = exp['quantization'].get('quantized_size_gb', 0)
                    row['compression_ratio'] = exp['quantization'].get('compression_ratio', 0)
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def visualize_experiments(self, output_dir: str = "experiments/visualizations") -> None:
        """実験結果を可視化"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # データフレームに変換
        df = self.compare_experiments([exp['id'] for exp in self.experiments])
        
        if df.empty:
            logger.warning("可視化するデータがありません")
            return
        
        # 1. MT-Benchスコアの推移
        if 'mt_bench_score' in df.columns:
            self._plot_mt_bench_trends(df, output_path)
        
        # 2. モデルサイズ vs 性能
        if 'quantized_size_gb' in df.columns and 'mt_bench_score' in df.columns:
            self._plot_size_vs_performance(df, output_path)
        
        # 3. マージ手法別の比較
        self._plot_method_comparison(df, output_path)
        
        # 4. インタラクティブダッシュボード
        self._create_interactive_dashboard(df, output_path)
        
        logger.info(f"可視化を生成: {output_path}")
    
    def _plot_mt_bench_trends(self, df: pd.DataFrame, output_path: Path) -> None:
        """MT-Benchスコアの推移をプロット"""
        plt.figure(figsize=(12, 6))
        
        # 時系列でソート
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_sorted = df.sort_values('timestamp')
        
        # マージ手法別にプロット
        for method in df_sorted['method'].unique():
            method_df = df_sorted[df_sorted['method'] == method]
            plt.plot(method_df['timestamp'], method_df['mt_bench_score'], 
                    marker='o', label=method, linewidth=2)
        
        plt.xlabel('実験日時')
        plt.ylabel('MT-Bench スコア')
        plt.title('Japanese MT-Bench スコアの推移')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path / 'mt_bench_trends.png', dpi=300)
        plt.close()
    
    def _plot_size_vs_performance(self, df: pd.DataFrame, output_path: Path) -> None:
        """モデルサイズ vs 性能のプロット"""
        plt.figure(figsize=(10, 8))
        
        # 散布図
        scatter = plt.scatter(df['quantized_size_gb'], df['mt_bench_score'], 
                            c=df['compression_ratio'], s=100, cmap='viridis', alpha=0.7)
        
        # カラーバー
        cbar = plt.colorbar(scatter)
        cbar.set_label('圧縮率', rotation=270, labelpad=20)
        
        # 各点にラベルを追加
        for idx, row in df.iterrows():
            plt.annotate(row['method'], 
                        (row['quantized_size_gb'], row['mt_bench_score']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('量子化後サイズ (GB)')
        plt.ylabel('MT-Bench スコア')
        plt.title('モデルサイズ vs 性能')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'size_vs_performance.png', dpi=300)
        plt.close()
    
    def _plot_method_comparison(self, df: pd.DataFrame, output_path: Path) -> None:
        """マージ手法別の比較"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('マージ手法別比較', fontsize=16)
        
        # 1. MT-Benchスコア
        if 'mt_bench_score' in df.columns:
            ax = axes[0, 0]
            method_scores = df.groupby('method')['mt_bench_score'].agg(['mean', 'std'])
            method_scores.plot(kind='bar', y='mean', yerr='std', ax=ax, legend=False)
            ax.set_title('MT-Bench スコア（平均）')
            ax.set_ylabel('スコア')
            ax.set_xlabel('')
        
        # 2. 数学推論精度
        if 'math_accuracy' in df.columns:
            ax = axes[0, 1]
            method_math = df.groupby('method')['math_accuracy'].agg(['mean', 'std'])
            method_math.plot(kind='bar', y='mean', yerr='std', ax=ax, legend=False, color='orange')
            ax.set_title('数学推論精度（平均）')
            ax.set_ylabel('精度')
            ax.set_xlabel('')
        
        # 3. モデルサイズ
        if 'quantized_size_gb' in df.columns:
            ax = axes[1, 0]
            method_size = df.groupby('method')['quantized_size_gb'].mean()
            method_size.plot(kind='bar', ax=ax, color='green')
            ax.set_title('量子化後サイズ（平均）')
            ax.set_ylabel('サイズ (GB)')
            ax.set_xlabel('')
        
        # 4. 実験数
        ax = axes[1, 1]
        method_count = df['method'].value_counts()
        method_count.plot(kind='bar', ax=ax, color='purple')
        ax.set_title('実験数')
        ax.set_ylabel('回数')
        ax.set_xlabel('マージ手法')
        
        plt.tight_layout()
        plt.savefig(output_path / 'method_comparison.png', dpi=300)
        plt.close()
    
    def _create_interactive_dashboard(self, df: pd.DataFrame, output_path: Path) -> None:
        """インタラクティブなダッシュボードを作成"""
        # Plotlyで複数のサブプロットを作成
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('MT-Benchスコア分布', '手法別性能', 
                          'モデルサイズ vs 性能', '実験タイムライン'),
            specs=[[{'type': 'box'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # 1. MT-Benchスコア分布（ボックスプロット）
        if 'mt_bench_score' in df.columns:
            for method in df['method'].unique():
                method_df = df[df['method'] == method]
                fig.add_trace(
                    go.Box(y=method_df['mt_bench_score'], name=method),
                    row=1, col=1
                )
        
        # 2. 手法別平均性能
        if 'mt_bench_score' in df.columns:
            method_avg = df.groupby('method')['mt_bench_score'].mean().reset_index()
            fig.add_trace(
                go.Bar(x=method_avg['method'], y=method_avg['mt_bench_score'],
                      name='平均スコア'),
                row=1, col=2
            )
        
        # 3. モデルサイズ vs 性能
        if 'quantized_size_gb' in df.columns and 'mt_bench_score' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['quantized_size_gb'], 
                    y=df['mt_bench_score'],
                    mode='markers+text',
                    text=df['method'],
                    textposition='top center',
                    marker=dict(size=10, color=df['compression_ratio'], 
                              colorscale='Viridis', showscale=True)
                ),
                row=2, col=1
            )
        
        # 4. 実験タイムライン
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'mt_bench_score' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'], 
                    y=df['mt_bench_score'],
                    mode='lines+markers',
                    name='スコア推移',
                    line=dict(width=2)
                ),
                row=2, col=2
            )
        
        # レイアウト設定
        fig.update_layout(
            title_text="LLMマージ実験ダッシュボード",
            showlegend=False,
            height=800
        )
        
        # 軸ラベル
        fig.update_xaxes(title_text="マージ手法", row=1, col=1)
        fig.update_yaxes(title_text="MT-Benchスコア", row=1, col=1)
        fig.update_xaxes(title_text="マージ手法", row=1, col=2)
        fig.update_yaxes(title_text="平均スコア", row=1, col=2)
        fig.update_xaxes(title_text="モデルサイズ (GB)", row=2, col=1)
        fig.update_yaxes(title_text="MT-Benchスコア", row=2, col=1)
        fig.update_xaxes(title_text="実験日時", row=2, col=2)
        fig.update_yaxes(title_text="MT-Benchスコア", row=2, col=2)
        
        # 保存
        fig.write_html(output_path / 'experiment_dashboard.html')
        
        # 個別の詳細ビューも作成
        self._create_detailed_view(df, output_path)
    
    def _create_detailed_view(self, df: pd.DataFrame, output_path: Path) -> None:
        """詳細な実験結果ビュー"""
        # 実験ごとの詳細カード
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLMマージ実験詳細</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .experiment-card { 
                    border: 1px solid #ddd; 
                    padding: 15px; 
                    margin: 10px 0; 
                    border-radius: 5px;
                }
                .metric { 
                    display: inline-block; 
                    margin: 5px 10px; 
                    padding: 5px 10px;
                    background: #f0f0f0;
                    border-radius: 3px;
                }
                .high-score { color: green; font-weight: bold; }
                .medium-score { color: orange; }
                .low-score { color: red; }
            </style>
        </head>
        <body>
            <h1>LLMマージ実験詳細レポート</h1>
        """
        
        for _, exp in df.iterrows():
            score_class = 'high-score' if exp.get('mt_bench_score', 0) >= 7 else \
                         'medium-score' if exp.get('mt_bench_score', 0) >= 5 else 'low-score'
            
            html_content += f"""
            <div class="experiment-card">
                <h3>実験ID: {exp['experiment_id']}</h3>
                <p><strong>手法:</strong> {exp['method']}</p>
                <p><strong>モデル:</strong> {exp['models']}</p>
                <p><strong>実行日時:</strong> {exp['timestamp']}</p>
                <div class="metrics">
            """
            
            if 'mt_bench_score' in exp:
                html_content += f'<span class="metric {score_class}">MT-Bench: {exp["mt_bench_score"]:.2f}/10</span>'
            
            if 'math_accuracy' in exp:
                html_content += f'<span class="metric">数学精度: {exp["math_accuracy"]:.2%}</span>'
            
            if 'quantized_size_gb' in exp:
                html_content += f'<span class="metric">サイズ: {exp["quantized_size_gb"]:.2f}GB</span>'
            
            if 'compression_ratio' in exp:
                html_content += f'<span class="metric">圧縮率: {exp["compression_ratio"]:.2f}x</span>'
            
            html_content += """
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path / 'experiment_details.html', 'w') as f:
            f.write(html_content)
    
    def generate_leaderboard(self) -> pd.DataFrame:
        """リーダーボードを生成"""
        # 'id'または'experiment_id'フィールドを持つ実験のみをフィルタ
        valid_experiments = []
        for exp in self.experiments:
            exp_id = exp.get('id', exp.get('experiment_id'))
            if exp_id:
                valid_experiments.append((exp_id, exp))
        
        if not valid_experiments:
            logger.warning("有効な実験データがありません")
            return pd.DataFrame()
        
        df = self.compare_experiments([exp_id for exp_id, _ in valid_experiments])
        
        if df.empty or 'mt_bench_score' not in df.columns:
            return pd.DataFrame()
        
        # スコアでソート
        leaderboard = df.sort_values('mt_bench_score', ascending=False)
        
        # ランキングを追加
        leaderboard['rank'] = range(1, len(leaderboard) + 1)
        
        # 必要な列のみ選択
        columns = ['rank', 'experiment_id', 'method', 'models', 'mt_bench_score']
        if 'math_accuracy' in leaderboard.columns:
            columns.append('math_accuracy')
        if 'quantized_size_gb' in leaderboard.columns:
            columns.append('quantized_size_gb')
        
        return leaderboard[columns]


def main():
    parser = argparse.ArgumentParser(description='実験追跡・可視化')
    parser.add_argument('--action', choices=['register', 'update', 'compare', 'visualize', 'leaderboard'],
                       required=True, help='実行するアクション')
    parser.add_argument('--experiment-id', help='実験ID')
    parser.add_argument('--data', help='JSON形式のデータ')
    parser.add_argument('--experiments-dir', default='experiments', help='実験ディレクトリ')
    parser.add_argument('--output-dir', default='experiments/visualizations', 
                       help='可視化の出力ディレクトリ')
    
    args = parser.parse_args()
    
    tracker = ExperimentTracker(args.experiments_dir)
    
    if args.action == 'register':
        if args.data:
            data = json.loads(args.data)
            exp_id = tracker.register_experiment(data)
            print(f"実験を登録しました: {exp_id}")
    
    elif args.action == 'update':
        if args.experiment_id and args.data:
            data = json.loads(args.data)
            tracker.update_experiment(args.experiment_id, data)
            print(f"実験を更新しました: {args.experiment_id}")
    
    elif args.action == 'visualize':
        tracker.visualize_experiments(args.output_dir)
        print(f"可視化を生成しました: {args.output_dir}")
    
    elif args.action == 'leaderboard':
        leaderboard = tracker.generate_leaderboard()
        print("\n=== LLMマージ実験リーダーボード ===")
        print(leaderboard.to_string(index=False))
        
        # CSVでも保存
        leaderboard.to_csv('experiments/leaderboard.csv', index=False)
        print("\nリーダーボードをCSVに保存しました: experiments/leaderboard.csv")


if __name__ == '__main__':
    main()