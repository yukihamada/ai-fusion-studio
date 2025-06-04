#!/usr/bin/env python3
"""
AI Fusion Studio モダンWebダッシュボード
Streamlitベースのプロフェッショナルなユーザーインターフェース
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import yaml
import os
import sys
from pathlib import Path
from datetime import datetime
import subprocess

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

# Streamlit設定
st.set_page_config(
    page_title="AI Fusion Studio 🚀",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS - 高コントラスト・可読性重視デザイン
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(30, 58, 138, 0.3);
    }
    
    .main-header h1 {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.1rem !important;
        font-weight: 400 !important;
        color: rgba(255,255,255,0.95) !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        margin-bottom: 0 !important;
    }
    
    .metric-card {
        background: white;
        padding: 2rem 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        margin: 1rem 0;
        transition: all 0.2s ease;
        border-left: 4px solid #3b82f6;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: #475569;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .experiment-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        transition: all 0.2s ease;
    }
    
    .experiment-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    
    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    
    .status-completed {
        background: #059669;
        color: white;
    }
    
    .status-running {
        background: #d97706;
        color: white;
    }
    
    .status-failed {
        background: #dc2626;
        color: white;
    }
    
    .status-unknown {
        background: #6b7280;
        color: white;
    }
    
    .dashboard-section {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #f1f5f9;
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .recent-experiment {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        transition: all 0.2s ease;
    }
    
    .recent-experiment:hover {
        background: #f1f5f9;
        transform: translateX(4px);
    }
    
    .score-highlight {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    /* Streamlit 컴포넌트 스타일링 */
    .stSelectbox > div > div {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        color: #1e293b;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
    }
    
    .stMetric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    .stMetric [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* 텍스트 가독성 강화 */
    h1, h2, h3, h4, h5, h6 {
        color: #1e293b !important;
    }
    
    p, div, span {
        color: #334155 !important;
    }
    
    .stMarkdown {
        color: #334155 !important;
    }
</style>
""", unsafe_allow_html=True)


class AIFusionStudioApp:
    """メインアプリケーションクラス"""
    
    def __init__(self):
        self.experiments_dir = Path("experiments")
        self.configs_dir = Path("configs")
        self.models_dir = Path("models")
        
        # ディレクトリ作成
        for dir_path in [self.experiments_dir, self.configs_dir, self.models_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def load_experiments(self):
        """実験データを読み込み"""
        db_path = self.experiments_dir / "experiments_db.json"
        if db_path.exists():
            with open(db_path, 'r') as f:
                return json.load(f)
        return []
    
    def load_configs(self):
        """利用可能な設定ファイルを取得"""
        configs = []
        for config_file in self.configs_dir.glob("*.yaml"):
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    config['filename'] = config_file.name
                    configs.append(config)
            except Exception as e:
                st.error(f"設定ファイル読み込みエラー: {config_file.name} - {e}")
        return configs
    
    def run_experiment(self, config_file, skip_steps=None):
        """実験を実行"""
        cmd = ["python", "scripts/run_experiment.py", str(config_file)]
        if skip_steps:
            cmd.extend(["--skip"] + skip_steps)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            return result.returncode == 0, result.stdout, result.stderr
        except Exception as e:
            return False, "", str(e)


def main():
    app = AIFusionStudioApp()
    
    # ヘッダー
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI Fusion Studio</h1>
        <p>最強のAIモデルを融合させるプロフェッショナルスタジオ</p>
    </div>
    """, unsafe_allow_html=True)
    
    # サイドバー
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/667eea/white?text=AI+Fusion", width=200)
        
        page = st.selectbox(
            "🔧 機能を選択",
            ["📊 ダッシュボード", "🚀 新しい実験", "📈 実験結果", "⚙️ 設定管理", "📚 ガイド", "🔍 実験比較", "📁 データ管理"]
        )
        
        st.markdown("---")
        
        # クイックアクション
        st.markdown("### ⚡ クイックアクション")
        
        if st.button("🎯 推奨実験実行", use_container_width=True):
            with st.spinner("Gemma × Qwen実験を実行中..."):
                success, stdout, stderr = app.run_experiment("configs/gemma_qwen_slerp.yaml")
                if success:
                    st.success("実験が完了しました！")
                else:
                    st.error(f"実験が失敗しました: {stderr}")
        
        if st.button("📊 リーダーボード更新", use_container_width=True):
            st.rerun()
    
    # メインコンテンツ
    if page == "📊 ダッシュボード":
        show_dashboard(app)
    elif page == "🚀 新しい実験":
        show_new_experiment(app)
    elif page == "📈 実験結果":
        show_experiment_results(app)
    elif page == "⚙️ 設定管理":
        show_config_management(app)
    elif page == "📚 ガイド":
        show_guide()
    elif page == "🔍 実験比較":
        show_experiment_comparison(app.load_experiments())
    elif page == "📁 データ管理":
        show_data_management(app)


def show_dashboard(app):
    """モダンダッシュボード表示"""
    # メインヘッダー - 高コントラスト・可読性重視
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AI Fusion Studio</h1>
        <p>最強のAIモデルを融合させるプロフェッショナルスタジオ</p>
        <div style="margin-top: 1.5rem;">
            <span style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem; 
                         color: white; font-weight: 500; font-size: 0.9rem;">🔬 Advanced AI Fusion</span>
            <span style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem; 
                         color: white; font-weight: 500; font-size: 0.9rem;">🏢 Enterprise Ready</span>
            <span style="background: rgba(255,255,255,0.25); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem; 
                         color: white; font-weight: 500; font-size: 0.9rem;">⭐ Production Grade</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 実験データを読み込み
    experiments = app.load_experiments()
    
    if not experiments:
        st.markdown("""
        <div class="dashboard-section">
            <div class="section-title">🎯 実験を開始しましょう</div>
            <p style="font-size: 1.2rem; color: #64748b; margin-bottom: 2rem;">
            まだ実験が実行されていません。サイドバーから「🚀 新しい実験」を選択して、
            最初のモデルマージ実験を開始してください。
            </p>
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                        padding: 1.5rem; border-radius: 12px; border-left: 4px solid #0ea5e9;">
                <strong>💡 推奨:</strong> Gemma × Qwen SLERP実験から始めることをお勧めします
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # 統計情報 - モダンなカードデザイン
    st.markdown('<div class="section-title">📊 実験統計</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    total_experiments = len(experiments)
    completed_experiments = len([e for e in experiments if e.get('status') == 'completed'])
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">📊 総実験数</div>
        </div>
        """.format(total_experiments), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">✅ 完了済み</div>
        </div>
        """.format(completed_experiments), unsafe_allow_html=True)
    
    with col3:
        if completed_experiments > 0:
            best_score = max([e.get('evaluations', {}).get('mt_bench_jp', {}).get('overall_score', 0) 
                            for e in experiments if e.get('status') == 'completed'])
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.2f}</div>
                <div class="metric-label">🏆 最高スコア</div>
            </div>
            """.format(best_score), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">--</div>
                <div class="metric-label">🏆 最高スコア</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        success_rate = (completed_experiments / total_experiments * 100) if total_experiments > 0 else 0
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1f}%</div>
            <div class="metric-label">📈 成功率</div>
        </div>
        """.format(success_rate), unsafe_allow_html=True)
    
    # 実験結果の可視化
    if completed_experiments > 0:
        df = create_experiments_dataframe(experiments)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 MT-Benchスコア分布")
            if 'mt_bench_score' in df.columns:
                fig = px.histogram(df, x='mt_bench_score', nbins=10, 
                                 title="MT-Benchスコア分布",
                                 color_discrete_sequence=['#667eea'])
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⚖️ 手法別性能比較")
            if 'method' in df.columns and 'mt_bench_score' in df.columns:
                method_avg = df.groupby('method')['mt_bench_score'].mean().reset_index()
                fig = px.bar(method_avg, x='method', y='mt_bench_score',
                           title="手法別平均スコア",
                           color='mt_bench_score',
                           color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
        
        # 最近の実験 - モダンなデザイン
        st.markdown('<div class="section-title">🕒 最近の実験</div>', unsafe_allow_html=True)
        recent_experiments = sorted(experiments, 
                                  key=lambda x: x.get('timestamp', ''), 
                                  reverse=True)[:5]
        
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        for exp in recent_experiments:
            show_modern_experiment_card(exp)
        st.markdown('</div>', unsafe_allow_html=True)


def show_new_experiment(app):
    """新しい実験設定画面"""
    st.header("🚀 新しい実験を開始")
    
    # 設定選択
    configs = app.load_configs()
    
    if not configs:
        st.error("設定ファイルがありません。configs/ディレクトリに設定ファイルを配置してください。")
        return
    
    # タブで表示
    tab1, tab2 = st.tabs(["📋 既存設定から選択", "✏️ カスタム設定"])
    
    with tab1:
        # 推奨設定のカード表示
        st.subheader("🌟 推奨設定")
        
        recommended_configs = [
            {
                'name': 'Gemma × Qwen SLERP',
                'filename': 'gemma_qwen_slerp.yaml',
                'description': 'コミュニティで8.5/10のスコアを記録した最強の組み合わせ',
                'expected_score': 8.5,
                'use_case': '汎用日本語タスク',
                'difficulty': '初級',
                'time': '30分'
            },
            {
                'name': 'EvoLLM-JP再現',
                'filename': 'evolllm_jp_reproduction.yaml',
                'description': '日本語×数理の進化的マージで数学タスクを強化',
                'expected_score': 7.3,
                'use_case': '数理推論',
                'difficulty': '中級',
                'time': '60分'
            },
            {
                'name': 'Gemma + Swallow LoRA',
                'filename': 'gemma_swallow_lora.yaml',
                'description': '簡単実装で日本語の自然さを向上',
                'expected_score': 7.5,
                'use_case': '日本語チャット',
                'difficulty': '初級',
                'time': '20分'
            }
        ]
        
        for config in recommended_configs:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="experiment-card">
                        <h4>{config['name']}</h4>
                        <p>{config['description']}</p>
                        <small>💡 用途: {config['use_case']} | ⏱️ 所要時間: {config['time']} | 📊 期待スコア: {config['expected_score']}/10</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    difficulty_color = {'初級': '🟢', '中級': '🟡', '上級': '🔴'}
                    st.markdown(f"**難易度**: {difficulty_color.get(config['difficulty'], '🔘')} {config['difficulty']}")
                
                with col3:
                    if st.button(f"▶️ 実行", key=f"run_{config['filename']}"):
                        run_experiment_flow(app, config['filename'])
    
    with tab2:
        st.subheader("✏️ カスタム実験設定")
        
        # 基本設定
        col1, col2 = st.columns(2)
        
        with col1:
            merge_method = st.selectbox(
                "マージ手法",
                ["slerp", "evolutionary", "lora"],
                help="マージアルゴリズムを選択"
            )
            
            model1 = st.text_input(
                "ベースモデル 1",
                "google/gemma-3-4b-it",
                help="HuggingFace Hub上のモデル名"
            )
            
            weight1 = st.slider("モデル1の重み", 0.0, 1.0, 0.6, 0.1)
        
        with col2:
            output_name = st.text_input(
                "出力モデル名",
                "my_custom_merge",
                help="マージ後のモデル名"
            )
            
            model2 = st.text_input(
                "ベースモデル 2", 
                "Qwen/Qwen3-4B-Instruct"
            )
            
            weight2 = 1.0 - weight1
            st.metric("モデル2の重み", f"{weight2:.1f}")
        
        # 評価設定
        st.subheader("📊 評価設定")
        benchmarks = st.multiselect(
            "実行するベンチマーク",
            ["mt-bench-jp", "jglue", "math"],
            default=["mt-bench-jp"]
        )
        
        # 実行ボタン
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 リアルタイム実行", type="primary"):
                # 設定ファイルを生成
                custom_config = {
                    'merge_method': merge_method,
                    'output_path': f'models/{output_name}',
                    'models': [
                        {'name': model1, 'weight': weight1},
                        {'name': model2, 'weight': weight2}
                    ],
                    'alpha': weight1,
                    'evaluation': {
                        'benchmarks': benchmarks
                    }
                }
                
                # 一時設定ファイルを作成
                temp_config_path = app.configs_dir / f"temp_{output_name}.yaml"
                with open(temp_config_path, 'w') as f:
                    yaml.dump(custom_config, f)
                
                run_experiment_with_realtime_logs(app, temp_config_path.name)
                
        with col2:
            if st.button("⚡ 簡易実行"):
                # 設定ファイルを生成
                custom_config = {
                    'merge_method': merge_method,
                    'output_path': f'models/{output_name}',
                    'models': [
                        {'name': model1, 'weight': weight1},
                        {'name': model2, 'weight': weight2}
                    ],
                    'alpha': weight1,
                    'evaluation': {
                        'benchmarks': benchmarks
                    }
                }
                
                # 一時設定ファイルを作成
                temp_config_path = app.configs_dir / f"temp_{output_name}.yaml"
                with open(temp_config_path, 'w') as f:
                    yaml.dump(custom_config, f)
                
                run_experiment_flow(app, temp_config_path.name)


def show_experiment_results(app):
    """実験結果詳細表示・エクスポート機能付き"""
    st.markdown('<div class="section-title">📈 実験結果詳細</div>', unsafe_allow_html=True)
    
    experiments = app.load_experiments()
    
    if not experiments:
        st.markdown("""
        <div class="dashboard-section">
            <p style="text-align: center; color: #64748b; font-size: 1.1rem;">
            📊 実験結果がありません。新しい実験を実行してください。
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # エクスポート機能
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### 📊 実験データの管理")
    with col2:
        if st.button("📁 JSON エクスポート", use_container_width=True):
            export_experiments_json(experiments)
    with col3:
        if st.button("📊 CSV エクスポート", use_container_width=True):
            export_experiments_csv(experiments)
    
    # フィルタリング
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "ステータスでフィルタ",
            ["すべて", "completed", "running", "failed"]
        )
    
    with col2:
        method_filter = st.selectbox(
            "手法でフィルタ",
            ["すべて"] + list(set([e.get('merge_method', '') for e in experiments]))
        )
    
    with col3:
        sort_by = st.selectbox(
            "ソート順",
            ["日時(新しい順)", "日時(古い順)", "スコア(高い順)", "スコア(低い順)"]
        )
    
    # フィルタリング適用
    filtered_experiments = experiments
    
    if status_filter != "すべて":
        filtered_experiments = [e for e in filtered_experiments if e.get('status') == status_filter]
    
    if method_filter != "すべて":
        filtered_experiments = [e for e in filtered_experiments if e.get('merge_method') == method_filter]
    
    # ソート
    if sort_by == "日時(新しい順)":
        filtered_experiments.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    elif sort_by == "日時(古い順)":
        filtered_experiments.sort(key=lambda x: x.get('timestamp', ''))
    elif sort_by == "スコア(高い順)":
        filtered_experiments.sort(
            key=lambda x: x.get('evaluations', {}).get('mt_bench_jp', {}).get('overall_score', 0), 
            reverse=True
        )
    elif sort_by == "スコア(低い順)":
        filtered_experiments.sort(
            key=lambda x: x.get('evaluations', {}).get('mt_bench_jp', {}).get('overall_score', 0)
        )
    
    # タブで結果表示とエクスポート・比較機能を分離
    tab1, tab2 = st.tabs(["📊 実験結果一覧", "🔍 実験比較"])
    
    with tab1:
        # 結果表示
        for experiment in filtered_experiments:
            show_detailed_experiment_card(experiment)
    
    with tab2:
        # 実験比較機能
        show_experiment_comparison(filtered_experiments)


def show_config_management(app):
    """設定管理画面"""
    st.header("⚙️ 設定管理")
    
    configs = app.load_configs()
    
    tab1, tab2 = st.tabs(["📋 既存設定", "➕ 新規作成"])
    
    with tab1:
        st.subheader("📋 利用可能な設定ファイル")
        
        for config in configs:
            with st.expander(f"📄 {config['filename']}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.code(yaml.dump(config, default_flow_style=False), language='yaml')
                
                with col2:
                    if st.button("✏️ 編集", key=f"edit_{config['filename']}"):
                        st.session_state[f"editing_{config['filename']}"] = True
                    
                    if st.button("🗑️ 削除", key=f"delete_{config['filename']}"):
                        try:
                            os.remove(app.configs_dir / config['filename'])
                            st.success("設定ファイルを削除しました")
                            st.rerun()
                        except Exception as e:
                            st.error(f"削除に失敗: {e}")
    
    with tab2:
        st.subheader("➕ 新しい設定ファイルを作成")
        
        # テンプレート選択
        template = st.selectbox(
            "テンプレートを選択",
            ["空の設定", "SLERP設定", "LoRA設定", "Evolutionary設定"]
        )
        
        filename = st.text_input("ファイル名", "my_config.yaml")
        
        # 設定エディタ
        if template == "SLERP設定":
            default_config = """merge_method: slerp
output_path: models/my_merged_model

models:
  - name: model1/name
    weight: 0.6
  - name: model2/name
    weight: 0.4

alpha: 0.6

evaluation:
  benchmarks:
    - mt-bench-jp
"""
        else:
            default_config = """merge_method: slerp
output_path: models/new_model

models:
  - name: model1
  - name: model2
"""
        
        config_text = st.text_area(
            "設定内容 (YAML)", 
            default_config, 
            height=400
        )
        
        if st.button("💾 保存"):
            try:
                # YAML形式チェック
                yaml.safe_load(config_text)
                
                # ファイル保存
                with open(app.configs_dir / filename, 'w') as f:
                    f.write(config_text)
                
                st.success(f"設定ファイル {filename} を保存しました")
                st.rerun()
                
            except yaml.YAMLError as e:
                st.error(f"YAML形式エラー: {e}")
            except Exception as e:
                st.error(f"保存エラー: {e}")


def show_guide():
    """ガイド表示"""
    st.header("📚 AI Fusion Studio ガイド")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 クイックスタート", "🔧 使い方", "💡 Tips", "❓ FAQ"])
    
    with tab1:
        st.markdown("""
        ## 🚀 5分で始めるAI融合
        
        ### 1. 推奨実験から始める
        
        **Gemma × Qwen SLERP** がおすすめです：
        - コミュニティで実証済み（MT-Bench 8.5/10）
        - 実装が簡単
        - 汎用的な日本語タスクに最適
        
        ### 2. サイドバーから「推奨実験実行」をクリック
        
        ### 3. 結果を確認
        - ダッシュボードで進捗をチェック
        - 完了後は実験結果タブで詳細を確認
        
        ### 4. 他の組み合わせも試す
        - 数理推論強化なら「EvoLLM-JP再現」
        - 日本語チャット改善なら「Gemma + Swallow LoRA」
        """)
    
    with tab2:
        st.markdown("""
        ## 🔧 詳細な使い方
        
        ### マージ手法
        
        **SLERP（球面線形補間）**
        - 2つのモデルを滑らかに補間
        - 安定した結果が期待できる
        - 推奨：初心者向け
        
        **Evolutionary Merge**
        - 遺伝的アルゴリズムで最適化
        - 高性能だが時間がかかる
        - 推奨：中級者向け
        
        **LoRA統合**
        - 既存モデルに軽量な改善を追加
        - 高速で安全
        - 推奨：特定機能の改善
        
        ### 評価指標
        
        **MT-Bench JP**
        - 日本語タスクの総合評価
        - 0-10のスコア
        - 7.0以上で実用レベル
        
        **数理推論**
        - 数学問題の正答率
        - 論理的思考能力を測定
        """)
    
    with tab3:
        st.markdown("""
        ## 💡 成功のためのTips
        
        ### 1. モデル選択
        - **日本語強化**: 海外モデル + 日本語特化モデル
        - **推論強化**: チャットモデル + 数理特化モデル
        - **バランス重視**: 同サイズの異系列モデル
        
        ### 2. 重み配分
        - **0.6:0.4** - バランスが良い
        - **0.7:0.3** - ベースモデル重視
        - **0.5:0.5** - 均等配分
        
        ### 3. 量子化
        - **AWQ 4bit** - 品質と効率のバランス
        - **2bit** - 最大圧縮（品質低下あり）
        - **GGUF** - CPU実行向け
        
        ### 4. トラブルシューティング
        - メモリ不足 → CPU実行に切り替え
        - 低スコア → 重み配分を調整
        - エラー → モデルの互換性確認
        """)
    
    with tab4:
        st.markdown("""
        ## ❓ よくある質問
        
        **Q: どのくらい時間がかかりますか？**
        A: 
        - SLERP: 20-30分
        - LoRA: 10-20分  
        - Evolutionary: 60-90分
        
        **Q: 必要なメモリは？**
        A:
        - 4Bモデル: 8GB以上推奨
        - 7Bモデル: 16GB以上推奨
        - CPU実行も可能（時間はかかります）
        
        **Q: 商用利用できますか？**
        A: 各モデルのライセンスを確認してください。
        
        **Q: カスタムモデルを使いたい**
        A: HuggingFace Hub上の任意のモデルが使用可能です。
        
        **Q: 実験が失敗しました**
        A: 
        1. モデル名のスペルチェック
        2. インターネット接続確認
        3. メモリ不足の場合はCPU実行を試す
        """)


def run_experiment_flow(app, config_filename):
    """実験実行フロー"""
    with st.spinner(f"実験を実行中: {config_filename}"):
        # プログレスバー
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # ステップごとの実行
        steps = [
            ("モデル検証", 20),
            ("モデルマージ", 40), 
            ("評価実行", 70),
            ("量子化", 90),
            ("結果保存", 100)
        ]
        
        for step_name, progress in steps:
            status_text.text(f"🔄 {step_name}中...")
            progress_bar.progress(progress)
            
            # 実際の処理時間をシミュレート
            import time
            time.sleep(1)
        
        # 実際の実験実行
        success, stdout, stderr = app.run_experiment(f"configs/{config_filename}")
        
        if success:
            st.success("✅ 実験が完了しました！")
            st.info("結果は「実験結果」タブで確認できます。")
        else:
            st.error(f"❌ 実験が失敗しました: {stderr}")
            with st.expander("詳細エラー"):
                st.code(stderr)


def show_experiment_card(experiment):
    """実験カードを表示"""
    status_class = f"status-{experiment.get('status', 'unknown')}"
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown(f"**{experiment.get('id', 'Unknown')}**")
        st.text(f"手法: {experiment.get('merge_method', 'N/A')}")
        
    with col2:
        status = experiment.get('status', 'unknown')
        st.markdown(f'<span class="status-badge {status_class}">{status}</span>', 
                   unsafe_allow_html=True)
    
    with col3:
        if 'evaluations' in experiment:
            score = experiment['evaluations'].get('mt_bench_jp', {}).get('overall_score', 0)
            st.metric("MT-Bench", f"{score:.2f}")


def show_modern_experiment_card(experiment):
    """モダンな実験カード表示"""
    exp_id = experiment.get('id', 'Unknown')
    method = experiment.get('merge_method', 'N/A')
    status = experiment.get('status', 'unknown')
    
    # ステータスに応じた色を設定
    status_colors = {
        'completed': '#10b981',
        'running': '#f59e0b', 
        'failed': '#ef4444',
        'unknown': '#6b7280'
    }
    
    status_color = status_colors.get(status, '#6b7280')
    
    # スコア取得
    score = 0
    if 'evaluations' in experiment:
        score = experiment['evaluations'].get('mt_bench_jp', {}).get('overall_score', 0)
    
    # モダンなカードデザイン
    st.markdown(f"""
    <div class="recent-experiment">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <div>
                <h4 style="margin: 0; color: #1e293b; font-size: 1.2rem;">{exp_id}</h4>
                <p style="margin: 0.25rem 0 0 0; color: #64748b; font-size: 0.95rem;">手法: <strong>{method}</strong></p>
            </div>
            <div style="text-align: right;">
                <span style="background: {status_color}; color: white; padding: 0.25rem 0.75rem; 
                           border-radius: 12px; font-size: 0.85rem; font-weight: 600; text-transform: uppercase;">
                    {status}
                </span>
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="color: #64748b; font-size: 0.9rem;">
                {experiment.get('timestamp', 'N/A')[:16] if experiment.get('timestamp') else 'N/A'}
            </div>
            {f'<div class="score-highlight">MT-Bench: {score:.2f}</div>' if score > 0 else '<div style="color: #94a3b8;">スコア未測定</div>'}
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_detailed_experiment_card(experiment):
    """詳細実験カードを表示"""
    with st.expander(f"📊 {experiment.get('id', 'Unknown')} - {experiment.get('merge_method', 'N/A')}"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("基本情報")
            st.write(f"**実験ID**: {experiment.get('id', 'N/A')}")
            st.write(f"**手法**: {experiment.get('merge_method', 'N/A')}")
            st.write(f"**ステータス**: {experiment.get('status', 'N/A')}")
            st.write(f"**実行日時**: {experiment.get('timestamp', 'N/A')}")
            
            if 'models' in experiment:
                st.subheader("使用モデル")
                for model in experiment['models']:
                    st.write(f"- {model.get('name', 'N/A')} (重み: {model.get('weight', 'N/A')})")
        
        with col2:
            if 'evaluations' in experiment:
                st.subheader("評価結果")
                evals = experiment['evaluations']
                
                if 'mt_bench_jp' in evals:
                    mt_score = evals['mt_bench_jp'].get('overall_score', 0)
                    st.metric("MT-Bench総合", f"{mt_score:.2f}/10")
                    
                    if 'category_scores' in evals['mt_bench_jp']:
                        st.write("**カテゴリ別スコア**:")
                        for cat, score in evals['mt_bench_jp']['category_scores'].items():
                            st.write(f"- {cat}: {score:.2f}")
                
                if 'mathematical_reasoning' in evals:
                    math_acc = evals['mathematical_reasoning'].get('accuracy', 0)
                    st.metric("数理推論精度", f"{math_acc:.2%}")
            
            if 'quantization' in experiment:
                st.subheader("量子化情報")
                quant = experiment['quantization']
                st.write(f"**手法**: {quant.get('method', 'N/A')}")
                st.write(f"**サイズ**: {quant.get('quantized_size_gb', 'N/A'):.2f} GB")
                st.write(f"**圧縮率**: {quant.get('compression_ratio', 'N/A'):.2f}x")


def create_experiments_dataframe(experiments):
    """実験データをDataFrameに変換"""
    data = []
    for exp in experiments:
        if exp.get('status') == 'completed':
            row = {
                'experiment_id': exp.get('id', ''),
                'method': exp.get('merge_method', ''),
                'timestamp': exp.get('timestamp', ''),
                'status': exp.get('status', ''),
                # デフォルト値を設定
                'mt_bench_score': 0,
                'math_accuracy': 0,
                'model_size_gb': 0,
                'compression_ratio': 0
            }
            
            if 'evaluations' in exp:
                if 'mt_bench_jp' in exp['evaluations']:
                    row['mt_bench_score'] = exp['evaluations']['mt_bench_jp'].get('overall_score', 0)
                if 'mathematical_reasoning' in exp['evaluations']:
                    row['math_accuracy'] = exp['evaluations']['mathematical_reasoning'].get('accuracy', 0)
            
            if 'quantization' in exp:
                row['model_size_gb'] = exp['quantization'].get('quantized_size_gb', 0)
                row['compression_ratio'] = exp['quantization'].get('compression_ratio', 0)
            
            data.append(row)
    
    # 空のDataFrameでも必要な列を持つようにする
    if not data:
        return pd.DataFrame(columns=['experiment_id', 'method', 'timestamp', 'status', 
                                    'mt_bench_score', 'math_accuracy', 'model_size_gb', 
                                    'compression_ratio'])
    
    return pd.DataFrame(data)


def export_experiments_json(experiments):
    """実験データをJSONでエクスポート"""
    import json
    from datetime import datetime
    
    export_data = {
        'export_timestamp': datetime.now().isoformat(),
        'total_experiments': len(experiments),
        'experiments': experiments
    }
    
    json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
    
    st.download_button(
        label="📁 JSON ファイルをダウンロード",
        data=json_str,
        file_name=f"llm_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    st.success("JSONエクスポートの準備が完了しました！")


def export_experiments_csv(experiments):
    """実験データをCSVでエクスポート"""
    from datetime import datetime
    
    df = create_experiments_dataframe(experiments)
    
    # より詳細なCSVデータを作成
    detailed_data = []
    for exp in experiments:
        row = {
            'experiment_id': exp.get('id', ''),
            'method': exp.get('merge_method', ''),
            'timestamp': exp.get('timestamp', ''),
            'status': exp.get('status', ''),
            'mt_bench_score': 0,
            'math_accuracy': 0,
            'model_size_gb': 0,
            'compression_ratio': 0,
            'models_used': '',
            'output_path': exp.get('output_path', ''),
            'description': exp.get('description', '')
        }
        
        # 評価結果
        if 'evaluations' in exp:
            if 'mt_bench_jp' in exp['evaluations']:
                row['mt_bench_score'] = exp['evaluations']['mt_bench_jp'].get('overall_score', 0)
            if 'mathematical_reasoning' in exp['evaluations']:
                row['math_accuracy'] = exp['evaluations']['mathematical_reasoning'].get('accuracy', 0)
        
        # 量子化情報
        if 'quantization' in exp:
            row['model_size_gb'] = exp['quantization'].get('quantized_size_gb', 0)
            row['compression_ratio'] = exp['quantization'].get('compression_ratio', 0)
        
        # 使用モデル
        if 'models' in exp:
            models_list = [f"{m.get('name', '')}({m.get('weight', '')})" for m in exp['models']]
            row['models_used'] = '; '.join(models_list)
        
        detailed_data.append(row)
    
    detailed_df = pd.DataFrame(detailed_data)
    csv_str = detailed_df.to_csv(index=False)
    
    st.download_button(
        label="📊 CSV ファイルをダウンロード",
        data=csv_str,
        file_name=f"llm_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    st.success("CSVエクスポートの準備が完了しました！")


def run_experiment_with_realtime_logs(app, config_filename):
    """リアルタイムログ付き実験実行"""
    import subprocess
    import time
    import threading
    import queue
    
    st.markdown("### 🚀 実験実行中...")
    
    # プログレスバーとステータス
    progress_bar = st.progress(0)
    status_container = st.empty()
    log_container = st.empty()
    
    # ログキューを作成
    log_queue = queue.Queue()
    
    def run_experiment():
        """実験を実際に実行する関数"""
        try:
            # 実際の実験スクリプトを実行
            process = subprocess.Popen(
                ["python", "scripts/run_experiment.py", f"configs/{config_filename}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=str(Path(__file__).parent.parent)
            )
            
            # 出力を逐次取得
            for line in iter(process.stdout.readline, ''):
                log_queue.put(('log', line.strip()))
                
            process.wait()
            log_queue.put(('done', process.returncode))
            
        except Exception as e:
            log_queue.put(('error', str(e)))
    
    # 実験を別スレッドで開始
    experiment_thread = threading.Thread(target=run_experiment)
    experiment_thread.start()
    
    # リアルタイムログ表示
    logs = []
    steps = [
        ("モデル検証", 10),
        ("モデルマージ", 40),
        ("評価実行", 70),
        ("量子化", 90),
        ("結果保存", 100)
    ]
    current_step = 0
    
    while experiment_thread.is_alive() or not log_queue.empty():
        try:
            # ログをチェック
            log_type, content = log_queue.get(timeout=0.1)
            
            if log_type == 'log':
                logs.append(content)
                
                # ステップ進行を判定
                if current_step < len(steps):
                    step_name, progress = steps[current_step]
                    status_container.text(f"🔄 {step_name}中...")
                    progress_bar.progress(progress)
                    current_step += 1
                
                # ログ表示（最新10行）
                recent_logs = logs[-10:]
                log_text = '\n'.join(recent_logs)
                log_container.text_area("📋 実行ログ", log_text, height=200)
                
            elif log_type == 'done':
                if content == 0:
                    progress_bar.progress(100)
                    status_container.success("✅ 実験が正常に完了しました！")
                else:
                    status_container.error(f"❌ 実験が失敗しました (エラーコード: {content})")
                break
                
            elif log_type == 'error':
                status_container.error(f"❌ エラーが発生しました: {content}")
                break
                
        except queue.Empty:
            continue
        except Exception as e:
            status_container.error(f"❌ 予期しないエラー: {e}")
            break
    
    experiment_thread.join()


def show_experiment_comparison(experiments):
    """実験比較機能"""
    st.markdown('<div class="section-title">🔍 実験比較</div>', unsafe_allow_html=True)
    
    if len(experiments) < 2:
        st.info("比較するには2つ以上の実験が必要です。")
        return
    
    # 比較する実験を選択
    col1, col2 = st.columns(2)
    
    with col1:
        exp1_options = [f"{exp.get('id', 'Unknown')} ({exp.get('merge_method', 'N/A')})" 
                      for exp in experiments]
        selected_exp1 = st.selectbox("実験1を選択", exp1_options, key="exp1")
        
    with col2:
        selected_exp2 = st.selectbox("実験2を選択", exp1_options, key="exp2")
    
    if selected_exp1 != selected_exp2:
        exp1 = experiments[exp1_options.index(selected_exp1)]
        exp2 = experiments[exp1_options.index(selected_exp2)]
        
        # 比較テーブル
        comparison_data = []
        
        metrics = [
            ("実験ID", "id", ""),
            ("手法", "merge_method", ""),
            ("ステータス", "status", ""),
            ("MT-Benchスコア", "evaluations.mt_bench_jp.overall_score", 0),
            ("数学精度", "evaluations.mathematical_reasoning.accuracy", 0),
            ("モデルサイズ(GB)", "quantization.quantized_size_gb", 0),
            ("圧縮率", "quantization.compression_ratio", 0)
        ]
        
        for metric_name, path, default in metrics:
            value1 = get_nested_value(exp1, path, default)
            value2 = get_nested_value(exp2, path, default)
            comparison_data.append({
                "項目": metric_name,
                "実験1": value1,
                "実験2": value2,
                "差分": calculate_diff(value1, value2)
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # 視覚的比較
        if all(get_nested_value(exp, "evaluations.mt_bench_jp.overall_score", 0) > 0 for exp in [exp1, exp2]):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 MT-Benchスコア比較")
                scores = [
                    get_nested_value(exp1, "evaluations.mt_bench_jp.overall_score", 0),
                    get_nested_value(exp2, "evaluations.mt_bench_jp.overall_score", 0)
                ]
                labels = [exp1.get('id', 'Exp1'), exp2.get('id', 'Exp2')]
                
                fig = px.bar(x=labels, y=scores, title="MT-Benchスコア比較")
                st.plotly_chart(fig, use_container_width=True)


def get_nested_value(data, path, default=None):
    """ネストされた辞書から値を取得"""
    keys = path.split('.')
    current = data
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def calculate_diff(val1, val2):
    """2つの値の差分を計算"""
    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
        diff = val2 - val1
        return f"{diff:+.2f}" if abs(diff) > 0.01 else "0.00"
    else:
        return "N/A" if val1 == val2 else "異なる"


def show_data_management(app):
    """データ管理ページ"""
    st.markdown('<div class="section-title">📁 データ管理</div>', unsafe_allow_html=True)
    
    experiments = app.load_experiments()
    
    # 統計情報
    st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("総実験数", len(experiments))
    with col2:
        completed = len([e for e in experiments if e.get('status') == 'completed'])
        st.metric("完了実験", completed)
    with col3:
        failed = len([e for e in experiments if e.get('status') == 'failed'])
        st.metric("失敗実験", failed)
    with col4:
        total_size = sum([e.get('quantization', {}).get('quantized_size_gb', 0) for e in experiments])
        st.metric("総データサイズ", f"{total_size:.1f}GB")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # エクスポート・インポート機能
    st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
    st.markdown("### 📤 データエクスポート")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📁 JSON エクスポート", use_container_width=True):
            export_experiments_json(experiments)
    
    with col2:
        if st.button("📊 CSV エクスポート", use_container_width=True):
            export_experiments_csv(experiments)
    
    with col3:
        if st.button("📋 レポート生成", use_container_width=True):
            generate_comprehensive_report(experiments)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # データクリーンアップ
    st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
    st.markdown("### 🧹 データクリーンアップ")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ 失敗実験を削除", use_container_width=True):
            cleanup_failed_experiments(app, experiments)
    
    with col2:
        if st.button("🔄 実験データ最適化", use_container_width=True):
            optimize_experiment_data(app, experiments)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 実験データ詳細
    if experiments:
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown("### 📊 実験データ詳細")
        
        # データフレーム表示
        df = create_experiments_dataframe(experiments)
        st.dataframe(df, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


def generate_comprehensive_report(experiments):
    """包括的レポート生成"""
    from datetime import datetime
    import json
    
    # レポートデータ作成
    report = {
        'generation_timestamp': datetime.now().isoformat(),
        'summary': {
            'total_experiments': len(experiments),
            'completed_experiments': len([e for e in experiments if e.get('status') == 'completed']),
            'failed_experiments': len([e for e in experiments if e.get('status') == 'failed']),
            'average_mt_bench_score': 0,
            'best_experiment': None,
            'worst_experiment': None
        },
        'detailed_analysis': {},
        'recommendations': []
    }
    
    # 完了実験の分析
    completed_experiments = [e for e in experiments if e.get('status') == 'completed']
    if completed_experiments:
        scores = []
        for exp in completed_experiments:
            score = exp.get('evaluations', {}).get('mt_bench_jp', {}).get('overall_score', 0)
            if score > 0:
                scores.append((score, exp))
        
        if scores:
            scores.sort(key=lambda x: x[0])
            avg_score = sum([s[0] for s in scores]) / len(scores)
            
            report['summary']['average_mt_bench_score'] = avg_score
            report['summary']['best_experiment'] = scores[-1][1]['id']
            report['summary']['worst_experiment'] = scores[0][1]['id']
            
            # 推奨事項
            if avg_score < 7.0:
                report['recommendations'].append("平均スコアが7.0を下回っています。より優秀なベースモデルの使用を検討してください。")
            if len(completed_experiments) < 5:
                report['recommendations'].append("実験数が少ないです。さまざまな組み合わせを試してパターンを見つけてください。")
    
    # レポートをJSON形式でダウンロード
    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    
    st.download_button(
        label="📋 包括的レポートをダウンロード",
        data=report_json,
        file_name=f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    st.success("包括的レポートが生成されました！")


def cleanup_failed_experiments(app, experiments):
    """失敗実験のクリーンアップ"""
    failed_experiments = [e for e in experiments if e.get('status') == 'failed']
    
    if not failed_experiments:
        st.info("削除する失敗実験がありません。")
        return
    
    st.warning(f"{len(failed_experiments)}個の失敗実験を削除しますか？")
    
    if st.button("⚠️ 削除を実行", type="primary"):
        # 成功実験のみを保持
        successful_experiments = [e for e in experiments if e.get('status') != 'failed']
        
        # データベースを更新
        db_path = app.experiments_dir / 'experiments_db.json'
        with open(db_path, 'w') as f:
            json.dump(successful_experiments, f, indent=2)
        
        st.success(f"{len(failed_experiments)}個の失敗実験を削除しました。")
        st.rerun()


def optimize_experiment_data(app, experiments):
    """実験データの最適化"""
    st.info("実験データを最適化中...")
    
    optimized_experiments = []
    for exp in experiments:
        # 不要なフィールドを削除
        optimized_exp = {
            'id': exp.get('id'),
            'merge_method': exp.get('merge_method'),
            'timestamp': exp.get('timestamp'),
            'status': exp.get('status'),
            'models': exp.get('models', []),
            'evaluations': exp.get('evaluations', {}),
            'quantization': exp.get('quantization', {}),
            'output_path': exp.get('output_path')
        }
        
        # 空の評価結果を削除
        if not optimized_exp['evaluations']:
            del optimized_exp['evaluations']
        if not optimized_exp['quantization']:
            del optimized_exp['quantization']
            
        optimized_experiments.append(optimized_exp)
    
    # 最適化されたデータを保存
    db_path = app.experiments_dir / 'experiments_db.json'
    with open(db_path, 'w') as f:
        json.dump(optimized_experiments, f, indent=2)
    
    st.success("実験データを最適化しました。")
    st.rerun()


if __name__ == "__main__":
    main()