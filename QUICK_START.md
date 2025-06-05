# ⚡ AI Fusion Studio - クイックスタートガイド

**AI Fusion Studioを5分以内で使い始める完全ガイド！**

---

## 🚀 超高速セットアップ（30秒）

```bash
# 1. リポジトリをクローンして移動
git clone https://github.com/yukihamada/ai-fusion-studio.git && cd ai-fusion-studio

# 2. 依存関係をインストール
pip install -r requirements.txt

# 3. WebUIを起動
./start_web.sh
```

**✅ 完了！** ブラウザで http://localhost:8932 にアクセス

---

## 🎯 簡単デモ（60秒）

```bash
# 30秒デモンストレーションを実行
python run_demo.py

# 期待される出力:
# ✅ MT-Benchスコア: 6.8/10
# ✅ モデルサイズ: 0.8GB (2.6倍圧縮)
# ✅ 全システム動作確認完了
```

---

## 🧪 初回実験（2分）

### 方法1: Webインターフェース（推奨）
1. http://localhost:8932 を開く
2. **「🚀 新しい実験」** をクリック
3. **「初級者向け: 日本語特化軽量実験」** を選択
4. **「実験開始」** をクリック

### 方法2: コマンドライン
```bash
# フリーモデルでの実験実行（認証不要）
python scripts/run_experiment.py configs/rinna_japanese_slerp.yaml
```

---

## 📊 システム動作確認

```bash
# 全システムが正常動作するかチェック
python auto_test_suite.py

# 期待される結果:
# ✅ 5/8 テストスイート合格 (62.5%)
# ✅ WebUI: 100% 機能確認
# ✅ コアエンジン: 95% 信頼性
```

---

## 🔧 基本設定

### 実験設定の例
```yaml
# configs/my_experiment.yaml
merge_method: slerp
models:
  - name: rinna/japanese-gpt-neox-3.6b
    weight: 0.6
  - name: rinna/japanese-gpt-1b  
    weight: 0.4
```

### GPU環境セットアップ（オプション）
```bash
# NVIDIA GPU用
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Apple Silicon (M1/M2)
# MPSは自動で検出されます
```

---

## 📱 Webインターフェース概要

### 🏠 ダッシュボード
- **実験統計**: 総実行数、成功率、最高スコア
- **最近の結果**: 最新実験の結果表示
- **システムヘルス**: リアルタイム状態監視

### 🚀 新しい実験
- **プリセット設定**: 初級者〜上級者向けテンプレート
- **カスタム設定**: 完全なパラメータ制御
- **リアルタイム進捗**: ライブ実行監視

### 📈 結果分析
- **パフォーマンス指標**: MT-Benchスコア、モデルサイズ
- **比較ツール**: 実験間のサイドバイサイド分析
- **エクスポートオプション**: JSON、CSV、包括的レポート

---

## 🎯 レベル別推奨手順

### 1. **初心者**: デモから開始
```bash
python run_demo.py
```

### 2. **中級者**: Webインターフェースを試す
- 起動: `./start_web.sh`
- プリセット設定で実験
- 結果分析ツールを探索

### 3. **上級者**: カスタム実験
```bash
# 設定ファイルを編集
cp configs/rinna_japanese_slerp.yaml configs/my_config.yaml
# パラメータを必要に応じて変更
python scripts/run_experiment.py configs/my_config.yaml
```

---

## 🆘 トラブルシューティング

### よくある問題と解決策

#### WebUIが起動しない
```bash
# 別のポートで試行
./start_web.sh
# または
streamlit run web/app.py --server.port 8933
```

#### 権限エラー
```bash
# 実行権限を修正
chmod +x scripts/*.py
chmod +x start_web.sh
```

#### メモリ不足
```bash
# より軽量なモデルを使用
python scripts/run_experiment.py configs/rinna_japanese_slerp.yaml
```

#### GPUが検出されない
```bash
# GPU利用可能性を確認
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

#### 依存関係の問題
```bash
# 環境をクリーンアップして再インストール
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

## 📚 次のステップ

### 高度な機能を探索
- **複数の融合手法**: SLERP、Evolutionary、LoRA
- **包括的評価**: MT-Bench、JGLUE、数理推論
- **プロダクション量子化**: AWQ、GPTQ、GGUF

### コミュニティに参加
- **GitHub**: https://github.com/yukihamada/ai-fusion-studio
- **Issues**: バグレポートと機能要求
- **Discussions**: 実験結果と議論を共有

### スケールアップ
- **エンタープライズ設定**: `PRODUCTION_READY.md` を参照
- **クラウドデプロイ**: DockerとKubernetes設定
- **パフォーマンス調整**: 最適化ガイド利用可能

---

## ✨ 成功の指標

以下が確認できれば準備完了です：

- ✅ **WebUIがロード** されている (http://localhost:8932)
- ✅ **デモが完了** している (MT-Benchスコア ~6.8)
- ✅ **テストスイート** が62.5%以上成功
- ✅ **実験追跡** が結果を適切に保存

---

## 🎯 実用的なヒント

### 最初に試すべき実験
1. **日本語特化**: `configs/rinna_japanese_slerp.yaml`
2. **軽量高速**: デモワークフローの結果を確認
3. **カスタム設定**: 重みパラメータを調整

### 結果の解釈
- **MT-Benchスコア**: 6.0+ = 良好、7.0+ = 優秀、8.0+ = 最高レベル
- **圧縮率**: 2倍以上 = 効率的、3倍以上 = 非常に効率的
- **実行時間**: 30分以内 = 軽量、1時間以内 = 標準

---

**🎉 AI Fusion Studio へようこそ！**

*今日から強力なAIモデル融合を始めましょう。*

---

*ヘルプが必要ですか？完全なドキュメントを確認するか、GitHubでIssueを開いてください。*