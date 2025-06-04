# AI Fusion Studio クイックスタートガイド

## 🚀 5分で始めるAIモデル融合

### 1. セットアップ

```bash
# リポジトリをクローン
git clone <your-repo>
cd ai-fusion-studio

# 依存関係をインストール
make install
```

### 2. 最初の実験：Gemma × Qwen SLERP

最も期待値の高い組み合わせから始めましょう：

```bash
# Gemma-3-4B-IT × Qwen3-4B-Instruct のSLERPマージ
make experiment-gemma-qwen
```

このコマンドで以下が自動実行されます：
1. モデルのダウンロードとマージ
2. Japanese MT-Benchによる評価
3. AWQ量子化（4bit）
4. 結果レポートの生成

期待される結果：
- MT-Benchスコア: 8.5/10
- 量子化後サイズ: ~3.7GB

### 3. 結果の確認

```bash
# リーダーボードを表示
make leaderboard

# 可視化ダッシュボードを生成
make visualize
# → experiments/visualizations/experiment_dashboard.html を開く
```

## 📊 推奨実験パターン

### パターン1: チャット最適化（初心者向け）
**Gemma + Swallow LoRA** - 既存モデルをチャット用に最適化

```bash
python scripts/run_experiment.py configs/gemma_swallow_lora.yaml
```

- 実装が簡単（2行のコード）
- MT-Bench +0.3pt の改善
- ビジネスコミュニケーションに最適

### パターン2: 数理能力強化（中級者向け）
**EvoLLM-JP再現実験** - 多言語×数理の進化的マージ

```bash
python scripts/run_experiment.py configs/evolllm_jp_reproduction.yaml
```

- 数学タスクで+2ptの改善
- 進化的アルゴリズムで最適化
- 4.5GB（int4）に収まる

### パターン3: 長文対応（上級者向け）
**GLM-4 + Swallow** - 1Mトークン長文対応

```bash
python scripts/run_experiment.py configs/glm4_swallow_long_context.yaml
```

- 超長文処理が可能
- 技術文書・ドキュメント処理に最適
- ~6.2GB（量子化後）

## 🛠️ カスタム実験の作成

### 1. 設定ファイルを作成

```yaml
# configs/my_experiment.yaml
merge_method: slerp
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
    - math
```

### 2. 実験を実行

```bash
python scripts/run_experiment.py configs/my_experiment.yaml
```

### 3. 個別ステップの実行

```bash
# マージのみ
make merge CONFIG=configs/my_experiment.yaml

# 評価のみ
make evaluate MODEL=models/my_merged_model

# 量子化のみ
make quantize MODEL=models/my_merged_model METHOD=awq
```

## 💡 Tips & トラブルシューティング

### メモリ不足の場合

1. 量子化ビット数を下げる：
```yaml
quantization:
  bits: 2  # 4 → 2に変更
```

2. CPU実行に切り替える：
```bash
python scripts/merge_models.py --config configs/your_config.yaml --device cpu
```

### 実験の中断と再開

```bash
# 評価からやり直す場合
python scripts/run_experiment.py configs/your_config.yaml --skip merge
```

### バッチ実験

複数の設定を一度に実行：

```bash
# すべての推奨ペアを実験
make batch-experiment
```

## 📈 次のステップ

1. **結果の分析**: `experiments/visualizations/` のダッシュボードを確認
2. **パラメータ調整**: α値やgroup_sizeを変更して再実験
3. **新しい組み合わせ**: まだ試されていないモデルペアを探索

詳細は[実験ガイド](experiment_guide.md)を参照してください。