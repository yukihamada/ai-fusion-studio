# AI Fusion Studio Makefile

.PHONY: help install test merge evaluate quantize experiment clean

help:
	@echo "AI Fusion Studio - 利用可能なコマンド:"
	@echo "  make install       - 依存関係をインストール"
	@echo "  make test          - テストを実行"
	@echo "  make merge         - モデルマージを実行"
	@echo "  make evaluate      - モデル評価を実行"
	@echo "  make quantize      - モデル量子化を実行"
	@echo "  make experiment    - 完全な実験を実行"
	@echo "  make clean         - 一時ファイルを削除"

install:
	pip install -r requirements.txt
	@echo "依存関係のインストールが完了しました"

# 推奨モデルペアの実験
experiment-gemma-qwen:
	python scripts/run_experiment.py configs/gemma_qwen_slerp.yaml

experiment-evolllm:
	python scripts/run_experiment.py configs/evolllm_jp_reproduction.yaml

experiment-gemma-swallow:
	python scripts/run_experiment.py configs/gemma_swallow_lora.yaml

# 個別ステップの実行
merge:
	@if [ -z "$(CONFIG)" ]; then \
		echo "使用法: make merge CONFIG=configs/your_config.yaml"; \
		exit 1; \
	fi
	python scripts/merge_models.py --config $(CONFIG)

evaluate:
	@if [ -z "$(MODEL)" ]; then \
		echo "使用法: make evaluate MODEL=path/to/model"; \
		exit 1; \
	fi
	python scripts/evaluate.py --model-path $(MODEL) --benchmarks mt-bench-jp jglue math

quantize:
	@if [ -z "$(MODEL)" ]; then \
		echo "使用法: make quantize MODEL=path/to/model METHOD=awq"; \
		exit 1; \
	fi
	python scripts/quantize.py --model-path $(MODEL) --method $(METHOD) --bits 4

# 実験結果の可視化
visualize:
	python scripts/experiment_tracker.py --action visualize
	@echo "可視化結果: experiments/visualizations/"

# リーダーボード表示
leaderboard:
	python scripts/experiment_tracker.py --action leaderboard

# バッチ実験（すべての推奨ペア）
batch-experiment:
	@echo "すべての推奨モデルペアで実験を実行..."
	@for config in configs/*.yaml; do \
		echo "実行中: $$config"; \
		python scripts/run_experiment.py $$config || true; \
	done
	@echo "バッチ実験完了"

# テスト実行
test:
	@echo "簡易動作確認を実行..."
	python -c "import torch; print('PyTorch:', torch.__version__)"
	python -c "import transformers; print('Transformers:', transformers.__version__)"
	@echo "基本的な依存関係の確認完了"

# クリーンアップ
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf experiments/temp_*
	@echo "一時ファイルを削除しました"

clean-all: clean
	rm -rf models/merged_*
	rm -rf models/quantized/*
	rm -rf evaluations/*
	rm -rf experiments/*
	@echo "すべての生成ファイルを削除しました"