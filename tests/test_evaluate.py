#!/usr/bin/env python3
"""
評価機能のテスト
"""

import pytest
import tempfile
import json
import os
import shutil
import unittest.mock as mock
from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from scripts.evaluate import JapaneseLLMEvaluator, main


class MockTokenizer:
    """テスト用のモックトークナイザー"""
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.pad_token_id = 0
    
    def __call__(self, text, **kwargs):
        # 簡易的なトークン化
        tokens = text.split()
        return {
            'input_ids': torch.tensor([list(range(len(tokens)))]),
            'attention_mask': torch.tensor([1] * len(tokens))
        }
    
    def decode(self, tokens, **kwargs):
        # 簡易的なデコード
        return "モックレスポンス: これはテスト用の応答です。"
    
    @classmethod
    def from_pretrained(cls, model_path):
        return cls()


class MockModel:
    """テスト用のモックモデル"""
    def __init__(self):
        pass
    
    def generate(self, **kwargs):
        # ダミーの出力
        return [torch.tensor([0, 1, 2, 3, 4])]
    
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        return cls()


class TestJapaneseLLMEvaluator:
    """JapaneseLLMEvaluatorクラスのテスト"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """テスト用の一時モデルディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        # 必要なファイルを作成
        Path(temp_dir, 'config.json').write_text('{"model_type": "test"}')
        Path(temp_dir, 'tokenizer.json').write_text('{}') 
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_evaluator(self, temp_model_dir):
        """モックされた評価器"""
        with mock.patch('scripts.evaluate.AutoModelForCausalLM', MockModel), \
             mock.patch('scripts.evaluate.AutoTokenizer', MockTokenizer), \
             mock.patch('scripts.evaluate.torch.cuda.is_available', return_value=False):
            evaluator = JapaneseLLMEvaluator(temp_model_dir, 'cpu')
            yield evaluator
    
    def test_init_success(self, temp_model_dir):
        """正常な初期化テスト"""
        with mock.patch('scripts.evaluate.AutoModelForCausalLM', MockModel), \
             mock.patch('scripts.evaluate.AutoTokenizer', MockTokenizer), \
             mock.patch('scripts.evaluate.torch.cuda.is_available', return_value=False):
            evaluator = JapaneseLLMEvaluator(temp_model_dir, 'cpu')
            assert evaluator.model_path == Path(temp_model_dir)
            assert evaluator.device.type == 'cpu'
            assert evaluator.results == {}
    
    def test_init_with_cuda(self, temp_model_dir):
        """CUDA利用可能時の初期化テスト"""
        with mock.patch('scripts.evaluate.AutoModelForCausalLM', MockModel), \
             mock.patch('scripts.evaluate.AutoTokenizer', MockTokenizer), \
             mock.patch('scripts.evaluate.torch.cuda.is_available', return_value=True):
            evaluator = JapaneseLLMEvaluator(temp_model_dir, 'cuda')
            assert str(evaluator.device) == 'cuda'
    
    def test_init_failure(self):
        """初期化失敗テスト"""
        with mock.patch('scripts.evaluate.AutoModelForCausalLM.from_pretrained', side_effect=Exception("Model load error")):
            with pytest.raises(RuntimeError, match="Failed to load model"):
                JapaneseLLMEvaluator("/nonexistent/path")
    
    def test_tokenizer_pad_token_setup(self, temp_model_dir):
        """トークナイザーのpad_token設定テスト"""
        class MockTokenizerNoPad:
            def __init__(self):
                self.pad_token = None
                self.eos_token = "</s>"
                self.pad_token_id = 0
            
            def __call__(self, text, **kwargs):
                return {'input_ids': torch.tensor([[1, 2, 3]])}
            
            def decode(self, tokens, **kwargs):
                return "test"
            
            @classmethod
            def from_pretrained(cls, model_path):
                return cls()
        
        with mock.patch('scripts.evaluate.AutoModelForCausalLM', MockModel), \
             mock.patch('scripts.evaluate.AutoTokenizer', MockTokenizerNoPad):
            evaluator = JapaneseLLMEvaluator(temp_model_dir)
            assert evaluator.tokenizer.pad_token == evaluator.tokenizer.eos_token
    
    def test_generate_response_success(self, mock_evaluator):
        """応答生成成功テスト"""
        response = mock_evaluator._generate_response("テストプロンプト")
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_generate_response_with_max_length(self, mock_evaluator):
        """最大長指定での応答生成テスト"""
        response = mock_evaluator._generate_response("テスト", max_length=100)
        assert isinstance(response, str)
    
    def test_generate_response_error(self, mock_evaluator):
        """応答生成エラーテスト"""
        with mock.patch.object(mock_evaluator.model, 'generate', side_effect=Exception("Generation error")):
            response = mock_evaluator._generate_response("テスト")
            assert response == ""
    
    def test_score_response_basic(self):
        """基本的なレスポンススコアリングテスト"""
        evaluator = JapaneseLLMEvaluator.__new__(JapaneseLLMEvaluator)
        
        # 良い応答
        good_response = "こんにちは。これは適切な長さの日本語の応答です。論理的で自然な文章になっています。"
        score = evaluator._score_response("テスト質問", good_response, "general")
        assert 3.0 <= score <= 10.0
        
        # 短すぎる応答
        short_response = "短い"
        score = evaluator._score_response("テスト質問", short_response, "general")
        assert score <= 5.0
        
        # 長すぎる応答
        long_response = "非常に長い応答" * 100
        score = evaluator._score_response("テスト質問", long_response, "general")
        assert score <= 6.0
        
        # 空の応答
        empty_response = ""
        score = evaluator._score_response("テスト質問", empty_response, "general")
        assert score == 3.0
    
    def test_score_response_coding(self):
        """コーディングタスクのスコアリングテスト"""
        evaluator = JapaneseLLMEvaluator.__new__(JapaneseLLMEvaluator)
        
        # コードブロックを含む応答
        code_response = """
        以下のPythonコードで実装できます：
        ```python
        def hello():
            print("Hello World")
        ```
        これで問題が解決します。
        """
        score = evaluator._score_response("プログラムを書いてください", code_response, "coding")
        assert score >= 6.0
        
        # def キーワードを含む応答
        def_response = "def function_name(): pass という形で関数を定義します。"
        score = evaluator._score_response("関数の書き方は？", def_response, "coding")
        assert score >= 6.0
        
        # function キーワードを含む応答
        func_response = "JavaScriptでfunction宣言を使用します。"
        score = evaluator._score_response("関数について", func_response, "coding")
        assert score >= 6.0
        
        # コードを含まない応答
        no_code_response = "プログラムについて説明します。"
        score = evaluator._score_response("プログラムを書いてください", no_code_response, "coding")
        assert score <= 6.0
    
    def test_score_response_math(self):
        """数学タスクのスコアリングテスト"""
        evaluator = JapaneseLLMEvaluator.__new__(JapaneseLLMEvaluator)
        
        # 数字を含む応答
        math_response = "答えは42です。計算過程は3.14 * 2 = 6.28となります。"
        score = evaluator._score_response("計算してください", math_response, "math")
        assert score >= 6.0
        
        # 数字を含まない応答
        no_math_response = "数学について説明します。"
        score = evaluator._score_response("計算してください", no_math_response, "math")
        assert score <= 6.0
    
    def test_score_response_reasoning(self):
        """推論タスクのスコアリングテスト"""
        evaluator = JapaneseLLMEvaluator.__new__(JapaneseLLMEvaluator)
        
        # 推論キーワードを含む応答
        reasoning_response = "まず前提を整理します。したがって、結論は〜です。"
        score = evaluator._score_response("推論してください", reasoning_response, "reasoning")
        assert score >= 6.0
        
        # よってキーワードを含む応答
        yotte_response = "条件から考えると、よって答えは〜です。"
        score = evaluator._score_response("推論してください", yotte_response, "reasoning")
        assert score >= 6.0
        
        # 推論キーワードを含まない応答
        no_reasoning_response = "これは簡単な問題です。"
        score = evaluator._score_response("推論してください", no_reasoning_response, "reasoning")
        assert score <= 6.0
    
    def test_score_response_japanese_naturalness(self):
        """日本語の自然さテスト"""
        evaluator = JapaneseLLMEvaluator.__new__(JapaneseLLMEvaluator)
        
        # 自然な日本語（句点あり）
        natural_response = "これは自然な日本語です。文章として完成しています。"
        score = evaluator._score_response("質問", natural_response, "general")
        base_score = evaluator._score_response("質問", natural_response.replace("。", ""), "general")
        assert score > base_score
    
    def test_score_response_bounds(self):
        """スコア境界値テスト"""
        evaluator = JapaneseLLMEvaluator.__new__(JapaneseLLMEvaluator)
        
        # 最低スコア確認
        worst_response = ""
        score = evaluator._score_response("質問", worst_response, "general")
        assert score >= 0.0
        
        # 最高スコア確認  
        best_response = "これは非常に優れた応答です。" * 10  # 適度な長さ
        score = evaluator._score_response("質問", best_response, "general")
        assert score <= 10.0
    
    def test_check_math_answer(self):
        """数学の答えチェックテスト"""
        evaluator = JapaneseLLMEvaluator.__new__(JapaneseLLMEvaluator)
        
        # 正しい答え - 直接一致
        assert evaluator._check_math_answer("答えは42です", "42")
        assert evaluator._check_math_answer("結果: 3.14", "3.14")
        assert evaluator._check_math_answer("分数は1/2です", "1/2")
        
        # 正しい答え - 数値リストから検出
        assert evaluator._check_math_answer("計算すると42.5になります", "42.5")
        assert evaluator._check_math_answer("値は -5 です", "-5")
        
        # 正しい答え - 分数リストから検出
        assert evaluator._check_math_answer("答えは3/4です", "3/4")
        
        # 間違った答え
        assert not evaluator._check_math_answer("答えは43です", "42")
        assert not evaluator._check_math_answer("わかりません", "42")
        assert not evaluator._check_math_answer("計算できません", "3.14")
        
        # エッジケース
        assert not evaluator._check_math_answer("", "42")
        assert not evaluator._check_math_answer("数字がありません", "42")
    
    def test_evaluate_mt_bench_jp(self, mock_evaluator):
        """MT-Bench-JP評価テスト"""
        result = mock_evaluator.evaluate_mt_bench_jp()
        
        # 結果構造の確認
        assert 'overall_score' in result
        assert 'category_scores' in result
        assert 'detailed_results' in result
        
        # スコア範囲の確認
        assert 0 <= result['overall_score'] <= 10
        
        # カテゴリ別スコアの確認
        expected_categories = ['writing', 'roleplay', 'reasoning', 'math', 'coding', 'extraction', 'stem', 'humanities']
        for category in expected_categories:
            assert category in result['category_scores']
            assert 0 <= result['category_scores'][category] <= 10
        
        # 詳細結果の確認
        for category in expected_categories:
            assert category in result['detailed_results']
            assert len(result['detailed_results'][category]) == 2  # 各カテゴリ2問
            
            for detail in result['detailed_results'][category]:
                assert 'question' in detail
                assert 'response' in detail
                assert 'score' in detail
                assert 0 <= detail['score'] <= 10
        
        # 結果がインスタンス変数に保存されているか確認
        assert 'mt_bench_jp' in mock_evaluator.results
    
    def test_evaluate_jglue(self, mock_evaluator):
        """JGLUE評価テスト"""
        result = mock_evaluator.evaluate_jglue()
        
        # 結果構造の確認
        expected_tasks = ['jcommonsenseqa', 'jnli', 'marc_ja', 'jsquad']
        for task in expected_tasks:
            assert task in result
        
        # 結果がインスタンス変数に保存されているか確認
        assert 'jglue' in mock_evaluator.results
    
    def test_evaluate_jglue_with_error(self, mock_evaluator):
        """JGLUEエラー処理テスト"""
        # jcommonsenseqaでエラーを発生させる
        original_func = mock_evaluator._evaluate_jcommonsenseqa
        mock_evaluator._evaluate_jcommonsenseqa = lambda: (_ for _ in ()).throw(Exception("Test error"))
        
        result = mock_evaluator.evaluate_jglue()
        
        # エラーが適切に処理されているか確認
        assert 'jcommonsenseqa' in result
        assert 'error' in result['jcommonsenseqa']
        assert result['jcommonsenseqa']['error'] == "Test error"
        
        # 他のタスクは正常に実行されているか確認
        assert 'jnli' in result
        assert 'error' not in result['jnli']
        
        # 復元
        mock_evaluator._evaluate_jcommonsenseqa = original_func
    
    def test_evaluate_mathematical_reasoning(self, mock_evaluator):
        """数理推論評価テスト"""
        result = mock_evaluator.evaluate_mathematical_reasoning()
        
        # 結果構造の確認
        assert 'accuracy' in result
        assert 'detailed_results' in result
        
        # 精度範囲の確認
        assert 0 <= result['accuracy'] <= 1
        
        # 詳細結果の確認
        assert len(result['detailed_results']) == 4  # 4つの数学問題
        
        for detail in result['detailed_results']:
            assert 'question' in detail
            assert 'expected' in detail
            assert 'response' in detail
            assert 'correct' in detail
            assert 'type' in detail
            assert isinstance(detail['correct'], bool)
            assert detail['type'] in ['algebra', 'arithmetic', 'logarithm', 'probability']
        
        # 結果がインスタンス変数に保存されているか確認
        assert 'mathematical_reasoning' in mock_evaluator.results
    
    def test_evaluate_jcommonsenseqa(self):
        """JCommonsenseQA評価テスト"""
        evaluator = JapaneseLLMEvaluator.__new__(JapaneseLLMEvaluator)
        
        # モック応答関数
        def mock_generate_response(prompt):
            if "屋根の下で待つ" in prompt:
                return "屋根の下で待つのが良いでしょう。"
            else:
                return "わかりません。"
        
        evaluator._generate_response = mock_generate_response
        
        result = evaluator._evaluate_jcommonsenseqa()
        
        assert 'accuracy' in result
        assert 0 <= result['accuracy'] <= 1
    
    def test_evaluate_jnli(self):
        """JNLI評価テスト"""
        evaluator = JapaneseLLMEvaluator.__new__(JapaneseLLMEvaluator)
        result = evaluator._evaluate_jnli()
        
        assert 'accuracy' in result
        assert result['accuracy'] == 0.75  # ダミー値
    
    def test_evaluate_marc_ja(self):
        """MARC-JA評価テスト"""
        evaluator = JapaneseLLMEvaluator.__new__(JapaneseLLMEvaluator)
        result = evaluator._evaluate_marc_ja()
        
        assert 'accuracy' in result
        assert result['accuracy'] == 0.82  # ダミー値
    
    def test_evaluate_jsquad(self):
        """JSQuAD評価テスト"""
        evaluator = JapaneseLLMEvaluator.__new__(JapaneseLLMEvaluator)
        result = evaluator._evaluate_jsquad()
        
        assert 'f1_score' in result
        assert result['f1_score'] == 0.78  # ダミー値
    
    @pytest.fixture
    def temp_output_dir(self):
        """テスト用出力ディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_generate_report(self, mock_evaluator, temp_output_dir):
        """レポート生成テスト"""
        # 評価データを準備
        mock_evaluator.results = {
            'mt_bench_jp': {
                'overall_score': 7.5,
                'category_scores': {
                    'writing': 8.0,
                    'reasoning': 7.0,
                    'math': 6.5
                },
                'detailed_results': {
                    'writing': [{'question': 'test', 'response': 'test', 'score': 8.0}]
                }
            },
            'mathematical_reasoning': {
                'accuracy': 0.75,
                'detailed_results': [
                    {'question': 'test', 'expected': '4', 'response': '4', 'correct': True, 'type': 'algebra'}
                ]
            }
        }
        
        # レポート生成実行
        with mock.patch('matplotlib.pyplot.savefig'), \
             mock.patch('matplotlib.pyplot.close'), \
             mock.patch('matplotlib.pyplot.figure'), \
             mock.patch('matplotlib.pyplot.bar'), \
             mock.patch('matplotlib.pyplot.ylim'), \
             mock.patch('matplotlib.pyplot.xlabel'), \
             mock.patch('matplotlib.pyplot.ylabel'), \
             mock.patch('matplotlib.pyplot.title'), \
             mock.patch('matplotlib.pyplot.xticks'), \
             mock.patch('matplotlib.pyplot.tight_layout'):
            
            mock_evaluator.generate_report(temp_output_dir)
        
        # 生成されたファイルの確認
        output_path = Path(temp_output_dir)
        assert (output_path / 'evaluation_results.json').exists()
        
        # JSONファイルの内容確認
        with open(output_path / 'evaluation_results.json', 'r') as f:
            saved_results = json.load(f)
            assert 'mt_bench_jp' in saved_results
            assert 'mathematical_reasoning' in saved_results
    
    def test_generate_mt_bench_report(self, mock_evaluator, temp_output_dir):
        """MT-Benchレポート生成テスト"""
        mock_evaluator.results = {
            'mt_bench_jp': {
                'overall_score': 7.5,
                'category_scores': {
                    'writing': 8.0,
                    'reasoning': 7.0,
                    'math': 6.5,
                    'coding': 5.0,
                    'roleplay': 4.0
                }
            }
        }
        
        output_path = Path(temp_output_dir)
        
        with mock.patch('matplotlib.pyplot.savefig'), \
             mock.patch('matplotlib.pyplot.close'), \
             mock.patch('matplotlib.pyplot.figure'), \
             mock.patch('matplotlib.pyplot.bar') as mock_bar, \
             mock.patch('matplotlib.pyplot.ylim'), \
             mock.patch('matplotlib.pyplot.xlabel'), \
             mock.patch('matplotlib.pyplot.ylabel'), \
             mock.patch('matplotlib.pyplot.title'), \
             mock.patch('matplotlib.pyplot.xticks'), \
             mock.patch('matplotlib.pyplot.tight_layout'):
            
            # バーオブジェクトのモック
            mock_bars = [mock.Mock() for _ in range(5)]
            mock_bar.return_value = mock_bars
            
            mock_evaluator._generate_mt_bench_report(output_path)
        
        # テキストレポートの確認
        assert (output_path / 'mt_bench_report.txt').exists()
        
        with open(output_path / 'mt_bench_report.txt', 'r') as f:
            content = f.read()
            assert '総合スコア: 7.50/10' in content
            assert 'writing: 8.00/10' in content
            assert 'reasoning: 7.00/10' in content
    
    def test_generate_math_report(self, mock_evaluator, temp_output_dir):
        """数理推論レポート生成テスト"""
        mock_evaluator.results = {
            'mathematical_reasoning': {
                'accuracy': 0.75,
                'detailed_results': [
                    {'type': 'algebra', 'correct': True},
                    {'type': 'algebra', 'correct': False},
                    {'type': 'arithmetic', 'correct': True},
                    {'type': 'logarithm', 'correct': True}
                ]
            }
        }
        
        output_path = Path(temp_output_dir)
        
        with mock.patch('matplotlib.pyplot.savefig'), \
             mock.patch('matplotlib.pyplot.close'), \
             mock.patch('matplotlib.pyplot.figure'), \
             mock.patch('matplotlib.pyplot.bar'), \
             mock.patch('matplotlib.pyplot.ylim'), \
             mock.patch('matplotlib.pyplot.xlabel'), \
             mock.patch('matplotlib.pyplot.ylabel'), \
             mock.patch('matplotlib.pyplot.title'), \
             mock.patch('matplotlib.pyplot.tight_layout'):
            
            mock_evaluator._generate_math_report(output_path)


class TestMainFunction:
    """main関数のテスト"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """テスト用の一時モデルディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        Path(temp_dir, 'config.json').write_text('{"model_type": "test"}')
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def temp_output_dir(self):
        """テスト用出力ディレクトリ"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_main_success(self, temp_model_dir, temp_output_dir):
        """main関数正常実行テスト"""
        test_args = [
            'evaluate.py',
            '--model-path', temp_model_dir,
            '--benchmarks', 'mt-bench-jp', 'jglue', 'math',
            '--output-dir', temp_output_dir,
            '--device', 'cpu'
        ]
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.evaluate.JapaneseLLMEvaluator') as mock_eval_class, \
             mock.patch('scripts.evaluate.Path.exists', return_value=True):
            
            # モック評価器の設定
            mock_evaluator = mock.Mock()
            mock_eval_class.return_value = mock_evaluator
            
            # main実行
            main()
            
            # 評価器が適切に作成されたか確認
            mock_eval_class.assert_called_once_with(temp_model_dir, 'cpu')
            
            # 各ベンチマークが実行されたか確認
            mock_evaluator.evaluate_mt_bench_jp.assert_called_once()
            mock_evaluator.evaluate_jglue.assert_called_once()
            mock_evaluator.evaluate_mathematical_reasoning.assert_called_once()
            
            # レポート生成が実行されたか確認
            mock_evaluator.generate_report.assert_called_once()
    
    def test_main_model_not_found(self, temp_output_dir):
        """存在しないモデルパス指定テスト"""
        test_args = [
            'evaluate.py',
            '--model-path', '/nonexistent/model',
            '--output-dir', temp_output_dir
        ]
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.evaluate.Path.exists', return_value=False), \
             pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1
    
    def test_main_default_arguments(self, temp_model_dir):
        """デフォルト引数でのmain実行テスト"""
        test_args = [
            'evaluate.py',
            '--model-path', temp_model_dir
        ]
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.evaluate.JapaneseLLMEvaluator') as mock_eval_class, \
             mock.patch('scripts.evaluate.Path.exists', return_value=True):
            
            mock_evaluator = mock.Mock()
            mock_eval_class.return_value = mock_evaluator
            
            main()
            
            # デフォルト値での作成確認
            mock_eval_class.assert_called_once_with(temp_model_dir, 'cuda')
            
            # デフォルトベンチマークの実行確認
            mock_evaluator.evaluate_mt_bench_jp.assert_called_once()
            mock_evaluator.evaluate_jglue.assert_called_once()
            mock_evaluator.evaluate_mathematical_reasoning.assert_called_once()
    
    def test_main_partial_benchmarks(self, temp_model_dir):
        """部分的なベンチマーク実行テスト"""
        test_args = [
            'evaluate.py',
            '--model-path', temp_model_dir,
            '--benchmarks', 'mt-bench-jp'
        ]
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.evaluate.JapaneseLLMEvaluator') as mock_eval_class, \
             mock.patch('scripts.evaluate.Path.exists', return_value=True):
            
            mock_evaluator = mock.Mock()
            mock_eval_class.return_value = mock_evaluator
            
            main()
            
            # 指定されたベンチマークのみ実行
            mock_evaluator.evaluate_mt_bench_jp.assert_called_once()
            mock_evaluator.evaluate_jglue.assert_not_called()
            mock_evaluator.evaluate_mathematical_reasoning.assert_not_called()
    
    def test_main_benchmark_error(self, temp_model_dir):
        """ベンチマーク実行エラーテスト"""
        test_args = [
            'evaluate.py',
            '--model-path', temp_model_dir,
            '--benchmarks', 'mt-bench-jp'
        ]
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.evaluate.JapaneseLLMEvaluator') as mock_eval_class, \
             mock.patch('scripts.evaluate.Path.exists', return_value=True), \
             mock.patch('scripts.evaluate.logger') as mock_logger:
            
            mock_evaluator = mock.Mock()
            mock_evaluator.evaluate_mt_bench_jp.side_effect = Exception("Benchmark error")
            mock_eval_class.return_value = mock_evaluator
            
            main()
            
            # エラーがログに記録されているか確認
            mock_logger.error.assert_called()
    
    def test_main_unexpected_error(self, temp_model_dir):
        """予期しないエラーテスト"""
        test_args = [
            'evaluate.py',
            '--model-path', temp_model_dir
        ]
        
        with mock.patch('sys.argv', test_args), \
             mock.patch('scripts.evaluate.JapaneseLLMEvaluator', side_effect=Exception("Unexpected error")), \
             mock.patch('scripts.evaluate.Path.exists', return_value=True), \
             pytest.raises(SystemExit) as exc_info:
            main()
        
        assert exc_info.value.code == 1
    
    def test_mathematical_reasoning_basic(self):
        """数理推論の基本テスト（モックデータ使用）"""
        evaluator = JapaneseLLMEvaluator.__new__(JapaneseLLMEvaluator)
        
        # モック応答関数
        def mock_generate_response(prompt):
            if "太郎" in prompt and "花子" in prompt:
                return "年齢差は1歳です"
            elif "2次方程式" in prompt:
                return "x = 2, x = 3"
            elif "半径5cm" in prompt:
                return "面積は約78.5平方cm、周長は約31.4cmです"
            else:
                return "4"
        
        evaluator._generate_response = mock_generate_response
        
        # 一部の問題をテスト
        test_problems = [
            {
                'question': "太郎は花子より3歳年上で、花子は次郎より2歳年下です。太郎と次郎の年齢差は何歳ですか？",
                'answer': "1",
                'type': 'algebra'
            },
            {
                'question': "ある数に3を足して2倍すると14になります。この数はいくつですか？",
                'answer': "4", 
                'type': 'algebra'
            }
        ]
        
        correct = 0
        for problem in test_problems:
            response = evaluator._generate_response(problem['question'])
            if evaluator._check_math_answer(response, problem['answer']):
                correct += 1
        
        accuracy = correct / len(test_problems)
        assert 0.0 <= accuracy <= 1.0