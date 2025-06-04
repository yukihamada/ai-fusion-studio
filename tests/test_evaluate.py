#!/usr/bin/env python3
"""
評価機能のテスト
"""

import pytest
import tempfile
import json
from pathlib import Path
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))

from scripts.evaluate import JapaneseLLMEvaluator


class TestJapaneseLLMEvaluator:
    """JapaneseLLMEvaluatorクラスのテスト"""
    
    def test_score_response_basic(self):
        """基本的なレスポンススコアリングテスト"""
        evaluator = JapaneseLLMEvaluator.__new__(JapaneseLLMEvaluator)  # インスタンス作成（初期化なし）
        
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
    
    def test_score_response_coding(self):
        """コーディングタスクのスコアリングテスト"""
        evaluator = JapaneseLLMEvaluator.__new__(JapaneseLLMEvaluator)
        
        # コードを含む応答
        code_response = """
        以下のPythonコードで実装できます：
        ```python
        def hello():
            print("Hello World")
        ```
        """
        score = evaluator._score_response("プログラムを書いてください", code_response, "coding")
        assert score >= 6.0  # コードがあるので高スコア
        
        # コードを含まない応答
        no_code_response = "プログラムについて説明します。"
        score = evaluator._score_response("プログラムを書いてください", no_code_response, "coding")
        assert score <= 6.0
    
    def test_check_math_answer(self):
        """数学の答えチェックテスト"""
        evaluator = JapaneseLLMEvaluator.__new__(JapaneseLLMEvaluator)
        
        # 正しい答え
        assert evaluator._check_math_answer("答えは42です", "42")
        assert evaluator._check_math_answer("結果: 3.14", "3.14")
        assert evaluator._check_math_answer("分数は1/2です", "1/2")
        
        # 間違った答え
        assert not evaluator._check_math_answer("答えは43です", "42")
        assert not evaluator._check_math_answer("わかりません", "42")
    
    def test_mathematical_reasoning_basic(self):
        """数理推論の基本テスト（モックデータ使用）"""
        # 実際のモデルを使わないモック評価
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