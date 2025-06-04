#!/usr/bin/env python3
"""
LLM Merge Lab - 対話テストスクリプト
デモ用軽量モデルを使用してLLMの日本語・英語会話能力をテストする
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BloomForCausalLM,
    BloomTokenizerFast,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    pipeline
)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chat_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ChatTestSuite:
    """LLM対話テストスイート"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small", device: str = "auto"):
        """
        Args:
            model_name: 使用するモデル名
            device: 使用デバイス (auto, cpu, cuda)
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        self.chat_pipeline = None
        
        # テスト項目定義
        self.test_cases = self._setup_test_cases()
        
        # 結果保存
        self.results = {
            'model_info': {
                'name': model_name,
                'device': str(self.device)
            },
            'test_results': [],
            'summary': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def _setup_device(self, device: str) -> torch.device:
        """デバイス設定"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("CUDA利用可能 - GPUを使用")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                logger.info("MPS利用可能 - Apple Siliconを使用")
            else:
                device = "cpu"
                logger.info("CPU使用")
        
        return torch.device(device)
    
    def _setup_test_cases(self) -> List[Dict]:
        """テストケース定義"""
        return [
            # 基本挨拶
            {
                'id': 'greeting_jp',
                'language': '日本語',
                'category': '挨拶',
                'input': 'こんにちは、調子はどうですか？',
                'expected_keywords': ['こんにちは', 'です', 'ます', '元気', '良い', 'ありがとう']
            },
            {
                'id': 'greeting_en',
                'language': 'English',
                'category': 'Greeting',
                'input': 'Hello, how are you doing today?',
                'expected_keywords': ['hello', 'good', 'fine', 'thank', 'great', 'nice']
            },
            
            # 時間・日付
            {
                'id': 'date_time_jp',
                'language': '日本語',
                'category': '時間',
                'input': '今日は何曜日ですか？',
                'expected_keywords': ['曜日', '今日', 'です', 'ます', '月', '火', '水', '木', '金', '土', '日']
            },
            {
                'id': 'date_time_en',
                'language': 'English',
                'category': 'Time',
                'input': 'What day is today?',
                'expected_keywords': ['today', 'day', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
            },
            
            # 自己紹介
            {
                'id': 'self_intro_jp',
                'language': '日本語',
                'category': '自己紹介',
                'input': 'あなたについて教えてください',
                'expected_keywords': ['AI', 'アシスタント', 'です', 'ます', '私', '助け', 'サポート']
            },
            {
                'id': 'self_intro_en',
                'language': 'English',
                'category': 'Self Introduction',
                'input': 'Tell me about yourself',
                'expected_keywords': ['AI', 'assistant', 'help', 'support', 'language', 'model']
            },
            
            # 基本的な質問
            {
                'id': 'favorite_color_jp',
                'language': '日本語',
                'category': '質問応答',
                'input': '好きな色は何ですか？',
                'expected_keywords': ['色', '好き', 'です', 'ます', '青', '赤', '緑', '黄', '白', '黒']
            },
            {
                'id': 'weather_en',
                'language': 'English',
                'category': 'Question Answering',
                'input': 'How is the weather today?',
                'expected_keywords': ['weather', 'today', 'sunny', 'cloudy', 'rainy', 'nice', 'good']
            },
            
            # 簡単な数学
            {
                'id': 'simple_math_jp',
                'language': '日本語',
                'category': '数学',
                'input': '2 + 3 はいくつですか？',
                'expected_keywords': ['5', '五', 'です', 'ます', '足し算', '計算']
            },
            {
                'id': 'simple_math_en',
                'language': 'English',
                'category': 'Math',
                'input': 'What is 2 + 3?',
                'expected_keywords': ['5', 'five', 'equals', 'answer', 'result']
            }
        ]
    
    def load_model(self) -> None:
        """モデルとトークナイザーをロード"""
        try:
            logger.info(f"モデルをロード中: {self.model_name}")
            
            # モデル固有の処理
            if "DialoGPT" in self.model_name:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                
                # パディングトークンを設定
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
            elif "distilgpt2" in self.model_name.lower():
                self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
                self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
                
                # パディングトークンを設定
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
            elif "bloom" in self.model_name.lower():
                self.tokenizer = BloomTokenizerFast.from_pretrained(self.model_name)
                self.model = BloomForCausalLM.from_pretrained(self.model_name)
                
            else:
                # 汎用的なローダー
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # モデルをデバイスに移動
            self.model.to(self.device)
            
            # パイプライン作成
            self.chat_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("モデルロード完了")
            
        except Exception as e:
            logger.error(f"モデルロードエラー: {e}")
            raise
    
    def generate_response(self, input_text: str, max_length: int = 100) -> str:
        """応答生成"""
        try:
            # 入力をトークン化
            inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            
            # 生成パラメータ
            generation_kwargs = {
                'max_length': min(inputs.shape[1] + max_length, 512),
                'num_return_sequences': 1,
                'temperature': 0.7,
                'do_sample': True,
                'pad_token_id': self.tokenizer.eos_token_id,
                'no_repeat_ngram_size': 3
            }
            
            # 応答生成
            with torch.no_grad():
                outputs = self.model.generate(inputs, **generation_kwargs)
            
            # デコード
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 入力部分を除去
            response = response[len(input_text):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"応答生成エラー: {e}")
            return f"[エラー: {str(e)}]"
    
    def evaluate_response(self, test_case: Dict, response: str) -> Dict:
        """応答品質評価"""
        result = {
            'test_id': test_case['id'],
            'language': test_case['language'],
            'category': test_case['category'],
            'input': test_case['input'],
            'response': response,
            'evaluation': {}
        }
        
        # 基本評価
        result['evaluation']['response_length'] = len(response)
        result['evaluation']['is_empty'] = len(response.strip()) == 0
        result['evaluation']['has_error'] = response.startswith('[エラー:')
        
        # キーワード評価
        response_lower = response.lower()
        matched_keywords = []
        for keyword in test_case['expected_keywords']:
            if keyword.lower() in response_lower:
                matched_keywords.append(keyword)
        
        result['evaluation']['matched_keywords'] = matched_keywords
        result['evaluation']['keyword_score'] = len(matched_keywords) / len(test_case['expected_keywords'])
        
        # 言語適合性評価
        if test_case['language'] == '日本語':
            # 日本語文字の存在確認
            japanese_chars = sum(1 for char in response if '\u3040' <= char <= '\u309F' or 
                                '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FAF')
            result['evaluation']['language_appropriateness'] = japanese_chars / max(len(response), 1)
        else:
            # 英語らしさ（アルファベットの割合）
            english_chars = sum(1 for char in response if char.isalpha())
            result['evaluation']['language_appropriateness'] = english_chars / max(len(response), 1)
        
        # 総合スコア
        if result['evaluation']['has_error']:
            result['evaluation']['overall_score'] = 0.0
        elif result['evaluation']['is_empty']:
            result['evaluation']['overall_score'] = 0.1
        else:
            score = (
                result['evaluation']['keyword_score'] * 0.4 +
                result['evaluation']['language_appropriateness'] * 0.3 +
                min(result['evaluation']['response_length'] / 50, 1.0) * 0.3
            )
            result['evaluation']['overall_score'] = min(score, 1.0)
        
        return result
    
    def run_test(self, test_case: Dict) -> Dict:
        """単一テスト実行"""
        logger.info(f"テスト実行: {test_case['id']} - {test_case['input']}")
        
        start_time = time.time()
        
        # 応答生成
        response = self.generate_response(test_case['input'])
        
        # 実行時間計測
        execution_time = time.time() - start_time
        
        # 評価
        result = self.evaluate_response(test_case, response)
        result['execution_time'] = execution_time
        
        logger.info(f"応答: {response[:100]}...")
        logger.info(f"スコア: {result['evaluation']['overall_score']:.2f}")
        
        return result
    
    def run_all_tests(self) -> Dict:
        """全テスト実行"""
        logger.info("チャットテスト開始")
        
        if self.model is None:
            self.load_model()
        
        # 各テスト実行
        for test_case in self.test_cases:
            try:
                result = self.run_test(test_case)
                self.results['test_results'].append(result)
            except Exception as e:
                logger.error(f"テスト {test_case['id']} でエラー: {e}")
                # エラー結果を記録
                error_result = {
                    'test_id': test_case['id'],
                    'language': test_case['language'],
                    'category': test_case['category'],
                    'input': test_case['input'],
                    'response': f"[エラー: {str(e)}]",
                    'evaluation': {
                        'overall_score': 0.0,
                        'has_error': True
                    },
                    'execution_time': 0.0
                }
                self.results['test_results'].append(error_result)
        
        # サマリ生成
        self._generate_summary()
        
        logger.info("チャットテスト完了")
        return self.results
    
    def _generate_summary(self) -> None:
        """結果サマリ生成"""
        if not self.results['test_results']:
            return
        
        # 基本統計
        total_tests = len(self.results['test_results'])
        successful_tests = sum(1 for r in self.results['test_results'] 
                              if not r['evaluation'].get('has_error', False))
        
        # スコア統計
        scores = [r['evaluation']['overall_score'] for r in self.results['test_results']]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # 言語別統計
        jp_results = [r for r in self.results['test_results'] if r['language'] == '日本語']
        en_results = [r for r in self.results['test_results'] if r['language'] == 'English']
        
        jp_avg_score = sum(r['evaluation']['overall_score'] for r in jp_results) / len(jp_results) if jp_results else 0
        en_avg_score = sum(r['evaluation']['overall_score'] for r in en_results) / len(en_results) if en_results else 0
        
        # カテゴリ別統計
        categories = {}
        for result in self.results['test_results']:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result['evaluation']['overall_score'])
        
        category_avg = {cat: sum(scores)/len(scores) for cat, scores in categories.items()}
        
        # 実行時間統計
        execution_times = [r.get('execution_time', 0) for r in self.results['test_results']]
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        self.results['summary'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0,
            'average_score': avg_score,
            'japanese_avg_score': jp_avg_score,
            'english_avg_score': en_avg_score,
            'category_scores': category_avg,
            'average_execution_time': avg_time,
            'total_execution_time': sum(execution_times)
        }
    
    def save_results(self, output_file: str = None) -> str:
        """結果保存"""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"chat_test_results_{timestamp}.json"
        
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"結果を保存: {output_path}")
        return str(output_path)
    
    def print_summary(self) -> None:
        """結果サマリ表示"""
        summary = self.results['summary']
        
        print("\n" + "="*60)
        print("チャットテスト結果サマリ")
        print("="*60)
        print(f"モデル: {self.results['model_info']['name']}")
        print(f"デバイス: {self.results['model_info']['device']}")
        print(f"実行日時: {self.results['timestamp']}")
        print()
        
        print(f"総テスト数: {summary['total_tests']}")
        print(f"成功テスト数: {summary['successful_tests']}")
        print(f"成功率: {summary['success_rate']:.1%}")
        print(f"平均スコア: {summary['average_score']:.3f}/1.0")
        print()
        
        print("言語別スコア:")
        print(f"  日本語: {summary['japanese_avg_score']:.3f}/1.0")
        print(f"  English: {summary['english_avg_score']:.3f}/1.0")
        print()
        
        print("カテゴリ別スコア:")
        for category, score in summary['category_scores'].items():
            print(f"  {category}: {score:.3f}/1.0")
        print()
        
        print(f"平均実行時間: {summary['average_execution_time']:.2f}秒")
        print(f"総実行時間: {summary['total_execution_time']:.2f}秒")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='LLM Chat Test Suite')
    parser.add_argument('--model', default='microsoft/DialoGPT-small',
                       help='使用するモデル名 (default: microsoft/DialoGPT-small)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='使用デバイス (default: auto)')
    parser.add_argument('--output', help='結果出力ファイル名')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='詳細ログ出力')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # テストスイート実行
        logger.info(f"LLM Chat Test開始 - モデル: {args.model}")
        
        test_suite = ChatTestSuite(model_name=args.model, device=args.device)
        results = test_suite.run_all_tests()
        
        # 結果表示
        test_suite.print_summary()
        
        # 結果保存
        output_file = test_suite.save_results(args.output)
        
        print(f"\n詳細結果: {output_file}")
        print("ログファイル: chat_test.log")
        
        # 簡易品質評価
        avg_score = results['summary']['average_score']
        if avg_score >= 0.7:
            print("\n✅ 品質評価: 良好")
        elif avg_score >= 0.5:
            print("\n⚠️ 品質評価: 改善の余地あり")
        else:
            print("\n❌ 品質評価: 要改善")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("ユーザによる中断")
        return 1
    except Exception as e:
        logger.error(f"実行エラー: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())