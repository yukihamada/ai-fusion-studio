#!/usr/bin/env python3
"""
Simple test runner that bypasses dependency conflicts
"""

import sys
import os
import unittest
import tempfile
import json
import yaml
from pathlib import Path
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestCoverage:
    """Simple test coverage tracker"""
    
    def __init__(self):
        self.tested_functions = set()
        self.total_functions = set()
        
    def mark_tested(self, module_name, function_name):
        self.tested_functions.add(f"{module_name}.{function_name}")
        
    def mark_total(self, module_name, function_name):
        self.total_functions.add(f"{module_name}.{function_name}")
        
    def get_coverage(self):
        if not self.total_functions:
            return 0.0
        return len(self.tested_functions) / len(self.total_functions) * 100

class MockTransformers:
    """Mock transformers module to avoid dependency conflicts"""
    
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            class MockTokenizer:
                def encode(self, text, **kwargs):
                    return [1, 2, 3, 4, 5]
                def decode(self, tokens, **kwargs):
                    return "mock response"
                def __call__(self, text, **kwargs):
                    return {"input_ids": [[1, 2, 3, 4, 5]], "attention_mask": [[1, 1, 1, 1, 1]]}
            return MockTokenizer()
    
    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            class MockModel:
                def generate(self, **kwargs):
                    return [[1, 2, 3, 4, 5, 6]]
                def to(self, device):
                    return self
                def eval(self):
                    return self
            return MockModel()

# Mock modules to avoid dependencies
sys.modules['transformers'] = MockTransformers()
sys.modules['torch'] = type('MockTorch', (), {
    'cuda': type('MockCuda', (), {'is_available': lambda: True})(),
    'backends': type('MockBackends', (), {
        'mps': type('MockMPS', (), {'is_available': lambda: True})()
    })(),
    'float16': 'float16',
    'device': lambda x: x
})()

def test_evaluate_module():
    """Test evaluate.py module functions"""
    print("Testing evaluate.py module...")
    
    # Import after mocking
    from scripts.evaluate import JapaneseLLMEvaluator
    
    coverage = TestCoverage()
    
    # Mark total functions
    coverage.mark_total("evaluate", "__init__")
    coverage.mark_total("evaluate", "evaluate_mt_bench_jp")
    coverage.mark_total("evaluate", "evaluate_jglue")
    coverage.mark_total("evaluate", "evaluate_mathematical_reasoning")
    coverage.mark_total("evaluate", "generate_response")
    coverage.mark_total("evaluate", "score_response")
    coverage.mark_total("evaluate", "save_evaluation_report")
    coverage.mark_total("evaluate", "main")
    
    # Test initialization
    try:
        evaluator = JapaneseLLMEvaluator("mock-model", "test-output")
        coverage.mark_tested("evaluate", "__init__")
        print("✅ JapaneseLLMEvaluator initialization")
    except Exception as e:
        print(f"❌ JapaneseLLMEvaluator initialization: {e}")
    
    # Test MT-Bench evaluation
    try:
        result = evaluator.evaluate_mt_bench_jp()
        assert "overall_score" in result
        coverage.mark_tested("evaluate", "evaluate_mt_bench_jp")
        print("✅ MT-Bench evaluation")
    except Exception as e:
        print(f"❌ MT-Bench evaluation: {e}")
    
    # Test JGLUE evaluation
    try:
        result = evaluator.evaluate_jglue()
        assert "overall_score" in result
        coverage.mark_tested("evaluate", "evaluate_jglue")
        print("✅ JGLUE evaluation")
    except Exception as e:
        print(f"❌ JGLUE evaluation: {e}")
    
    # Test mathematical reasoning
    try:
        result = evaluator.evaluate_mathematical_reasoning()
        assert "accuracy" in result
        coverage.mark_tested("evaluate", "evaluate_mathematical_reasoning")
        print("✅ Mathematical reasoning evaluation")
    except Exception as e:
        print(f"❌ Mathematical reasoning evaluation: {e}")
    
    # Test response generation
    try:
        response = evaluator.generate_response("Test prompt")
        assert isinstance(response, str)
        coverage.mark_tested("evaluate", "generate_response")
        print("✅ Response generation")
    except Exception as e:
        print(f"❌ Response generation: {e}")
    
    # Test response scoring
    try:
        score = evaluator.score_response("Test response", "Test expected")
        assert 0 <= score <= 10
        coverage.mark_tested("evaluate", "score_response")
        print("✅ Response scoring")
    except Exception as e:
        print(f"❌ Response scoring: {e}")
    
    # Test report saving
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_report.json")
            test_results = {"test": "data"}
            evaluator.save_evaluation_report(test_results, output_path)
            assert os.path.exists(output_path)
            coverage.mark_tested("evaluate", "save_evaluation_report")
            print("✅ Report saving")
    except Exception as e:
        print(f"❌ Report saving: {e}")
    
    print(f"Coverage for evaluate.py: {coverage.get_coverage():.1f}%")
    return coverage.get_coverage()

def test_experiment_tracker_module():
    """Test experiment_tracker.py module functions"""
    print("\nTesting experiment_tracker.py module...")
    
    from scripts.experiment_tracker import ExperimentTracker
    
    coverage = TestCoverage()
    
    # Mark total functions
    coverage.mark_total("experiment_tracker", "__init__")
    coverage.mark_total("experiment_tracker", "register_experiment")
    coverage.mark_total("experiment_tracker", "update_experiment")
    coverage.mark_total("experiment_tracker", "get_experiment")
    coverage.mark_total("experiment_tracker", "list_experiments")
    coverage.mark_total("experiment_tracker", "generate_leaderboard")
    coverage.mark_total("experiment_tracker", "compare_experiments")
    coverage.mark_total("experiment_tracker", "visualize_results")
    coverage.mark_total("experiment_tracker", "export_results")
    coverage.mark_total("experiment_tracker", "main")
    
    # Test initialization
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_experiments.json")
            tracker = ExperimentTracker(db_path)
            coverage.mark_tested("experiment_tracker", "__init__")
            print("✅ ExperimentTracker initialization")
    except Exception as e:
        print(f"❌ ExperimentTracker initialization: {e}")
    
    # Test experiment registration
    try:
        experiment_data = {
            "experiment_id": "test_001",
            "merge_method": "slerp",
            "models": [{"name": "model1", "weight": 0.5}]
        }
        experiment_id = tracker.register_experiment(experiment_data)
        assert experiment_id == "test_001"
        coverage.mark_tested("experiment_tracker", "register_experiment")
        print("✅ Experiment registration")
    except Exception as e:
        print(f"❌ Experiment registration: {e}")
    
    # Test experiment update
    try:
        update_data = {"status": "completed", "score": 8.5}
        tracker.update_experiment("test_001", update_data)
        coverage.mark_tested("experiment_tracker", "update_experiment")
        print("✅ Experiment update")
    except Exception as e:
        print(f"❌ Experiment update: {e}")
    
    # Test experiment retrieval
    try:
        experiment = tracker.get_experiment("test_001")
        assert experiment["experiment_id"] == "test_001"
        coverage.mark_tested("experiment_tracker", "get_experiment")
        print("✅ Experiment retrieval")
    except Exception as e:
        print(f"❌ Experiment retrieval: {e}")
    
    # Test experiment listing
    try:
        experiments = tracker.list_experiments()
        assert len(experiments) >= 1
        coverage.mark_tested("experiment_tracker", "list_experiments")
        print("✅ Experiment listing")
    except Exception as e:
        print(f"❌ Experiment listing: {e}")
    
    print(f"Coverage for experiment_tracker.py: {coverage.get_coverage():.1f}%")
    return coverage.get_coverage()

def test_all_modules():
    """Test all modules and calculate overall coverage"""
    print("🧪 Starting comprehensive test suite...")
    print("=" * 60)
    
    coverages = []
    
    # Test each module
    try:
        coverages.append(test_evaluate_module())
    except Exception as e:
        print(f"❌ evaluate.py testing failed: {e}")
        coverages.append(0)
    
    try:
        coverages.append(test_experiment_tracker_module())
    except Exception as e:
        print(f"❌ experiment_tracker.py testing failed: {e}")
        coverages.append(0)
    
    # Calculate overall coverage
    overall_coverage = sum(coverages) / len(coverages) if coverages else 0
    
    print("\n" + "=" * 60)
    print("📊 FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Overall Coverage: {overall_coverage:.1f}%")
    
    if overall_coverage >= 90:
        print("🎉 EXCELLENT! High test coverage achieved!")
    elif overall_coverage >= 70:
        print("✅ GOOD! Acceptable test coverage")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Low test coverage")
    
    return overall_coverage

if __name__ == "__main__":
    test_all_modules()