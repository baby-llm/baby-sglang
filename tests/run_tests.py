#!/usr/bin/env python3
"""
Baby-SGLang Test Runner

Comprehensive test suite runner with multiple test categories:
- Unit tests: Individual component testing
- Integration tests: Cross-component testing  
- Performance tests: Benchmarking and resource usage
- Security tests: Cross-request isolation verification

Usage:
    python run_tests.py [OPTIONS]
    
Options:
    --unit          Run unit tests only
    --integration   Run integration tests only  
    --performance   Run performance tests only
    --security      Run security-focused tests only
    --all           Run all test suites (default)
    --verbose       Enable verbose output
    --coverage      Generate test coverage report
    --benchmark     Run full benchmark suite
    --device        Device to run tests on (cpu/cuda/mps)
"""

import argparse
import sys
import os
import unittest
import time
import subprocess
from typing import List, Dict, Any
import importlib.util

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestResult:
    """Container for test results."""
    
    def __init__(self, suite_name: str):
        self.suite_name = suite_name
        self.tests_run = 0
        self.failures = 0
        self.errors = 0
        self.skipped = 0
        self.duration = 0.0
        self.success = False
    
    def from_unittest_result(self, result: unittest.TestResult, duration: float):
        """Populate from unittest.TestResult."""
        self.tests_run = result.testsRun
        self.failures = len(result.failures)
        self.errors = len(result.errors)
        self.skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
        self.duration = duration
        self.success = self.failures == 0 and self.errors == 0
        return self
    
    def __str__(self):
        status = "PASS" if self.success else "FAIL"
        return (f"{self.suite_name}: {status} "
                f"({self.tests_run} tests, {self.failures} failures, "
                f"{self.errors} errors, {self.duration:.2f}s)")


class BabySGLangTestRunner:
    """Main test runner for Baby-SGLang."""
    
    def __init__(self, verbose: bool = False, device: str = "cpu"):
        self.verbose = verbose
        self.device = device
        self.results: List[TestResult] = []
        
        # Test discovery paths
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.unit_tests = [
            "test_memory_pool",
            "test_scheduler", 
            "test_qwen_model",
        ]
        self.integration_tests = [
            "test_integration",
        ]
        self.performance_tests = [
            "test_performance",
        ]
    
    def run_unit_tests(self) -> List[TestResult]:
        """Run all unit tests."""
        print("\\n" + "="*60)
        print("Running Unit Tests")
        print("="*60)
        
        results = []
        for test_module in self.unit_tests:
            result = self._run_test_module(test_module, "Unit")
            results.append(result)
            
        return results
    
    def run_integration_tests(self) -> List[TestResult]:
        """Run integration tests."""
        print("\\n" + "="*60)
        print("Running Integration Tests")
        print("="*60)
        
        results = []
        for test_module in self.integration_tests:
            result = self._run_test_module(test_module, "Integration")
            results.append(result)
            
        return results
    
    def run_performance_tests(self) -> List[TestResult]:
        """Run performance tests."""
        print("\\n" + "="*60)
        print("Running Performance Tests")
        print("="*60)
        
        results = []
        for test_module in self.performance_tests:
            result = self._run_test_module(test_module, "Performance")
            results.append(result)
            
        return results
    
    def run_security_tests(self) -> List[TestResult]:
        """Run security-focused tests."""
        print("\\n" + "="*60)
        print("Running Security Tests")
        print("="*60)
        
        # Security tests are embedded in other test modules
        # Run specific test cases that focus on security
        security_test_cases = [
            ("test_qwen_model", ["TestAttentionMechanism.test_prefill_attention_security",
                                "TestAttentionMechanism.test_decode_attention_security"]),
            ("test_integration", ["TestConcurrencyAndConsistency.test_cross_request_independence"]),
        ]
        
        results = []
        for module_name, test_cases in security_test_cases:
            result = self._run_specific_tests(module_name, test_cases, "Security")
            results.append(result)
            
        return results
    
    def run_benchmark_suite(self):
        """Run comprehensive benchmark suite."""
        print("\\n" + "="*60)
        print("Running Benchmark Suite")
        print("="*60)
        
        try:
            # Import and run benchmark
            from tests.test_performance import PerformanceBenchmark
            results = PerformanceBenchmark.run_benchmark_suite()
            
            print("\\nBenchmark completed successfully!")
            return results
            
        except Exception as e:
            print(f"Benchmark failed: {e}")
            return None
    
    def _run_test_module(self, module_name: str, suite_type: str) -> TestResult:
        """Run a specific test module."""
        print(f"\\nRunning {module_name}...")
        
        result = TestResult(f"{suite_type}: {module_name}")
        
        try:
            # Import the test module
            module_path = os.path.join(self.test_dir, f"{module_name}.py")
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Discover and run tests
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(module)
            
            # Run tests
            start_time = time.time()
            test_result = unittest.TextTestRunner(
                verbosity=2 if self.verbose else 1,
                stream=sys.stdout if self.verbose else open(os.devnull, 'w')
            ).run(suite)
            duration = time.time() - start_time
            
            result.from_unittest_result(test_result, duration)
            
        except Exception as e:
            print(f"ERROR: Failed to run {module_name}: {e}")
            result.errors = 1
            result.success = False
        
        print(result)
        return result
    
    def _run_specific_tests(self, module_name: str, test_cases: List[str], suite_type: str) -> TestResult:
        """Run specific test cases from a module."""
        print(f"\\nRunning {suite_type} tests from {module_name}...")
        
        result = TestResult(f"{suite_type}: {module_name} (selected)")
        
        try:
            # Import the test module
            module_path = os.path.join(self.test_dir, f"{module_name}.py")
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Create test suite from specific cases
            loader = unittest.TestLoader()
            suite = unittest.TestSuite()
            
            for test_case in test_cases:
                try:
                    suite.addTest(loader.loadTestsFromName(test_case, module))
                except Exception as e:
                    print(f"Warning: Could not load test {test_case}: {e}")
            
            # Run tests
            start_time = time.time()
            test_result = unittest.TextTestRunner(
                verbosity=2 if self.verbose else 1,
                stream=sys.stdout if self.verbose else open(os.devnull, 'w')
            ).run(suite)
            duration = time.time() - start_time
            
            result.from_unittest_result(test_result, duration)
            
        except Exception as e:
            print(f"ERROR: Failed to run {module_name} security tests: {e}")
            result.errors = 1
            result.success = False
        
        print(result)
        return result
    
    def generate_coverage_report(self) -> bool:
        """Generate test coverage report."""
        print("\\n" + "="*60)
        print("Generating Coverage Report")
        print("="*60)
        
        try:
            # Try to run coverage
            cmd = [
                sys.executable, "-m", "coverage", "run", "--source=.", 
                "--omit=tests/*", __file__, "--unit", "--integration"
            ]
            
            subprocess.run(cmd, check=True)
            
            # Generate report
            subprocess.run([sys.executable, "-m", "coverage", "report"], check=True)
            subprocess.run([sys.executable, "-m", "coverage", "html"], check=True)
            
            print("Coverage report generated in htmlcov/")
            return True
            
        except subprocess.CalledProcessError:
            print("Coverage generation failed. Install coverage: pip install coverage")
            return False
        except FileNotFoundError:
            print("Coverage not available. Install coverage: pip install coverage")
            return False
    
    def print_summary(self):
        """Print test results summary."""
        print("\\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        total_tests = sum(r.tests_run for r in self.results)
        total_failures = sum(r.failures for r in self.results)
        total_errors = sum(r.errors for r in self.results)
        total_duration = sum(r.duration for r in self.results)
        
        all_passed = all(r.success for r in self.results)
        
        print(f"Total Tests Run: {total_tests}")
        print(f"Total Failures: {total_failures}")
        print(f"Total Errors: {total_errors}")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"Overall Status: {'PASS' if all_passed else 'FAIL'}")
        
        print("\\nSuite Details:")
        for result in self.results:
            print(f"  {result}")
        
        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Baby-SGLang Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--security", action="store_true", help="Run security tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark suite")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                       help="Device for testing")
    
    args = parser.parse_args()
    
    # Default to all tests if no specific suite selected
    if not any([args.unit, args.integration, args.performance, args.security]):
        args.all = True
    
    print("Baby-SGLang Test Suite")
    print(f"Device: {args.device}")
    print(f"Verbose: {args.verbose}")
    
    runner = BabySGLangTestRunner(verbose=args.verbose, device=args.device)
    
    # Run selected test suites
    if args.unit or args.all:
        runner.results.extend(runner.run_unit_tests())
    
    if args.integration or args.all:
        runner.results.extend(runner.run_integration_tests())
    
    if args.performance or args.all:
        runner.results.extend(runner.run_performance_tests())
    
    if args.security or args.all:
        runner.results.extend(runner.run_security_tests())
    
    # Run benchmark if requested
    if args.benchmark:
        runner.run_benchmark_suite()
    
    # Generate coverage report if requested
    if args.coverage:
        runner.generate_coverage_report()
    
    # Print summary and determine exit code
    success = runner.print_summary()
    
    if not success:
        print("\\nSome tests failed. Please check the output above.")
        sys.exit(1)
    else:
        print("\\nAll tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()