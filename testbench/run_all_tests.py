"""
Test Runner - Run All Test Suites

Executes all test files and generates a comprehensive report.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_tests():
    """
    Run all test suites and generate report
    """
    print("\n" + "=" * 80)
    print(" NEURAL SIGNAL DSP - COMPREHENSIVE TEST SUITE")
    print("=" * 80 + "\n")
    
    test_modules = [
        ('test_signal_gen', 'Neural Signal Generator Tests'),
        ('test_adc_sim', 'ADC Simulator Tests'),
        ('test_integration', 'Integration Tests')
    ]
    
    all_results = []
    
    for module_name, description in test_modules:
        print(f"\n{'─' * 80}")
        print(f" {description}")
        print('─' * 80)
        
        try:
            # Import the test module
            module = __import__(module_name)
            
            # Run the validation tests
            if hasattr(module, 'run_validation_tests'):
                success = module.run_validation_tests()
                all_results.append((description, success))
            elif hasattr(module, 'run_integration_tests'):
                success = module.run_integration_tests()
                all_results.append((description, success))
            else:
                print(f"Warning: No test runner found in {module_name}")
                all_results.append((description, None))
        
        except Exception as e:
            print(f"Error running {module_name}: {str(e)}")
            all_results.append((description, False))
    
    # Generate final report
    print("\n\n" + "=" * 80)
    print(" FINAL TEST REPORT")
    print("=" * 80)
    
    total_suites = len(all_results)
    passed_suites = sum(1 for _, result in all_results if result is True)
    failed_suites = sum(1 for _, result in all_results if result is False)
    skipped_suites = sum(1 for _, result in all_results if result is None)
    
    for description, result in all_results:
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⊘ SKIPPED"
        
        print(f"{status:>12} - {description}")
    
    print("\n" + "-" * 80)
    print(f"Total Test Suites: {total_suites}")
    print(f"  Passed: {passed_suites}")
    print(f"  Failed: {failed_suites}")
    print(f"  Skipped: {skipped_suites}")
    print("-" * 80)
    
    if failed_suites == 0 and skipped_suites == 0:
        print("\n✓ ALL TESTS PASSED! System validated and ready for use.")
        return True
    elif failed_suites > 0:
        print(f"\n✗ {failed_suites} test suite(s) failed. Please review errors above.")
        return False
    else:
        print(f"\n⊘ {skipped_suites} test suite(s) skipped.")
        return True


def run_with_pytest():
    """
    Run tests using pytest if available
    """
    try:
        import pytest
        
        print("\n" + "=" * 80)
        print(" Running tests with pytest")
        print("=" * 80 + "\n")
        
        # Run pytest with verbose output
        args = [
            'testbench/',
            '-v',
            '--tb=short',
            '--color=yes'
        ]
        
        result = pytest.main(args)
        
        return result == 0
    
    except ImportError:
        print("pytest not available, using built-in test runner")
        return None


if __name__ == '__main__':
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                 NEURAL SIGNAL DSP - TEST SUITE                              ║
║                                                                              ║
║  This test suite validates the functionality of all modules:                 ║
║    • Neural Signal Generator (signal_gen.py)                                 ║
║    • ADC Simulator (adc_sim.py)                                              ║
║    • Integration (complete pipeline)                                         ║
║                                                                              ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Try pytest first, fall back to built-in runner
    pytest_result = run_with_pytest()
    
    if pytest_result is None:
        # pytest not available, use built-in runner
        success = run_all_tests()
        sys.exit(0 if success else 1)
    else:
        sys.exit(0 if pytest_result else 1)

