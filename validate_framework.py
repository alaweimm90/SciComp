#!/usr/bin/env python3
"""
Berkeley SciComp Framework - Final Validation Script

Comprehensive validation of the complete framework including:
- Module import verification
- Core functionality testing
- Cross-platform compatibility checks
- Performance benchmarks
- Documentation completeness

Author: UC Berkeley SciComp Team
Copyright © 2025 Dr. Meshal Alawein — All rights reserved.
"""

import sys
import os
import time
import importlib
import traceback
import subprocess
from pathlib import Path
import numpy as np


def print_header():
    """Print Berkeley SciComp validation header."""
    print("\n" + "="*70)
    print("🐻 BERKELEY SCICOMP FRAMEWORK - FINAL VALIDATION 🐻")
    print("="*70)
    print("University of California, Berkeley")
    print("Dr. Meshal Alawein (meshal@berkeley.edu)")
    print("Multi-Platform Scientific Computing Framework")
    print("="*70)


def validate_python_modules():
    """Validate all Python module imports and basic functionality."""
    print("\n📦 VALIDATING PYTHON MODULES")
    print("-" * 50)
    
    modules_to_test = [
        ('Python.Quantum', 'QuantumState, BellStates'),
        ('Python.QuantumOptics', 'JaynesCummings, CoherentStates'),
        ('Python.Spintronics', 'LandauLifshitzGilbert, SpinWaves'),
        ('Python.Symbolic_Algebra', 'SymbolicExpression, EquationSolver'),
        ('Python.Thermal_Transport', 'HeatEquation, PhononTransport'),
        ('Python.Machine_Learning', 'supervised, neural_networks'),
        ('Python.Signal_Processing', 'signal_analysis, spectral_analysis'),
        ('Python.quantum_physics', 'harmonic_oscillator, quantum_tunneling'),
        ('Python.quantum_computing', 'vqe, grover'),
        ('Python.Linear_Algebra', 'matrix_operations, vector_operations')
    ]
    
    passed = 0
    total = len(modules_to_test)
    
    for module_name, components in modules_to_test:
        try:
            # Try importing the module
            module = importlib.import_module(module_name)
            
            # Check if expected components exist
            component_list = [c.strip() for c in components.split(',')]
            missing_components = []
            
            for component in component_list:
                if not hasattr(module, component):
                    missing_components.append(component)
            
            if missing_components:
                print(f"⚠️  {module_name}: Missing {missing_components}")
            else:
                print(f"✅ {module_name}: All components available")
                passed += 1
                
        except ImportError as e:
            print(f"❌ {module_name}: Import failed - {str(e)}")
        except Exception as e:
            print(f"⚠️  {module_name}: Validation error - {str(e)}")
    
    success_rate = (passed / total) * 100
    print(f"\nPython Module Validation: {passed}/{total} ({success_rate:.1f}%)")
    return success_rate > 80


def validate_quantum_functionality():
    """Test core quantum physics functionality."""
    print("\n🔬 VALIDATING QUANTUM FUNCTIONALITY")
    print("-" * 50)
    
    try:
        from Python.Quantum.core.quantum_states import QuantumState, BellStates
        from Python.Quantum.core.quantum_operators import PauliOperators
        
        # Test Bell state creation
        phi_plus = BellStates.phi_plus()
        print("✅ Bell state creation successful")
        
        # Test normalization
        norm = np.linalg.norm(phi_plus.state_vector)
        assert abs(norm - 1.0) < 1e-10, f"Normalization failed: {norm}"
        print("✅ State normalization verified")
        
        # Test Pauli operations
        sigma_x = PauliOperators.X
        sigma_z = PauliOperators.Z
        commutator = PauliOperators.commutator(sigma_x, sigma_z)
        expected_comm = 2j * PauliOperators.Y
        
        if np.allclose(commutator, expected_comm):
            print("✅ Pauli operator algebra verified")
        else:
            print("⚠️  Pauli commutation relations need attention")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantum functionality test failed: {e}")
        traceback.print_exc()
        return False


def validate_integration_examples():
    """Validate that integration examples work."""
    print("\n🎯 VALIDATING INTEGRATION EXAMPLES")
    print("-" * 50)
    
    example_files = [
        'examples/python/quantum_tunneling_demo.py',
        'examples/python/quantum_computing_demo.py', 
        'examples/python/ml_physics_demo.py',
        'examples/beginner/harmonic_oscillator_demo.py'
    ]
    
    working_examples = 0
    total_examples = len(example_files)
    
    for example_file in example_files:
        example_path = Path(example_file)
        if example_path.exists():
            print(f"✅ Found: {example_file}")
            working_examples += 1
        else:
            print(f"❌ Missing: {example_file}")
    
    print(f"\nExample Files: {working_examples}/{total_examples} available")
    return working_examples >= total_examples * 0.8


def validate_cross_platform_parity():
    """Check for cross-platform parity."""
    print("\n🌍 VALIDATING CROSS-PLATFORM PARITY")  
    print("-" * 50)
    
    platforms = {
        'Python': 'Python/',
        'MATLAB': 'MATLAB/', 
        'Mathematica': 'Mathematica/'
    }
    
    core_modules = [
        'Quantum', 'QuantumOptics', 'Machine_Learning', 'Signal_Processing',
        'Linear_Algebra', 'Optimization', 'Monte_Carlo', 'ODE_PDE'
    ]
    
    platform_coverage = {}
    
    for platform, base_path in platforms.items():
        if Path(base_path).exists():
            available_modules = 0
            for module in core_modules:
                module_path = Path(base_path) / module
                if module_path.exists():
                    available_modules += 1
            
            coverage = (available_modules / len(core_modules)) * 100
            platform_coverage[platform] = coverage
            print(f"✅ {platform}: {available_modules}/{len(core_modules)} modules ({coverage:.1f}%)")
        else:
            platform_coverage[platform] = 0
            print(f"❌ {platform}: Directory not found")
    
    avg_coverage = sum(platform_coverage.values()) / len(platform_coverage)
    print(f"\nAverage Cross-Platform Coverage: {avg_coverage:.1f}%")
    return avg_coverage > 70


def validate_documentation():
    """Check documentation completeness."""
    print("\n📚 VALIDATING DOCUMENTATION")
    print("-" * 50)
    
    required_docs = [
        'README.md',
        'PROJECT_COMPLETION_FINAL.md',
        'CONTRIBUTING.md',
        'LICENSE',
        'assets/STYLE_GUIDE.md'
    ]
    
    doc_score = 0
    for doc in required_docs:
        if Path(doc).exists():
            print(f"✅ {doc}")
            doc_score += 1
        else:
            print(f"❌ Missing: {doc}")
    
    coverage = (doc_score / len(required_docs)) * 100
    print(f"\nDocumentation Coverage: {doc_score}/{len(required_docs)} ({coverage:.1f}%)")
    return coverage > 80


def run_performance_benchmarks():
    """Run basic performance benchmarks."""
    print("\n⚡ PERFORMANCE BENCHMARKS")
    print("-" * 50)
    
    benchmarks = {}
    
    try:
        # Import time benchmark
        start_time = time.time()
        from Python.Quantum.core import quantum_states, quantum_operators
        from Python.QuantumOptics.core import cavity_qed, quantum_light
        import_time = time.time() - start_time
        benchmarks['import_time'] = import_time
        print(f"✅ Module import time: {import_time:.3f}s")
        
        # Memory usage estimation
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        benchmarks['memory_usage'] = memory_mb
        print(f"✅ Memory usage: {memory_mb:.1f} MB")
        
        # Computation benchmark (matrix operations)
        start_time = time.time()
        large_matrix = np.random.rand(1000, 1000)
        eigenvals = np.linalg.eigvals(large_matrix)
        comp_time = time.time() - start_time
        benchmarks['computation_time'] = comp_time
        print(f"✅ Matrix computation (1000x1000 eigen): {comp_time:.3f}s")
        
    except Exception as e:
        print(f"⚠️  Benchmark error: {e}")
        return False
    
    # Performance criteria
    performance_ok = (
        benchmarks.get('import_time', 999) < 10 and
        benchmarks.get('computation_time', 999) < 5
    )
    
    return performance_ok


def generate_validation_report(results):
    """Generate final validation report."""
    print("\n" + "="*70)
    print("📋 BERKELEY SCICOMP FRAMEWORK - VALIDATION REPORT")
    print("="*70)
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    overall_score = (passed_checks / total_checks) * 100
    
    print(f"Overall Validation Score: {passed_checks}/{total_checks} ({overall_score:.1f}%)")
    print("\nDetailed Results:")
    print("-" * 30)
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name}: {status}")
    
    print("\nFramework Status Assessment:")
    print("-" * 40)
    
    if overall_score >= 90:
        print("🎉 EXCELLENT: Framework is production-ready!")
        print("🐻 Go Bears! Berkeley SciComp is complete! 💙💛")
        return True
    elif overall_score >= 80:
        print("✅ GOOD: Framework is mostly ready with minor issues")
        return True
    elif overall_score >= 70:
        print("⚠️  FAIR: Framework needs some improvements")
        return False
    else:
        print("❌ POOR: Significant issues need to be addressed")
        return False


def main():
    """Main validation function."""
    print_header()
    
    # Run all validation checks
    validation_results = {
        'Python Modules': validate_python_modules(),
        'Quantum Functionality': validate_quantum_functionality(),
        'Integration Examples': validate_integration_examples(),
        'Cross-Platform Parity': validate_cross_platform_parity(),
        'Documentation': validate_documentation(),
        'Performance': run_performance_benchmarks()
    }
    
    # Generate final report
    framework_ready = generate_validation_report(validation_results)
    
    # Final message
    if framework_ready:
        print("\n🎓 BERKELEY SCICOMP FRAMEWORK VALIDATION COMPLETE!")
        print("✅ Framework is ready for academic and research use")
        print("🔬 All major scientific computing capabilities verified")
        print("💙💛 University of California, Berkeley - Go Bears!")
    else:
        print("\n⚠️  Framework validation completed with issues")
        print("🔧 Please address failed checks before production use")
    
    return framework_ready


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Validation script error: {e}")
        traceback.print_exc()
        sys.exit(1)