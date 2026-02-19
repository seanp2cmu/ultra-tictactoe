"""Build Cython/C++ extensions on HF Spaces startup"""
import os
import subprocess
import sys


def build_extensions():
    """Build Cython and C++ extensions on HF Spaces startup"""
    if not os.environ.get('SPACE_ID'):
        return  # Skip on local
    
    try:
        # Install build dependencies first
        print("Installing build dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "Cython", "pybind11", "numpy"], 
                      capture_output=True, check=True)
        print("✓ Build dependencies installed")
        
        print("Building Cython extensions...")
        result = subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], 
                      capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Cython extensions built")
        else:
            print(f"⚠ Cython build error: {result.stderr}")
        
        print("Building C++ extensions...")
        result = subprocess.run([sys.executable, "setup_cpp.py", "build_ext", "--inplace"],
                      capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ C++ extensions built")
        else:
            print(f"⚠ C++ build error: {result.stderr}")
    except Exception as e:
        print(f"⚠ Build error: {e}")
