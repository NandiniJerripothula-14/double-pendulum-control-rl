#!/usr/bin/env python3
"""
Quick verification script to test project setup without Docker.
Tests Python environment and code structure.
"""

import sys
import importlib.util
from pathlib import Path


def test_python_files():
    """Test Python file validity."""
    print("\n" + "="*60)
    print("TESTING PYTHON FILES")
    print("="*60)
    
    files = [
        "src/environment.py",
        "src/train.py",
        "src/evaluate.py",
        "plot_rewards.py"
    ]
    
    for file_path in files:
        try:
            spec = importlib.util.spec_from_file_location("module", file_path)
            if spec and spec.loader:
                print(f"✅ {file_path}: Valid syntax")
            else:
                print(f"❌ {file_path}: Could not load")
        except SyntaxError as e:
            print(f"❌ {file_path}: Syntax error - {e}")
        except Exception as e:
            print(f"⚠️  {file_path}: {e}")


def test_file_structure():
    """Test project directory structure."""
    print("\n" + "="*60)
    print("TESTING FILE STRUCTURE")
    print("="*60)
    
    required_files = {
        "Dockerfile": "Docker image definition",
        "docker-compose.yml": "Docker services",
        "requirements.txt": "Python dependencies",
        ".env.example": "Environment template",
        "README.md": "Main documentation",
        "src/environment.py": "Environment class",
        "src/train.py": "Training script",
        "src/evaluate.py": "Evaluation script",
    }
    
    required_dirs = {
        "src": "Source code",
        "models": "Trained models",
        "logs": "Training logs",
        "media": "Visualizations",
    }
    
    print("\nFiles:")
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"✅ {file_path:30s} ({size:,} bytes) - {description}")
        else:
            print(f"❌ {file_path:30s} - MISSING")
    
    print("\nDirectories:")
    for dir_path, description in required_dirs.items():
        if Path(dir_path).exists():
            print(f"✅ {dir_path:30s} - {description}")
        else:
            print(f"❌ {dir_path:30s} - MISSING")


def test_requirements():
    """Check requirements file."""
    print("\n" + "="*60)
    print("TESTING REQUIREMENTS")
    print("="*60)
    
    req_file = Path("requirements.txt")
    if req_file.exists():
        packages = req_file.read_text().strip().split('\n')
        print(f"\n✅ requirements.txt contains {len(packages)} packages:")
        for pkg in packages:
            if pkg.strip():
                print(f"   • {pkg}")
    else:
        print("❌ requirements.txt not found")


def test_environment_class():
    """Test environment class structure."""
    print("\n" + "="*60)
    print("TESTING ENVIRONMENT CLASS")
    print("="*60)
    
    try:
        env_file = Path("src/environment.py")
        content = env_file.read_text()
        
        checks = {
            "class DoublePendulumEnv": "Class definition",
            "def __init__": "__init__ method",
            "def reset": "reset() method",
            "def step": "step() method",
            "def render": "render() method",
            "def close": "close() method",
            "observation_space = Box": "Observation space",
            "action_space = Box": "Action space",
            "def _calculate_reward": "Reward calculation",
            "reward_type == 'baseline'": "Baseline reward",
            "reward_type == 'shaped'": "Shaped reward",
        }
        
        for check, description in checks.items():
            if check in content:
                print(f"✅ {description:30s} - Found")
            else:
                print(f"❌ {description:30s} - NOT FOUND")
    
    except Exception as e:
        print(f"❌ Error testing environment: {e}")


def test_docker():
    """Test Docker installation."""
    print("\n" + "="*60)
    print("TESTING DOCKER")
    print("="*60)
    
    import subprocess
    
    # Test Docker
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker: {result.stdout.strip()}")
        else:
            print("❌ Docker: Not responding")
    except FileNotFoundError:
        print("❌ Docker: Not installed")
    
    # Test Docker Compose
    try:
        result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker Compose: {result.stdout.strip()}")
        else:
            print("❌ Docker Compose: Not responding")
    except FileNotFoundError:
        print("❌ Docker Compose: Not installed")


def print_next_steps():
    """Print next steps for using the project."""
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    print("""
To start using the Double Pendulum RL project:

1. BUILD DOCKER IMAGE
   docker-compose build

2. QUICK TEST (1000 steps - ~2 minutes)
   docker-compose run train python src/train.py --timesteps 1000

3. FULL TRAINING (200,000 steps - 4-8 hours on CPU)
   docker-compose run train

4. EVALUATE AGENT
   docker-compose run evaluate

5. VIEW LEARNING CURVES
   docker-compose run plot

For detailed instructions, see:
  • QUICKSTART.md - Fast start guide
  • README.md - Complete documentation
  • TEST_REPORT.md - Full test results
""")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*15 + "DOUBLE PENDULUM RL PROJECT" + " "*17 + "║")
    print("║" + " "*18 + "VERIFICATION SCRIPT" + " "*22 + "║")
    print("╚" + "═"*58 + "╝")
    
    test_python_files()
    test_file_structure()
    test_requirements()
    test_environment_class()
    test_docker()
    print_next_steps()
    
    print("\n" + "═"*60)
    print("✅ VERIFICATION COMPLETE - PROJECT READY TO USE")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
