# install_script.py
#!/usr/bin/env python3
"""
Installation and setup script for Football Sound Detection System
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ {description} failed: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8+ is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_system_dependencies():
    """Install system-level dependencies"""
    system = platform.system().lower()
    
    if system == "linux":
        # Ubuntu/Debian
        commands = [
            "sudo apt update",
            "sudo apt install -y ffmpeg portaudio19-dev python3-dev",
        ]
    elif system == "darwin":  # macOS
        commands = [
            "brew install ffmpeg portaudio",
        ]
    elif system == "windows":
        print("Please install FFmpeg manually for Windows:")
        print("1. Download FFmpeg from https://ffmpeg.org/download.html")
        print("2. Add FFmpeg to your PATH environment variable")
        return True
    else:
        print(f"Unsupported system: {system}")
        return False
    
    for cmd in commands:
        if not run_command(cmd, f"Installing system dependencies ({cmd})"):
            return False
    
    return True

def create_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = "football_detection_env"
    
    if os.path.exists(venv_path):
        print(f"Virtual environment '{venv_path}' already exists")
        return True
    
    if not run_command(f"python -m venv {venv_path}", "Creating virtual environment"):
        return False
    
    print(f"Virtual environment created at: {venv_path}")
    print("To activate it:")
    if platform.system().lower() == "windows":
        print(f"  {venv_path}\\Scripts\\activate")
    else:
        print(f"  source {venv_path}/bin/activate")
    
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    commands = [
        "pip install --upgrade pip",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        "pip install numpy>=1.21.0",
        "pip install librosa>=0.9.0",
        "pip install openai-whisper>=20230314",
        "pip install scipy>=1.7.0",
        "pip install matplotlib>=3.5.0",
        "pip install ffmpeg-python>=0.2.0",
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"Installing Python package ({cmd.split()[-1]})"):
            return False
    
    return True

def test_installation():
    """Test if installation was successful"""
    test_code = '''
import numpy as np
import librosa
import whisper
import scipy
import matplotlib
print("✓ All packages imported successfully")

# Test Whisper model loading
try:
    model = whisper.load_model("tiny")
    print("✓ Whisper model loaded successfully")
except Exception as e:
    print(f"✗ Whisper model loading failed: {e}")

print("Installation test completed!")
'''
    
    print("\nTesting installation...")
    try:
        exec(test_code)
        return True
    except Exception as e:
        print(f"✗ Installation test failed: {e}")
        return False

def main():
    """Main installation process"""
    print("=== Football Sound Detection System Installation ===\n")
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing system dependencies", install_system_dependencies),
        ("Creating virtual environment", create_virtual_environment),
        ("Installing Python dependencies", install_python_dependencies),
        ("Testing installation", test_installation),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        if not step_func():
            failed_steps.append(step_name)
    
    print("\n" + "="*50)
    if failed_steps:
        print("❌ Installation completed with errors:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nPlease fix the errors above and run the script again.")
    else:
        print("✅ Installation completed successfully!")
        print("\nNext steps:")
        print("1. Activate the virtual environment:")
        if platform.system().lower() == "windows":
            print("   football_detection_env\\Scripts\\activate")
        else:
            print("   source football_detection_env/bin/activate")
        print("2. Run the football sound detection script:")
        print("   python football_sound_detection.py")

if __name__ == "__main__":
    main()