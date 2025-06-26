#!/usr/bin/env python3
"""
Test script to check if all required dependencies are installed
"""

import sys

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {module_name} - OK")
        return True
    except ImportError as e:
        pkg = package_name or module_name
        print(f"❌ {module_name} - MISSING (install with: pip install {pkg})")
        print(f"   Error: {e}")
        return False

def main():
    print("🔍 Checking required dependencies for Football Sound Detection...")
    print("=" * 60)
    
    all_good = True
    
    # Core dependencies
    dependencies = [
        ("numpy", "numpy"),
        ("librosa", "librosa"),
        ("whisper", "openai-whisper"),
        ("scipy", "scipy"),
        ("dataclasses", None),  # Built-in for Python 3.7+
        ("json", None),  # Built-in
        ("os", None),  # Built-in
        ("logging", None),  # Built-in
        ("datetime", None),  # Built-in
        ("threading", None),  # Built-in
        ("queue", None),  # Built-in
        ("re", None),  # Built-in
        ("warnings", None),  # Built-in
        ("typing", None),  # Built-in
        ("pathlib", None),  # Built-in
        ("argparse", None),  # Built-in
        ("subprocess", None),  # Built-in
    ]
    
    for module, package in dependencies:
        if package is None:
            # Built-in module, should always work
            try:
                __import__(module)
                print(f"✅ {module} - OK (built-in)")
            except ImportError:
                print(f"❌ {module} - ERROR (built-in module missing!)")
                all_good = False
        else:
            if not test_import(module, package):
                all_good = False
    
    print("\n" + "=" * 60)
    
    if all_good:
        print("🎉 All dependencies are installed!")
        print("\nTesting the import of our main module...")
        
        try:
            from football_sound_detection import FootballAudioAnalyzer, AudioEvent
            print("✅ football_sound_detection module imports successfully!")
            print("✅ FootballAudioAnalyzer class found!")
            print("✅ AudioEvent class found!")
            
            # Test basic initialization
            analyzer = FootballAudioAnalyzer(model_size="tiny")
            print("✅ FootballAudioAnalyzer can be initialized!")
            
        except Exception as e:
            print(f"❌ Error importing football_sound_detection: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Some dependencies are missing. Please install them using:")
        print("pip install numpy librosa openai-whisper scipy")
        print("\nAlso make sure you have ffmpeg installed:")
        print("- Windows: Download from https://ffmpeg.org/download.html")
        print("- Or use: winget install ffmpeg")
    
    print("\n" + "=" * 60)
    print("🔧 Additional requirements:")
    print("- FFmpeg must be installed and available in PATH")
    print("- Enough disk space for audio processing")
    print("- Internet connection for first-time Whisper model download")

if __name__ == "__main__":
    main()