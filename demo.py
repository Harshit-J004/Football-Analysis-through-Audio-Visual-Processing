#!/usr/bin/env python3
"""
Demo script for Football Sound Event Detection System
This script demonstrates how to use the system with sample videos
"""

import os
import argparse
import sys
from pathlib import Path

try:
    from football_sound_detection import FootballAudioAnalyzer
    # Import AudioEvent separately since it's defined in the same module
    from football_sound_detection import AudioEvent
except ImportError:
    print("Error: Could not import FootballAudioAnalyzer or AudioEvent")
    print("Make sure you have installed all dependencies and the main script is available")
    sys.exit(1)

def process_single_video(video_path: str, output_dir: str = "output"):
    """
    Process a single video file for football event detection
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save output files
    """
    print(f"🏈 Processing video: {video_path}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize analyzer
    print("🔄 Initializing analyzer...")
    analyzer = FootballAudioAnalyzer(model_size="base")
    
    try:
        # Extract audio
        print("🎵 Extracting audio from video...")
        audio_path = analyzer.extract_audio_from_video(
            video_path, 
            os.path.join(output_dir, "extracted_audio.wav")
        )
        
        # Process audio
        print("🔍 Analyzing audio for football events...")
        events = analyzer.process_audio_file(audio_path)
        
        if not events:
            print("❌ No events detected in the video")
            return
        
        print(f"✅ Detected {len(events)} events!")
        
        # Generate highlight clips information
        print("📝 Generating highlight clips...")
        clips = analyzer.get_highlight_clips(events, buffer_before=3.0, buffer_after=7.0)
        
        # Export results
        results_path = os.path.join(output_dir, "detected_events.json")
        analyzer.export_results(results_path)
        
        # Generate and save report
        report = analyzer.generate_report()
        report_path = os.path.join(output_dir, "analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        print("\n" + "="*60)
        print(report)
        print("="*60)
        
        # Show highlight clips
        print("\n🎬 HIGHLIGHT CLIPS DETECTED:")
        print("-" * 40)
        
        for i, clip in enumerate(clips[:10], 1):  # Show top 10 clips
            minutes_start = int(clip['start_time'] // 60)
            seconds_start = int(clip['start_time'] % 60)
            minutes_end = int(clip['end_time'] // 60)
            seconds_end = int(clip['end_time'] % 60)
            
            print(f"Clip {i}: {clip['event_type'].upper()}")
            print(f"  ⏰ Time: {minutes_start:02d}:{seconds_start:02d} - {minutes_end:02d}:{seconds_end:02d}")
            print(f"  📏 Duration: {clip['duration']:.1f} seconds")
            print(f"  🎯 Confidence: {clip['confidence']:.2f}")
            print(f"  📋 Description: {clip['description'][:80]}...")
            print()
        
        # Save clip information
        clips_path = os.path.join(output_dir, "highlight_clips.txt")
        with open(clips_path, 'w') as f:
            f.write("=== HIGHLIGHT CLIPS ===\n\n")
            for i, clip in enumerate(clips, 1):
                f.write(f"Clip {i}: {clip['event_type'].upper()}\n")
                f.write(f"Start: {clip['start_time']:.2f}s, End: {clip['end_time']:.2f}s\n")
                f.write(f"Duration: {clip['duration']:.2f}s\n")
                f.write(f"Confidence: {clip['confidence']:.2f}\n")
                f.write(f"Description: {clip['description']}\n")
                f.write("-" * 50 + "\n")
        
        print(f"\n💾 Results saved to:")
        print(f"  📊 Analysis report: {report_path}")
        print(f"  📄 Event details: {results_path}")
        print(f"  🎬 Clip information: {clips_path}")
        print(f"  🎵 Extracted audio: {audio_path}")
        
        # Generate FFmpeg commands for clip extraction
        generate_ffmpeg_commands(video_path, clips, output_dir)
        
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        import traceback
        traceback.print_exc()

def generate_ffmpeg_commands(video_path: str, clips: list, output_dir: str):
    """
    Generate FFmpeg commands to extract video clips
    
    Args:
        video_path: Original video file path
        clips: List of clip information
        output_dir: Output directory
    """
    commands_file = os.path.join(output_dir, "extract_clips.sh")
    
    with open(commands_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# FFmpeg commands to extract highlight clips\n\n")
        
        for i, clip in enumerate(clips, 1):
            start_time = clip['start_time']
            duration = clip['duration']
            event_type = clip['event_type']
            
            output_filename = f"clip_{i:02d}_{event_type}_{start_time:.0f}s.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            cmd = f'ffmpeg -i "{video_path}" -ss {start_time:.2f} -t {duration:.2f} -c copy "{output_path}"\n'
            f.write(cmd)
        
        f.write("\necho 'All clips extracted!'\n")
    
    # Make script executable on Unix systems
    try:
        os.chmod(commands_file, 0o755)
    except:
        pass
    
    print(f"  🔧 FFmpeg commands: {commands_file}")
    print("     Run this script to extract actual video clips!")

def batch_process_videos(video_dir: str, output_base_dir: str = "batch_output"):
    """
    Process multiple videos in a directory
    
    Args:
        video_dir: Directory containing video files
        output_base_dir: Base directory for output files
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'}
    video_files = []
    
    for file_path in Path(video_dir).iterdir():
        if file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    if not video_files:
        print(f"❌ No video files found in {video_dir}")
        return
    
    print(f"🎥 Found {len(video_files)} video files to process")
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing video {i}/{len(video_files)}: {video_path.name}")
        print('='*60)
        
        # Create individual output directory
        video_output_dir = os.path.join(output_base_dir, video_path.stem)
        
        try:
            process_single_video(str(video_path), video_output_dir)
        except Exception as e:
            print(f"❌ Failed to process {video_path.name}: {e}")
            continue
    
    print(f"\n✅ Batch processing complete! Results saved in: {output_base_dir}")

def test_with_sample_audio():
    """
    Test the system with generated sample audio (for testing without video files)
    """
    print("🧪 Testing with synthetic audio...")
    
    try:
        import numpy as np
        import librosa
        
        # Generate sample audio with speech-like characteristics
        duration = 30  # seconds
        sample_rate = 16000
        t = np.linspace(0, duration, duration * sample_rate)
        
        # Create base audio with some noise
        audio = np.random.normal(0, 0.1, len(t))
        
        # Add some "excitement" spikes at specific times
        spike_times = [5, 12, 20, 25]  # seconds
        for spike_time in spike_times:
            spike_start = int(spike_time * sample_rate)
            spike_end = int((spike_time + 2) * sample_rate)
            if spike_end < len(audio):
                audio[spike_start:spike_end] += np.random.normal(0, 0.5, spike_end - spike_start)
        
        # Save sample audio
        sample_path = "test_sample_audio.wav"
        import soundfile as sf
        sf.write(sample_path, audio, sample_rate)
        
        # Process with our analyzer
        analyzer = FootballAudioAnalyzer(model_size="tiny")  # Use smaller model for testing
        events = analyzer.process_audio_file(sample_path)
        
        print(f"✅ Test completed! Detected {len(events)} events in synthetic audio")
        
        # Clean up
        if os.path.exists(sample_path):
            os.remove(sample_path)
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="Football Sound Event Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py video.mp4                    # Process single video
  python demo.py --batch ./videos/            # Process all videos in directory
  python demo.py --test                       # Run test with synthetic audio
  python demo.py video.mp4 --output ./results # Custom output directory
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input video file or directory')
    parser.add_argument('--output', '-o', default='output', 
                       help='Output directory (default: output)')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Process all videos in input directory')
    parser.add_argument('--test', '-t', action='store_true',
                       help='Run test with synthetic audio')
    
    args = parser.parse_args()
    
    print("🏈 Football Sound Event Detection System")
    print("=" * 50)
    
    if args.test:
        test_with_sample_audio()
        return
    
    if not args.input:
        print("❌ Error: No input provided")
        parser.print_help()
        return
    
    if not os.path.exists(args.input):
        print(f"❌ Error: Input path '{args.input}' does not exist")
        return
    
    if args.batch:
        if not os.path.isdir(args.input):
            print(f"❌ Error: '{args.input}' is not a directory")
            return
        batch_process_videos(args.input, args.output)
    else:
        if os.path.isdir(args.input):
            print(f"❌ Error: '{args.input}' is a directory. Use --batch flag for batch processing")
            return
        process_single_video(args.input, args.output)

if __name__ == "__main__":
    main()


# Additional utility functions for advanced usage

class FootballEventAnalyzer:
    """Extended analyzer with additional football-specific features"""
    
    def __init__(self, base_analyzer: FootballAudioAnalyzer):
        self.analyzer = base_analyzer
    
    def analyze_match_flow(self, events: list) -> dict:
        """
        Analyze the flow of the match based on detected events
        
        Args:
            events: List of detected events
            
        Returns:
            Dictionary with match flow analysis
        """
        if not events:
            return {}
        
        # Sort events by timestamp
        events.sort(key=lambda x: x.timestamp)
        
        # Divide match into periods (assuming 90 minute match)
        total_time = events[-1].timestamp if events else 5400  # 90 minutes
        first_half = [e for e in events if e.timestamp <= total_time * 0.5]
        second_half = [e for e in events if e.timestamp > total_time * 0.5]
        
        # Count event types by half
        def count_events(event_list):
            counts = {}
            for event in event_list:
                counts[event.event_type] = counts.get(event.event_type, 0) + 1
            return counts
        
        analysis = {
            'total_events': len(events),
            'match_duration': total_time / 60,  # in minutes
            'first_half': {
                'events': len(first_half),
                'event_types': count_events(first_half)
            },
            'second_half': {
                'events': len(second_half),
                'event_types': count_events(second_half)
            },
            'most_active_period': self._find_most_active_period(events),
            'event_distribution': count_events(events)
        }
        
        return analysis
    
    def _find_most_active_period(self, events: list, window_minutes: int = 10) -> dict:
        """Find the most active period in the match"""
        if not events:
            return {}
        
        window_seconds = window_minutes * 60
        max_events = 0
        best_period = None
        
        # Slide window through the match
        start_time = 0
        end_time = events[-1].timestamp
        
        for window_start in range(0, int(end_time), 60):  # Check every minute
            window_end = window_start + window_seconds
            window_events = [e for e in events 
                           if window_start <= e.timestamp <= window_end]
            
            if len(window_events) > max_events:
                max_events = len(window_events)
                best_period = {
                    'start_time': window_start,
                    'end_time': window_end,
                    'event_count': max_events,
                    'events': window_events
                }
        
        return best_period or {}
    
    def generate_highlight_reel_order(self, events: list) -> list:
        """
        Order events for optimal highlight reel creation
        
        Args:
            events: List of detected events
            
        Returns:
            Ordered list of events for highlight reel
        """
        # Priority weights for different event types
        priority_weights = {
            'goal': 1.0,
            'card': 0.7,
            'excitement': 0.5,
            'whistle': 0.3,
            'audio_excitement': 0.4
        }
        
        # Calculate priority scores
        scored_events = []
        for event in events:
            base_weight = priority_weights.get(event.event_type, 0.3)
            priority_score = event.confidence * base_weight
            
            scored_events.append({
                'event': event,
                'priority_score': priority_score
            })
        
        # Sort by priority score (descending)
        scored_events.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return [item['event'] for item in scored_events]


# Example usage with the extended analyzer
def advanced_analysis_example(video_path: str):
    """Example of advanced match analysis"""
    
    # Basic analysis
    analyzer = FootballAudioAnalyzer(model_size="base")
    audio_path = analyzer.extract_audio_from_video(video_path)
    events = analyzer.process_audio_file(audio_path)
    
    # Extended analysis
    extended_analyzer = FootballEventAnalyzer(analyzer)
    
    # Match flow analysis
    flow_analysis = extended_analyzer.analyze_match_flow(events)
    print("Match Flow Analysis:")
    print(f"  Total Duration: {flow_analysis.get('match_duration', 0):.1f} minutes")
    print(f"  First Half Events: {flow_analysis.get('first_half', {}).get('events', 0)}")
    print(f"  Second Half Events: {flow_analysis.get('second_half', {}).get('events', 0)}")
    
    # Optimal highlight order
    ordered_events = extended_analyzer.generate_highlight_reel_order(events)
    print(f"\nTop 5 Highlights (by priority):")
    for i, event in enumerate(ordered_events[:5], 1):
        print(f"  {i}. {event.event_type} at {event.timestamp:.1f}s (conf: {event.confidence:.2f})")
