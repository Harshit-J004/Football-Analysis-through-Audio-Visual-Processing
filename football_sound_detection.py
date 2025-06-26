import os
import numpy as np
import librosa
import whisper
import json
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import threading
import queue
import logging
from dataclasses import dataclass
import re
from scipy.signal import find_peaks

# Suppress warnings
warnings.filterwarnings("ignore")

@dataclass
class AudioEvent:
    """Data class for detected audio events"""
    timestamp: float
    event_type: str
    confidence: float
    description: str
    audio_features: Dict
    duration: float = 0.0

class FootballAudioAnalyzer:
    """Main class for football sound event detection"""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the football audio analyzer
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        """
        self.setup_logging()
        self.logger.info("Initializing Football Audio Analyzer...")
        
        # Load Whisper model
        try:
            self.whisper_model = whisper.load_model(model_size)
            self.logger.info(f"Whisper model '{model_size}' loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise
        
        # Football-specific keywords
        self.keywords = {
            'goal': {
                'keywords': ['goal', 'scored', 'scores', 'brilliant', 'amazing', 'fantastic', 
                           'incredible', 'wonderful', 'magnificent', 'beautiful', 'stunning',
                           'nets', 'back of the net', 'finds the net', 'celebration'],
                'weight': 1.0,
                'min_confidence': 0.7
            },
            'card': {
                'keywords': ['yellow card', 'red card', 'booking', 'booked', 'referee', 
                           'card', 'caution', 'sent off', 'dismissed', 'penalty'],
                'weight': 0.8,
                'min_confidence': 0.6
            },
            'excitement': {
                'keywords': ['crowd', 'roar', 'cheering', 'wild', 'erupts', 'explodes',
                           'unbelievable', 'what a', 'oh my', 'sensational', 'dramatic'],
                'weight': 0.6,
                'min_confidence': 0.5
            },
            'whistle': {
                'keywords': ['whistle', 'blown', 'referee', 'stop', 'halt', 'end'],
                'weight': 0.7,
                'min_confidence': 0.6
            }
        }
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.window_size = 3.0  # seconds
        self.overlap = 0.5  # 50% overlap
        self.min_event_gap = 5.0  # minimum seconds between same event type
        
        # Event storage
        self.detected_events = []
        self.audio_timeline = []
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('football_audio_detection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_audio_from_video(self, video_path: str, output_path: str = None) -> str:
        """
        Extract audio from video file using ffmpeg
        
        Args:
            video_path: Path to input video file
            output_path: Path for output audio file
            
        Returns:
            Path to extracted audio file
        """
        if output_path is None:
            output_path = video_path.rsplit('.', 1)[0] + '_audio.wav'
        
        try:
            import subprocess
            cmd = [
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(self.sample_rate), '-ac', '1', '-y', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg error: {result.stderr}")
            
            self.logger.info(f"Audio extracted to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to extract audio: {e}")
            raise
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file using librosa
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            self.logger.info(f"Audio loaded: {len(audio)/sr:.2f} seconds")
            return audio, sr
        except Exception as e:
            self.logger.error(f"Failed to load audio: {e}")
            raise
    
    def extract_audio_features(self, audio_segment: np.ndarray) -> Dict:
        """
        Extract audio features from audio segment
        
        Args:
            audio_segment: Audio data segment
            
        Returns:
            Dictionary of audio features
        """
        try:
            # Basic features
            rms_energy = librosa.feature.rms(y=audio_segment)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_segment)[0]
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_segment, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=self.sample_rate)[0]
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_segment, sr=self.sample_rate, n_mfcc=13)
            
            # Tempo and rhythm
            tempo, _ = librosa.beat.beat_track(y=audio_segment, sr=self.sample_rate)
            
            features = {
                'rms_energy_mean': float(np.mean(rms_energy)),
                'rms_energy_std': float(np.std(rms_energy)),
                'zcr_mean': float(np.mean(zero_crossing_rate)),
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'mfcc_mean': [float(x) for x in np.mean(mfccs, axis=1)],
                'tempo': float(tempo),
                'energy_variance': float(np.var(rms_energy))
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Failed to extract audio features: {e}")
            return {}
    
    def detect_excitement_in_audio(self, audio_segment: np.ndarray, timestamp: float) -> Optional[AudioEvent]:
        """
        Detect excitement/energy spikes in audio segment
        
        Args:
            audio_segment: Audio data segment
            timestamp: Timestamp of the segment
            
        Returns:
            AudioEvent if excitement detected, None otherwise
        """
        try:
            # Calculate energy
            rms_energy = librosa.feature.rms(y=audio_segment)[0]
            energy_mean = np.mean(rms_energy)
            energy_std = np.std(rms_energy)
            
            # Detect energy spikes
            energy_threshold = energy_mean + 2 * energy_std
            peaks, _ = find_peaks(rms_energy, height=energy_threshold, distance=10)
            
            if len(peaks) > 0:
                # Calculate excitement confidence
                max_energy = np.max(rms_energy[peaks])
                confidence = min(max_energy / (energy_mean + 1e-8), 1.0)
                
                if confidence > 0.3:  # Minimum excitement threshold
                    features = self.extract_audio_features(audio_segment)
                    
                    return AudioEvent(
                        timestamp=timestamp,
                        event_type='audio_excitement',
                        confidence=confidence,
                        description=f'Audio excitement spike detected (confidence: {confidence:.2f})',
                        audio_features=features,
                        duration=len(audio_segment) / self.sample_rate
                    )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to detect excitement: {e}")
            return None
    
    def transcribe_and_analyze(self, audio_segment: np.ndarray, timestamp: float) -> List[AudioEvent]:
        """
        Transcribe audio segment and analyze for keywords
        
        Args:
            audio_segment: Audio data segment
            timestamp: Timestamp of the segment
            
        Returns:
            List of detected AudioEvents
        """
        events = []
        
        try:
            # Transcribe audio
            result = self.whisper_model.transcribe(
                audio_segment,
                fp16=False,
                language='en',
                task='transcribe'
            )
            
            text = result['text'].lower().strip()
            
            if not text:
                return events
            
            self.logger.debug(f"Transcribed at {timestamp:.2f}s: {text}")
            
            # Analyze keywords
            for event_type, config in self.keywords.items():
                keyword_matches = []
                total_confidence = 0.0
                
                for keyword in config['keywords']:
                    if keyword in text:
                        keyword_matches.append(keyword)
                        # Simple confidence based on keyword importance and frequency
                        keyword_count = text.count(keyword)
                        keyword_confidence = min(0.3 + (keyword_count * 0.2), 1.0)
                        total_confidence += keyword_confidence
                
                if keyword_matches and total_confidence > 0:
                    # Normalize confidence
                    confidence = min(total_confidence * config['weight'], 1.0)
                    
                    if confidence >= config['min_confidence']:
                        # Extract audio features
                        features = self.extract_audio_features(audio_segment)
                        
                        event = AudioEvent(
                            timestamp=timestamp,
                            event_type=event_type,
                            confidence=confidence,
                            description=f"Keywords detected: {', '.join(keyword_matches)}. Text: '{text[:100]}...'",
                            audio_features=features,
                            duration=len(audio_segment) / self.sample_rate
                        )
                        
                        events.append(event)
                        self.logger.info(f"Event detected: {event_type} at {timestamp:.2f}s (confidence: {confidence:.2f})")
            
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to transcribe audio segment: {e}")
            return events
    
    def filter_duplicate_events(self, events: List[AudioEvent]) -> List[AudioEvent]:
        """
        Filter out duplicate or overlapping events
        
        Args:
            events: List of detected events
            
        Returns:
            Filtered list of events
        """
        if not events:
            return events
        
        # Sort events by timestamp
        events.sort(key=lambda x: x.timestamp)
        
        filtered_events = []
        last_event_time = {}
        
        for event in events:
            event_type = event.event_type
            
            # Check if enough time has passed since last event of same type
            if event_type in last_event_time:
                time_diff = event.timestamp - last_event_time[event_type]
                if time_diff < self.min_event_gap:
                    # Keep the event with higher confidence
                    last_event_idx = None
                    for i, prev_event in enumerate(filtered_events):
                        if (prev_event.event_type == event_type and 
                            abs(prev_event.timestamp - last_event_time[event_type]) < 1.0):
                            last_event_idx = i
                            break
                    
                    if last_event_idx is not None and event.confidence > filtered_events[last_event_idx].confidence:
                        filtered_events[last_event_idx] = event
                        last_event_time[event_type] = event.timestamp
                    continue
            
            filtered_events.append(event)
            last_event_time[event_type] = event.timestamp
        
        return filtered_events
    
    def process_audio_file(self, audio_path: str) -> List[AudioEvent]:
        """
        Process entire audio file and detect events
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of detected events
        """
        self.logger.info(f"Processing audio file: {audio_path}")
        
        # Load audio
        audio, sr = self.load_audio(audio_path)
        total_duration = len(audio) / sr
        
        # Process audio in sliding windows
        window_samples = int(self.window_size * sr)
        hop_samples = int(window_samples * (1 - self.overlap))
        
        all_events = []
        
        for start_sample in range(0, len(audio) - window_samples + 1, hop_samples):
            end_sample = start_sample + window_samples
            audio_segment = audio[start_sample:end_sample]
            timestamp = start_sample / sr
            
            # Progress logging
            progress = (timestamp / total_duration) * 100
            if int(progress) % 10 == 0 and progress > 0:
                self.logger.info(f"Processing: {progress:.1f}% complete")
            
            # Detect excitement spikes
            excitement_event = self.detect_excitement_in_audio(audio_segment, timestamp)
            if excitement_event:
                all_events.append(excitement_event)
            
            # Transcribe and analyze keywords
            keyword_events = self.transcribe_and_analyze(audio_segment, timestamp)
            all_events.extend(keyword_events)
        
        # Filter duplicates and overlaps
        filtered_events = self.filter_duplicate_events(all_events)
        
        self.detected_events = filtered_events
        self.logger.info(f"Processing complete. Detected {len(filtered_events)} events")
        
        return filtered_events
    
    def get_highlight_clips(self, events: List[AudioEvent], buffer_before: float = 5.0, 
                          buffer_after: float = 5.0) -> List[Dict]:
        """
        Generate highlight clip information based on detected events
        
        Args:
            events: List of detected events
            buffer_before: Seconds to include before event
            buffer_after: Seconds to include after event
            
        Returns:
            List of clip information dictionaries
        """
        clips = []
        
        for event in events:
            clip_start = max(0, event.timestamp - buffer_before)
            clip_end = event.timestamp + event.duration + buffer_after
            
            clip_info = {
                'start_time': clip_start,
                'end_time': clip_end,
                'duration': clip_end - clip_start,
                'event_type': event.event_type,
                'confidence': event.confidence,
                'description': event.description,
                'timestamp': event.timestamp
            }
            
            clips.append(clip_info)
        
        return clips
    
    def export_results(self, output_path: str = 'football_events.json'):
        """
        Export detected events to JSON file
        
        Args:
            output_path: Path for output JSON file
        """
        try:
            export_data = {
                'metadata': {
                    'total_events': len(self.detected_events),
                    'processing_timestamp': datetime.now().isoformat(),
                    'model_info': 'Whisper + Custom Football Analysis'
                },
                'events': []
            }
            
            for event in self.detected_events:
                event_data = {
                    'timestamp': event.timestamp,
                    'event_type': event.event_type,
                    'confidence': event.confidence,
                    'description': event.description,
                    'duration': event.duration,
                    'audio_features': event.audio_features
                }
                export_data['events'].append(event_data)
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Results exported to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
    
    def generate_report(self) -> str:
        """
        Generate a summary report of detected events
        
        Returns:
            Formatted report string
        """
        if not self.detected_events:
            return "No events detected."
        
        # Count events by type
        event_counts = {}
        for event in self.detected_events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
        
        # Calculate average confidence by type
        event_confidences = {}
        for event_type in event_counts:
            confidences = [e.confidence for e in self.detected_events if e.event_type == event_type]
            event_confidences[event_type] = np.mean(confidences)
        
        report = f"""
=== FOOTBALL SOUND EVENT DETECTION REPORT ===

Total Events Detected: {len(self.detected_events)}

Event Summary:
"""
        
        for event_type, count in event_counts.items():
            avg_conf = event_confidences[event_type]
            report += f"  {event_type.upper()}: {count} events (avg confidence: {avg_conf:.2f})\n"
        
        report += "\nTop Events by Confidence:\n"
        
        # Sort by confidence and show top 10
        top_events = sorted(self.detected_events, key=lambda x: x.confidence, reverse=True)[:10]
        
        for i, event in enumerate(top_events, 1):
            minutes = int(event.timestamp // 60)
            seconds = int(event.timestamp % 60)
            report += f"  {i}. {event.event_type} at {minutes:02d}:{seconds:02d} (confidence: {event.confidence:.2f})\n"
        
        return report