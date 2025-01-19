import librosa
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from enum import Enum

class TransitionType(Enum):
    BUILDING_UP = "building_up"
    CALMING_DOWN = "calming_down"
    TENSION_CHANGE = "tension_change"
    NEUTRAL = "neutral"

def smooth_features(feature_array, window_length=11, polyorder=3):
    """
    Apply Savitzky-Golay filtering to smooth feature arrays and reduce noise.
    """
    return savgol_filter(feature_array, window_length, polyorder)

def classify_transition(before_segment, after_segment, sr, hop_length):
    """
    Classify the type of emotional transition based on musical features.
    
    Parameters:
        before_segment: Audio segment before breakpoint
        after_segment: Audio segment after breakpoint
        sr: Sample rate
        hop_length: Hop length for feature extraction
    
    Returns:
        TransitionType: Type of emotional transition
    """
    # Calculate features for both segments
    def get_segment_features(segment):
        rms = np.mean(librosa.feature.rms(y=segment, hop_length=hop_length))
        spectral = np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr, hop_length=hop_length))
        return rms, spectral

    before_rms, before_spectral = get_segment_features(before_segment)
    after_rms, after_spectral = get_segment_features(after_segment)
    
    # Calculate relative changes
    rms_change = (after_rms - before_rms) / (before_rms + 1e-8)
    spectral_change = (after_spectral - before_spectral) / (before_spectral + 1e-8)
    
    # Classification logic
    if rms_change > 0.2 and spectral_change > 0.1:
        return TransitionType.BUILDING_UP
    elif rms_change < -0.2 and spectral_change < -0.1:
        return TransitionType.CALMING_DOWN
    elif abs(spectral_change) > 0.2:
        return TransitionType.TENSION_CHANGE
    else:
        return TransitionType.NEUTRAL

def detect_emotional_breakpoints(audio_path, threshold=0.7, min_distance_seconds=1.0, 
                               window_size_seconds=3.0, smooth_window=11):
    """
    Detect and classify emotional transition points in classical/soundtrack music.
    
    Parameters:
        audio_path (str): Path to the audio file
        threshold (float): Sensitivity of breakpoint detection (0.0 to 1.0)
        min_distance_seconds (float): Minimum time between breakpoints
        window_size_seconds (float): Size of analysis window for transition classification
        smooth_window (int): Window size for feature smoothing
    
    Returns:
        List[dict]: List of dictionaries containing breakpoint information
                   (timestamp and transition type)
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    hop_length = 512
    min_distance = int(min_distance_seconds * sr / hop_length)
    window_samples = int(window_size_seconds * sr)
    
    # 1. Extract and smooth features
    # Harmonic features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    chroma_diff = np.mean(np.abs(np.diff(chroma, axis=1)), axis=0)
    chroma_diff_smooth = smooth_features(chroma_diff, smooth_window)
    
    # Dynamic features
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_diff = np.abs(np.diff(rms))
    rms_diff_smooth = smooth_features(rms_diff, smooth_window)
    
    # Spectral features
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    contrast_diff = np.mean(np.abs(np.diff(contrast, axis=1)), axis=0)
    contrast_diff_smooth = smooth_features(contrast_diff, smooth_window)
    
    # 2. Combine smoothed features
    def normalize(x): 
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
    
    combined_diff = (
        0.5 * normalize(chroma_diff_smooth) +
        0.3 * normalize(rms_diff_smooth) +
        0.2 * normalize(contrast_diff_smooth)
    )
    
    # 3. Find breakpoints
    peaks, _ = find_peaks(
        combined_diff,
        height=threshold,
        distance=min_distance
    )
    
    # 4. Analyze and classify each breakpoint
    breakpoints = []
    for peak in peaks:
        # Convert frame index to time
        time_idx = librosa.frames_to_time(peak, sr=sr, hop_length=hop_length)
        sample_idx = int(time_idx * sr)
        
        # Get audio segments before and after breakpoint
        before_segment = y[max(0, sample_idx - window_samples):sample_idx]
        after_segment = y[sample_idx:min(len(y), sample_idx + window_samples)]
        
        # Classify transition
        transition_type = classify_transition(before_segment, after_segment, sr, hop_length)
        
        breakpoints.append({
            'timestamp': time_idx,
            'type': transition_type.value
        })
    
    return breakpoints

# # Example usage
# if __name__ == "__main__":
#     audio_file = "/content/3.2.mp3"
#     breakpoints = detect_emotional_breakpoints(
#         audio_file,
#         threshold=0.7,
#         min_distance_seconds=1.0,
#         window_size_seconds=3.0,
#         smooth_window=11
#     )
#     print(breakpoints)
#     # Print detected breakpoints and their types
#     for bp in breakpoints:
#         print(f"Breakpoint at {bp['timestamp']:.2f}s - Type: {bp['type']}")