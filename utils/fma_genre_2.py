import numpy as np
import os
import warnings
from scipy import stats
import pandas as pd
import librosa
import joblib,pickle


def get_feature_columns():
    """Generate column names for audio features."""
    feature_sizes = dict(
        chroma_cqt=12,
        tonnetz=6,
        mfcc=20,
        rmse=1,
        zcr=1,
        spectral_centroid=1,
        spectral_contrast=7
    )
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)
    return columns.sort_values()

def extract_features(audio_path):
    """Extract audio features from a single audio file."""
    features = pd.Series(index=get_feature_columns(), dtype=np.float32)
    
    warnings.filterwarnings('error', module='librosa')

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1).astype(np.float32)
        features[name, 'std'] = np.std(values, axis=1).astype(np.float32)
        features[name, 'skew'] = stats.skew(values, axis=1).astype(np.float32)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1).astype(np.float32)
        features[name, 'median'] = np.median(values, axis=1).astype(np.float32)
        features[name, 'min'] = np.min(values, axis=1).astype(np.float32)
        features[name, 'max'] = np.max(values, axis=1).astype(np.float32)

    try:
        x, sr = librosa.load(audio_path, sr=None, mono=True)

        f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
        feature_stats('zcr', f)

        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                               n_bins=7*12, tuning=None))
        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cqt', f)
        f = librosa.feature.tonnetz(chroma=f)
        feature_stats('tonnetz', f)
        del cqt

        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        f = librosa.feature.rms(S=stft)
        feature_stats('rmse', f)
        f = librosa.feature.spectral_centroid(S=stft)
        feature_stats('spectral_centroid', f)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        feature_stats('spectral_contrast', f)

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feature_stats('mfcc', f)

    except Exception as e:
        print(f'Error processing {audio_path}: {repr(e)}')
        return None

    return features

class GenreClassifier:
    def __init__(self, model_path, scaler_path, pca_path):
        """
        Initialize the genre classifier with pre-trained model and transformers.
        
        Args:
            model_path: Path to saved model
            scaler_path: Path to saved StandardScaler
            pca_path: Path to saved PCA transformer
        """
        # Load the model using pickle
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)

        self.scaler = joblib.load(scaler_path)
        self.pca = joblib.load(pca_path)
        self.genres = ['Blues', 'Classical', 'Electronic', 'Experimental', 'Folk', 
                      'Hip-Hop', 'Instrumental', 'Jazz', 'Pop', 'Rock'] 
        
    def predict_file(self, audio_path):
        """
        Predict genre for a single audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary containing predicted genre and probabilities
        """
        # Extract features
        features = extract_features(audio_path)
        if features is None:
            return None
            
        # Convert to DataFrame with single row
        features_df = pd.DataFrame([features])
        
        # Flatten MultiIndex columns if they exist
        if isinstance(features_df.columns, pd.MultiIndex):
            features_df.columns = [f"{feat}_{num}_{stat}" for feat, stat, num in features_df.columns]
        
        # Transform features using saved scaler and PCA
        X_scaled = self.scaler.transform(features_df)
        X_pca = self.pca.transform(X_scaled)

        print(f'Shape of PCA and Model Input:{X_pca.shape}')
        
        # Make prediction
        probs = self.model.predict_proba(X_pca)
        pred_class = self.genres[np.argmax(probs)]
        
        # Create results dictionary
        results = {
            'predicted_genre': pred_class,
            'probabilities': dict(zip(self.genres, probs[0].tolist()))
        }
        top3 = sorted(results['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
        return results , top3

# Example usage:
if __name__ == "__main__":
    # Initialize the classifier with your saved model and transformers
    classifier = GenreClassifier(
        model_path='models/fma_models/xgb_model.pkl',
        scaler_path='models/fma_models/standard_scaler.joblib',
        pca_path='models/fma_models/pca_model.joblib'
    )
    
    # Single file prediction
    audio_file = "audio_files\A Beacon of Hope.mp3"
    result,top3 = classifier.predict_file(audio_file)
    if result:
        print(f"Predicted genre: {result['predicted_genre']}")
        print("\nProbabilities:")
        for genre, prob in result['probabilities'].items():
            print(f"{genre}: {prob:.3f}")
    
    print(f'Top 3 Genre Are:{top3}')
    # Or process a whole directory
    # audio_dir = "path/to/audio/directory"
    # results = classifier.predict_batch(audio_dir)