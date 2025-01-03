import json
import numpy as np
import warnings
from scipy import stats
import pandas as pd
import librosa
import joblib,pickle
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D,TensorflowPredictMusiCNN , TensorflowPredictVGGish

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


class AudioGenreClassifier:
    def __init__(self,
                 genre_model_path,
                 model_json_path,
                 essentia_genre_model_path,
                 essentia_genre_json_path,
                 embedding_model_path,
                 fma_model_path,
                 fma_scaler_path,
                 fma_pca_path):
        
        #For FMA-Classification.

        # Load the model using pickle
        with open(fma_model_path, 'rb') as model_file:
            self.fma_model = pickle.load(model_file)

        self.fma_scaler = joblib.load(fma_scaler_path)
        self.fma_pca = joblib.load(fma_pca_path)
        self.fma_genres = ['Blues', 'Classical', 'Electronic', 'Experimental', 'Folk', 
                      'Hip-Hop', 'Instrumental', 'Jazz', 'Pop', 'Rock'] 

        self.models = []
        self.models_names = []
        self.metadatas = []
        #Genre Models From MTG-Jamendro Dataset.
        self.genre_model_path = genre_model_path
        self.model_json_path = model_json_path

        #Genre Model Using Essentia AutoTagging.
        self.essentia_genre_model_path = essentia_genre_model_path
        self.essentia_genre_json_path = essentia_genre_json_path

        #Embedding Model for MTG.
        self.embedding_model_path = embedding_model_path
      
    def predict_fma_genre(self, audio_path):
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
        X_scaled = self.fma_scaler.transform(features_df)
        X_pca = self.fma_pca.transform(X_scaled)

        print(f'Shape of PCA and Model Input:{X_pca.shape}')
        
        # Make prediction
        probs = self.fma_model.predict_proba(X_pca)
        pred_class = self.fma_genres[np.argmax(probs)]
        
        # Create results dictionary
        results = {
            'predicted_genre': pred_class,
            'probabilities': dict(zip(self.fma_genres, probs[0].tolist()))
        }
        
        return results

    def load_genre_model(self):

        # Load Genre Prediction Model and its Metadata Path
        model_path = self.genre_model_path
        metadata_path = self.model_json_path

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Prepare model
        model = TensorflowPredict2D(graphFilename=model_path)

        return model,metadata

    def load_essentia_autotagging_model(self):

      with open(self.essentia_genre_json_path, 'r') as f:
          metadata = json.load(f)

      model = TensorflowPredictMusiCNN(graphFilename=self.essentia_genre_model_path)

      return model,metadata

    def load_embeddings(self,audio):

      embedding_model_path = self.embedding_model_path
      embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=embedding_model_path, output="PartitionedCall:1")
      embeddings = embedding_model(audio)

      return embeddings

    def get_top_3_predictions(self, predictions):
        """
        Get top 3 predictions from a prediction dictionary

        Args:
            predictions (dict): Predictions dictionary

        Returns:
            list: Top 3 predictions as [(label, probability)]
        """
        # Sort predictions by probability in descending order
        return sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]

    def predict(self, audio_path, sample_rate=16000):

        # Load audio
        loader = MonoLoader(sampleRate=sample_rate, filename=audio_path)
        audio = loader()

        #Load embeddings from Discog model.
        embeddings = self.load_embeddings(audio)

        #Load Genre Model and MetaData.
        model,metadata = self.load_genre_model()

        #Load Essentia Auto-Tagging Model and MetaData.
        essentia_model, essentia_metadata = self.load_essentia_autotagging_model()

        #Take Mean across-each timeStamp for mtg.
        mtg_predictions = model(embeddings)
        mtg_predictions = mtg_predictions.mean(axis=0)
        mtg_results = {}
        for label, probability in zip(metadata['classes'], mtg_predictions):
          mtg_results[label] = float(f'{100 * probability:.1f}')

        #Take Mean across-each timeStamp for essentia.
        essentia_predictions = essentia_model(audio)
        essentia_predictions = essentia_predictions.mean(axis=0)
        essentia_genre_results = {}
        for label, probability in zip(essentia_metadata['classes'], essentia_predictions):
          essentia_genre_results[label] = float(f'{100 * probability:.1f}')

        #Get Top3 Genres Of Music.
        mtg_results = self.get_top_3_predictions(mtg_results)
        essentia_genre_results = self.get_top_3_predictions(essentia_genre_results)

        #Get Top3 Genres for FMA Music.
        fma_results = self.predict_fma_genre(audio_path=audio_path)
        fma_top3_res = self.get_top_3_predictions(predictions=fma_results['probabilities'])

        return mtg_results , essentia_genre_results, fma_top3_res