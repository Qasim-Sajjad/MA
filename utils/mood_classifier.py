import os
import json
from essentia.standard import MonoLoader, TensorflowPredictMusiCNN, TensorflowPredict2D, TensorflowPredictEffnetDiscogs

class AudioMoodClassifier:
    def __init__(self,
                models_dir,
                metadatas_dir,
                mtg_moodtheme_model_path=None,
                mtg_moodtheme_metadata_path=None,
                embedding_model_path=None):
        """
        Initialize the MoodClassifier with mood models directory and metadatas dir

        Args:
            models_dir (str): Directory containing model .pb and .json files
            metadatas_dir (str): Directory containing metadata files
            mtg_moodtheme_model_path (str): Path to the MTG mood theme model
            mtg_moodtheme_metadata_path (str): Path to the MTG mood theme metadata
            embedding_model_path (str): Path to the embedding model
        """
        self.models_dir = models_dir
        self.metadatas_dir = metadatas_dir
        self.mtg_moodtheme_model_path = mtg_moodtheme_model_path
        self.mtg_moodtheme_metadata_path = mtg_moodtheme_metadata_path
        self.embedding_model_path = embedding_model_path
        self.models = []
        self.models_names = []
        self.metadatas = []
        self._load_models()

    def _load_models(self):
        """
        Load models and their respective json files.
        """
        # Find all .pb and .json files
        pb_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pb')]
        json_files = [f for f in os.listdir(self.metadatas_dir) if f.endswith('.json')]

        # Sort files to ensure matching .pb and .json files
        pb_files.sort()
        json_files.sort()

        print(pb_files)
        print(json_files)

        # Load metadata and models
        for pb_file, json_file in zip(pb_files, json_files):
            # Full paths
            pb_path = os.path.join(self.models_dir, pb_file)
            json_path = os.path.join(self.metadatas_dir, json_file)

            # Load metadata
            print(f'JSON PATH: {json_path}')
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            # Prepare model
            model = TensorflowPredictMusiCNN(graphFilename=pb_path)
            name = pb_file.replace('.pb','')

            # Store model and metadata and names
            self.models_names.append(name)
            self.models.append(model)
            self.metadatas.append(metadata)

    def load_mtg_moodtheme_model(self):
        """Load MTG mood theme model and its metadata"""
        if not self.mtg_moodtheme_model_path or not self.mtg_moodtheme_metadata_path:
            return None, None

        with open(self.mtg_moodtheme_metadata_path, 'r') as f:
            metadata = json.load(f)

        model = TensorflowPredict2D(graphFilename=self.mtg_moodtheme_model_path)
        return model, metadata

    def load_embeddings(self, audio):
        """Load embeddings from the audio using EffnetDiscogs model"""
        if not self.embedding_model_path:
            return None

        embedding_model = TensorflowPredictEffnetDiscogs(
            graphFilename=self.embedding_model_path, 
            output="PartitionedCall:1"
        )
        return embedding_model(audio)

    def predict(self, audio_path, sample_rate=16000):
        """
        Predict audio Mood Classification.

        Args:
            audio_path (str): Path to the audio file
            sample_rate (int): Sampling rate for audio loading

        Returns:
            dict: Classification probabilities for each model
        """
        # Load audio
        loader = MonoLoader(sampleRate=sample_rate, filename=audio_path)
        audio = loader()

        # Predictions from all models
        results = {}
        # Prediction from MTG Mood Theme Model
        mtg_results = {}

        # Get predictions from Essentia models
        for metadata, model, model_name in zip(self.metadatas, self.models, self.models_names):
            model_name = model_name.replace('-musicnn-msd-2', '')
            activations = model(audio)
            mean_activations = activations.mean(axis=0)

            for label, probability in zip(metadata['classes'], mean_activations):
                if not label.startswith('non') and not label.startswith('not'):
                    results[label] = float(f'{100 * probability:.1f}')

        # Get predictions from MTG mood theme model
        mtg_model, mtg_metadata = self.load_mtg_moodtheme_model()
        if mtg_model and mtg_metadata:
            embeddings = self.load_embeddings(audio)
            if embeddings is not None:
                mtg_predictions = mtg_model(embeddings)
                mtg_predictions = mtg_predictions.mean(axis=0)
                
                for label, probability in zip(mtg_metadata['classes'], mtg_predictions):
                    mtg_results[label] = float(f'{100 * probability:.1f}')

        # Get Top 3 Predictions
        results = self.get_top_3_predictions(results)
        mtg_results = self.get_top_3_predictions(mtg_results)
        return results, mtg_results

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

    def print_predictions(self, predictions):
        """
        Print predictions in a formatted manner.

        Args:
            predictions (dict): Prediction results from predict method
        """
        for model_name, classes in predictions.items():
            print(f"\n{model_name} Predictions:")
            for label, probability in classes.items():
                print(f'{label}: {probability}%')