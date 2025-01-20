import os
import json
from essentia.standard import MonoLoader, TensorflowPredictMusiCNN

class AudioMoodClassifier:
    def __init__(self, models_dir,metadatas_dir):
        """
        Initialize the MoodClassifier with mood models directory and metadatas dir

        Args:
            models_dir (str): Directory containing model .pb and .json files
        """
        self.models_dir = models_dir
        self.metadatas_dir = metadatas_dir
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

        # Predictions from all models.
        results = {}

        for metadata, model, model_name in zip(self.metadatas, self.models,self.models_names):
            # Get model name from the metadata filename
            model_name = model_name.replace('-musicnn-msd-2', '')

            # Compute model activations and take mean across time
            activations = model(audio)
            mean_activations = activations.mean(axis=0)

            for label, probability in zip(metadata['classes'], mean_activations):
              if not label.startswith('non') and not label.startswith('not'):
                results[label] = float(f'{100 * probability:.1f}')

        #Get Top 3 Predictions.
        results = self.get_top_3_predictions(results)

        #Return Top 3 Moods Of the Song.
        return results

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