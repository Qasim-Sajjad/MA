import json
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D,TensorflowPredictMusiCNN , TensorflowPredictVGGish


class AudioGenreClassifier:
    def __init__(self,
                 genre_model_path,
                 model_json_path,
                 essentia_genre_model_path,
                 essentia_genre_json_path,
                 embedding_model_path):
        
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

        return mtg_results, essentia_genre_results