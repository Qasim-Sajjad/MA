import json
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D

class AudioInstrumentClassifier:
    def __init__(self,instrument_model_path,model_json_path,embedding_model_path):

        self.models = []
        self.models_names = []
        self.metadatas = []
        self.instrument_model_path = instrument_model_path
        self.model_json_path = model_json_path
        self.embedding_model_path = embedding_model_path

    #Load Instrument Model and its JSON Metadata.
    def load_instrument_model(self):

        # Load Instrument Prediction Model and its Metadata Path
        model_path = self.instrument_model_path
        metadata_path = self.model_json_path

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Prepare model
        model = TensorflowPredict2D(graphFilename=model_path)

        return model,metadata

    #Load Embeddings of the audio.
    def load_embeddings(self,audio):

      embedding_model_path = self.embedding_model_path
      embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=embedding_model_path, output="PartitionedCall:1")
      embeddings = embedding_model(audio)

      return embeddings

    def predict(self, audio_path, sample_rate=16000):

        # Load audio
        loader = MonoLoader(sampleRate=sample_rate, filename=audio_path)
        audio = loader()

        #Load embeddings from Discog model.
        embeddings = self.load_embeddings(audio)

        #Load Instrument Model and MetaData.
        model,metadata = self.load_instrument_model()

        predictions = model(embeddings)

        #Take Mean across-each timeStamp
        predictions = predictions.mean(axis=0)
        results = {}

        for label, probability in zip(metadata['classes'], predictions):
            results[label] = float(f'{100 * probability:.1f}')

        #Get Top3 Instruments within Music.
        results = self.get_top_3_predictions(results)

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

      print(f"\nInstrument Model Predictions:")
      for label, probability in predictions.items():
          print(f'{label}: {probability}%')
