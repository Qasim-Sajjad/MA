import essentia
import numpy as np
import essentia.standard as estd
from essentia.standard import MonoLoader, KeyExtractor
import librosa

class KeyBPMExtractor:
    def __init__(self):
        """
        Initialize the KeyBPMExtractor with a specific audio file.
        """
        self.audio = None
        self.sr = None

    def _load_audio(self,file_path):
        """
        Load audio using Essentia's MonoLoader and librosa.
        """
        try:
            self.file_path = file_path
            #Feature Extraction for tonal based Approaach of extraction of key/BPM from audio.
            self.features, self.features_frames = estd.MusicExtractor(lowlevelStats=['mean', 'stdev'],
                                                          rhythmStats=['mean', 'stdev'],
                                                          tonalStats=['mean', 'stdev'])(self.file_path)

            # Load with librosa to get sample rate for BPM Detection.
            self.y, self.sr = librosa.load(self.file_path, sr=None)

        except Exception as e:
            print(f"Error loading audio file {self.file_path}: {e}")
            raise

    def extract_key(self):
        """
        Extract the musical key of the audio file.

        Returns:
            str: Formatted key and scale (e.g., "C Major")
        """
        try:

            # key_extractor = KeyExtractor()
            # key, scale, _ = key_extractor(self.audio)

            keys,scales = [],[]

            #Get Key from tonal.key_edma.key for analysis.
            keys.append(self.features['tonal.key_edma.key'])
            scales.append(self.features['tonal.key_edma.scale'])

            #Get Key from tonal.key_krumhansl.key for analysis.
            keys.append(self.features['tonal.key_krumhansl.key'])
            scales.append(self.features['tonal.key_krumhansl.scale'])

            #Get Key from tonal.key_temperley.key for analysis.
            keys.append(self.features['tonal.key_temperley.key'])
            scales.append(self.features['tonal.key_temperley.scale'])

            #Get Key from librosa.
            chroma = librosa.feature.chroma_cqt(y=self.y, sr=self.sr)
            chroma_mean = np.mean(chroma,axis=1)
            lib_keys = [
                "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
            ]
            key_index = np.argmax(chroma_mean)
            predicted_key = lib_keys[key_index]

            detection_key_scales = [f"{key} {scale}" for key, scale in zip(keys, scales)]

            detection_key_scales.append(predicted_key)
            print(f'Detected Key Scales are: {detection_key_scales}')

            return detection_key_scales

        except Exception as e:
            print(f"Error extracting key for {self.file_path}: {e}")
            return "Key Not Found"

    def extract_bpm(self):
        """
        Extract the tempo (BPM) of the audio file.

        Returns:
            int: Rounded beats per minute
        """
        try:
            bpms = []
            #Extracting BPM using Essentia.
            bpm = self.features['rhythm.bpm']
            bpms.append(round(float(bpm)))

            #Extracting BPM using librosa.beat_track.
            tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
            bpms.append(round(float(tempo)))

            #Using librosa.beat.tempo for analysis now.
            tempo = librosa.beat.tempo(y=self.y, sr=self.sr)[0]

            bpms.append(round(float(tempo)))

            return bpms
        except Exception as e:
            print(f"Error extracting BPM for {self.file_path}: {e}")
            return 0

    def analyze(self,file_path):
        """
        Perform complete audio analysis.

        Returns:
            dict: Dictionary containing analysis results
        """
        self._load_audio(file_path)

        results = {
            'Key': self.extract_key(),
            'BPM' : self.extract_bpm()
        }

        return results
