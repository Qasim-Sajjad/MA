# SENTIMENT ANALYSIS
import os
import numpy as np
import librosa
import webrtcvad
from spleeter.separator import Separator
import whisper
from transformers import pipeline

class AudioSentimentAnalyzer:
    def __init__(self,
                 LyricsExtractor,
                 model_dir=None):
        """
        Initialize the AudioSentimentAnalyzer.

        Args:
            model_dir (str, optional): Directory to store temporary files
        """
        self.model_dir = model_dir or os.path.join(os.getcwd(), 'audio_analysis_temp')
        self.Lyrics_Analyzer = LyricsExtractor

        # Ensure temporary directory exists
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize models
        self._init_models()

    def _init_models(self):
        """
        Initialize required models.
        """
        # Vocal Separator
        self.separator = Separator('spleeter:2stems')

        # Whisper for transcription
        self.whisper_model = whisper.load_model("base")

        # Sentiment Analysis Pipeline
        self.sentiment_analyzer = pipeline("sentiment-analysis")

    def check_vocals_presence(self,audio_to_check_vocals,aggressiveness):
        """
        Check if vocals are present in the audio file.

        Returns:
            bool: True if vocals are present, False otherwise
        """

        print(f'Vocals Audio File Path:{audio_to_check_vocals}')
        #Load the Audio File in Librosa and check frequencies that whether it has lyrics or not.
        self.y ,self.sr = librosa.load(audio_to_check_vocals,sr=16000,mono=True)

        # Convert to 16-bit PCM
        audio = (self.y * 32767).astype(np.int16)

        # Initialize VAD
        vad = webrtcvad.Vad(aggressiveness)

        # Frame the audio (30ms frames - recommended for VAD)
        frame_duration = 30  # ms
        frame_size = int(self.sr * (frame_duration / 1000))

        # Count voice frames
        voice_frames = 0
        total_frames = 0

        for start in range(0, len(audio), frame_size):
            frame = audio[start:start+frame_size]

            # Ensure frame is the correct length
            if len(frame) == frame_size:
                total_frames += 1
                try:
                    if vad.is_speech(frame.tobytes(), self.sr):
                        voice_frames += 1
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue

        # Calculate vocal presence ratio
        vocal_ratio = voice_frames / total_frames if total_frames > 0 else 0

        return {
            'has_vocals': vocal_ratio > 0.2,
            'vocal_ratio': vocal_ratio,
            'total_frames': total_frames,
            'voice_frames': voice_frames
        }


    def separate_vocals(self,audio_path):
        """
        Separate vocals from the audio file.

        Returns:
            str: Path to extracted vocals file or None
        """
        try:
            # Separate to temporary directory
            output_dir = os.path.join(self.model_dir, 'separated')
            self.separator.separate_to_file(audio_path, output_dir)

            # Find vocals file
            vocals_path = os.path.join(output_dir, os.path.splitext(os.path.basename(audio_path))[0], 'vocals.wav')

            return vocals_path if os.path.exists(vocals_path) else None
        except Exception as e:
            print(f"Error separating vocals: {e}")
            return None

    def transcribe_vocals(self, vocals_path):
        """
        Transcribe vocals to text.

        Args:
            vocals_path (str): Path to vocals audio file

        Returns:
            str: Transcribed lyrics
        """
        try:
            result = self.whisper_model.transcribe(vocals_path)
            return result["text"]
        except Exception as e:
            print(f"Error transcribing vocals: {e}")
            return None

    def analyze_sentiment(self, text):
        """
        Analyze sentiment of given text.

        Args:
            text (str): Text to analyze

        Returns:
            dict: Sentiment analysis results
        """
        try:
            # If text is too short or empty, return None
            if not text or len(text.strip()) < 5:
                return None

            # Use sentiment analysis pipeline
            sentiment_result = self.sentiment_analyzer(text)[0]
            return sentiment_result
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return None

    def analyze(self,audio_path, moods):
        """
        Perform comprehensive audio analysis.

        Args:
            mood (str, optional): Predefined mood if no lyrics found

        Returns:
            dict: Comprehensive analysis results
        """

        # Separate vocals from music first, then check whther the serperated file actually has vocals in it.
        vocals_path = self.separate_vocals(audio_path)

        #Check for Vocals Presence.
        vocals_features = self.check_vocals_presence(vocals_path,aggressiveness=3)

        # Initialize result dictionary
        result = {}
        has_lyrics = False

        if vocals_features['has_vocals']:

          # Attempt lyric transcription
          lyrics = None
          if vocals_path:
              lyrics = self.transcribe_vocals(vocals_path)

              sentiment = self.Lyrics_Analyzer.analyze_sentiment(lyrics)
              result.update({
                  'lyrics': lyrics,
                  'sentiment': sentiment
              })

              has_lyrics = True
        else:
            # If no lyrics, use predefined mood
            mood_prompt = f"Describe the sentiment of a song with following {moods[0]},{moods[1]}, {moods[2]} moods."
            sentiment = self.analyze_sentiment(mood_prompt)
            result.update({
                'lyrics': "No Lyrics, Sentiment Based on Mood",
                'sentiment': f"The Sentiment of the Music is {sentiment['label']}"
            })

        return result, has_lyrics

    def cleanup(self):
        """
        Clean up temporary files and directories.
        """
        try:
            import shutil
            if os.path.exists(self.model_dir):
                shutil.rmtree(self.model_dir)
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Lyrics = LyricsExtractor()
# sentiment_anz = AudioSentimentAnalyzer(LyricsExtractor=Lyrics)
# sentiment_anz.analyze(audio_path='audio_files/A Beacon of Hope.mp3',moods=['happy','sad'])