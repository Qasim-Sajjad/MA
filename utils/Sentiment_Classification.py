# SENTIMENT ANALYSIS
import os,time
import numpy as np
import librosa
import webrtcvad
import requests,re
from collections import Counter
from typing import List, Dict
from pydub import AudioSegment
from dotenv import load_dotenv
from spleeter.separator import Separator
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification


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
        self.lyrics_seg_dir = os.path.join(os.getcwd(), 'lyrics_analysis_temp')
        self.Lyrics_Analyzer = LyricsExtractor
        self.MAX_SEGMENT_DURATION = 25000  # 25 seconds in milliseconds
        self.MAX_RETRIES = 3
        self.RETRY_DELAY = 35  # seconds

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

        #Using HuggingFace.
        # API URL and headers
        load_dotenv("utils\.env")
        token = os.getenv(key="hf_whisper")
        base_tk = os.getenv(key="hf_whisper_base_tk")
        self.API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
        self.BACKUP_API_URL = "https://api-inference.huggingface.co/models/openai/whisper-base"
        self.headers = {"Authorization": f"Bearer {token}"}
        self.BACKUP_headers = {"Authorization": f"Bearer {base_tk}"}

        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        
        # Create pipeline with specified model and tokenizer
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            padding=True,
            max_length=512
        )

    def split_audio(self, audio_path: str) -> List[str]:
        """
        Split audio file into 30-second segments.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            List[str]: List of paths to the segmented audio files
        """
        audio = AudioSegment.from_file(audio_path)
        duration = len(audio)
        segment_paths = []
        
        for i in range(0, duration, self.MAX_SEGMENT_DURATION):
            segment = audio[i:i + self.MAX_SEGMENT_DURATION]
            segment_path = os.path.join(
                self.lyrics_seg_dir, 
                f"segment_{i//self.MAX_SEGMENT_DURATION}.wav"
            )
            segment.export(segment_path, format="wav")
            segment_paths.append(segment_path)
            
        return segment_paths

    def query_with_retry(self, filename: str) -> Dict:
        """
        Query the API with retry logic.
        
        Args:
            filename (str): Path to the audio file
            
        Returns:
            Dict: API response
            
        Raises:
            Exception: If all retries fail
        """
        # Try with whisper-large-v3 first
        for attempt in range(self.MAX_RETRIES):
            try:
                with open(filename, "rb") as f:
                    data = f.read()
                response = requests.post(
                    self.API_URL, 
                    headers=self.headers, 
                    data=data,
                    timeout=(180, 180)  # (connect timeout, read timeout)
                )
                
                if response.status_code == 503:
                    print(f"Service unavailable (attempt {attempt + 1}/{self.MAX_RETRIES})")
                    if attempt < self.MAX_RETRIES - 1:
                        time.sleep(self.RETRY_DELAY * (attempt + 1))  # Exponential backoff
                        continue
                        
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                print(f"Error during attempt {attempt + 1} with whisper-large: {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                    continue
                
                # If all retries failed, try whisper-base
                print("Falling back to whisper-base model...")
                try:
                    with open(filename, "rb") as f:
                        data = f.read()
                    response = requests.post(
                        self.BACKUP_API_URL,
                        headers=self.BACKUP_headers,
                        data=data,
                        timeout=(180, 180)
                    )
                    response.raise_for_status()
                    return response.json()
                    
                except requests.exceptions.RequestException as backup_error:
                    raise Exception(f"Failed with both models. Last error: {str(backup_error)}")

    def cleanse_lyrics(self, lyrics: str) -> str:
        """
        Clean lyrics by removing excessive repetitions and noise.
        
        Args:
            lyrics (str): Raw lyrics text
        
        Returns:
            str: Cleaned lyrics
        """
        # Step 1: Tokenize the lyrics into words
        words = re.findall(r'\b\w+\b', lyrics.lower())
        
        # Step 2: Count word frequencies
        word_counts = Counter(words)
        
        # Step 3: Filter out words that repeat more than 5 times
        filtered_words = [word for word in words if word_counts[word] <= 5]
        
        # Step 4: Join the filtered words back into a string
        cleansed_lyrics = " ".join(filtered_words)
        
        # Step 5: Handle edge cases where no meaningful lyrics remain
        if not cleansed_lyrics.strip():
            return "No meaningful lyrics."
        
        return cleansed_lyrics

    def query(self, filename: str) -> str:
        """
        Process audio file by splitting into segments and combining transcriptions.
        
        Args:
            filename (str): Path to the audio file
            
        Returns:
            str: Combined transcription from all segments
        """
        # Split audio into segments
        segment_paths = self.split_audio(filename)
        print(f'\n\nSegment Paths: {segment_paths}\n\n')
        transcriptions = []
        
        # Process each segment
        for segment_path in segment_paths:
            try:
                response = self.query_with_retry(segment_path)
                if 'text' in response:
                    # Clean the segment's lyrics before adding
                    cleaned_segment = self.cleanse_lyrics(response['text'])
                    if cleaned_segment != "No meaningful lyrics.":
                        transcriptions.append(cleaned_segment)
            except Exception as e:
                print(f"Error processing segment {segment_path}: {str(e)}")
                continue
            finally:
                # Clean up segment file
                if os.path.exists(segment_path):
                    os.remove(segment_path)
        
        # Combine all transcriptions
        return ' '.join(transcriptions)
    
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

    # Convert wav to mp3
    def convert_wav_to_mp3(self,wav_filename, mp3_filename):
        audio = AudioSegment.from_wav(wav_filename)  # Load the wav file
        audio.export(mp3_filename, format="mp3")  # Export as mp3

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
            self.convert_wav_to_mp3(vocals_path,mp3_filename=os.path.join(output_dir, os.path.splitext(os.path.basename(audio_path))[0], 'vocals.mp3'))

            vocals_path = os.path.join(output_dir, os.path.splitext(os.path.basename(audio_path))[0], 'vocals.mp3')
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
            output = self.query(filename=vocals_path)
            return output
        except Exception as e:
            print(f"Error transcribing vocals: {e}")
            return "No Lyrics"

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
              print(f'Lyrics Before Analysis:{lyrics}')

              analysis = self.Lyrics_Analyzer.analyze_sentiment(lyrics)

              if analysis['valid'] == True:
                print(f'Lyrics Analysis was Valid: {analysis["validation_details"]}')
                result.update({
                    'lyrics': lyrics,
                    'sentiment': analysis['sentiment']
                })
                has_lyrics = True

              else:
                print(f'Lyrics Analysis was Invalid: {analysis["reason"]}')

                mood_prompt = f"Describe the sentiment of a song with following moods: {', '.join(moods)}."
                sentiment = self.analyze_sentiment(mood_prompt)              
                result.update({
                    'lyrics': "No Lyrics",
                    'sentiment': f"The Sentiment of the Music is {sentiment['label']}"
                })
        else:
          # If no lyrics, use predefined mood
          mood_prompt = f"Describe the sentiment of a song with following moods: {', '.join(moods)}."
          sentiment = self.analyze_sentiment(mood_prompt)
          result.update({
              'lyrics': "No Lyrics",
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

