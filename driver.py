# Function to Store results in CSV.
import csv
import os
import pandas as pd
import librosa
from pydub import AudioSegment
import tempfile
from utils.emotional_breakpoints import detect_emotional_breakpoints
from utils.genre_classifier import AudioGenreClassifier
from utils.instrument_classifier import AudioInstrumentClassifier
from utils.key_bpm_extractor import KeyBPMExtractor
from utils.mood_classifier import AudioMoodClassifier
from utils.LyricsAnalyzer import LyricsExtractor
from utils.Sentiment_Classification import AudioSentimentAnalyzer

#Splt Audio Into Segments.
def split_audio_at_breakpoints(file_path, breakpoints):
    """
    Split audio file at detected breakpoints and return paths to temporary files.
    
    Args:
        file_path: Path to the audio file
        breakpoints: List of dictionaries containing timestamps and transition types
                    e.g. [{'timestamp': 59.74, 'type': 'neutral'}, ...]
    
    Returns:
        List of dictionaries containing segment information
    """
    # Load audio using pydub
    audio = AudioSegment.from_file(file_path)
    total_duration = len(audio) / 1000  # Convert to seconds
    
    # Create temporary directory for segments
    temp_dir = tempfile.mkdtemp()
    segments = []
    
    # Create list of all split points including start and end
    split_points = [0] + [bp['timestamp'] for bp in breakpoints] + [total_duration]
    transition_types = ['start'] + [bp['type'] for bp in breakpoints] + ['end']
    
    # Split audio at each breakpoint
    for i in range(len(split_points) - 1):
        start_time = split_points[i]
        end_time = split_points[i + 1]
        
        # Convert to milliseconds for pydub
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)
        
        # Extract segment
        segment = audio[start_ms:end_ms]
        
        # Save to temporary file
        temp_path = os.path.join(temp_dir, f"segment_{i}.wav")
        segment.export(temp_path, format="wav")
        
        # Store segment info with exact timestamps and transition type
        segments.append({
            'path': temp_path,
            'start_time': start_time,  # Original timestamp in seconds
            'end_time': end_time,      # Original timestamp in seconds
            'transition_type': transition_types[i]
        })
        
        print(f"Created segment {i}: {start_time:.2f}s to {end_time:.2f}s - {transition_types[i]}")
    
    return segments

def process_directory(audio_dir,
                      output_path,
                      genre_classifier,
                      mood_classifier,
                      Instrument_classifier,
                      KeyBPM,
                      Sentiment_analyzer):
    """
    Process audio files, skipping those already processed
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try reading with different encodings
    encodings_to_try = ['latin-1', 'iso-8859-1', 'cp1252']
    processed_files = set()

    for encoding in encodings_to_try:
        try:
            existing_df = pd.read_csv(output_path, encoding=encoding)
            print(f"Successfully read file with {encoding} encoding")
            processed_files = set(existing_df['Filename'])
            break
        except Exception as e:
            print(f"Failed to read with {encoding} encoding: {e}")

    # If CSV doesn't exist, create it with headers
    if not os.path.exists(output_path):
        with open(output_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            header = ['Filename', 
                      'Segment_Number',
                      'Segment_Start_Time',
                      'Segment_End_Time',
                      'Transition_Type'] + [
                      'MTG-Jamendo-Genre1', 'Genre1_Prob',
                      'MTG-Jamendo-Genre2', 'Genre2_Prob',
                      'MTG-Jamendo-Genre3', 'Genre3_Prob',
                      'FMA-Genre1', 'Genre1_Prob',
                      'FMA-Genre2', 'Genre2_Prob',
                      'FMA-Genre3', 'Genre3_Prob',
                      'Essentia-Autotagging-Genre1', 'Genre1_Prob',
                      'Essentia-Autotagging-Genre2', 'Genre2_Prob',
                      'Essentia-Autotagging-Genre3', 'Genre3_Prob',
                      'Essentia-Mood1', 'Mood1_Prob',
                      'Essentia-Mood2', 'Mood2_Prob',
                      'Essentia-Mood3', 'Mood3_Prob',
                      'MTG-Jamendo-Instrument1', 'Instrument1_Prob',
                      'MTG-Jamendo-Instrument2', 'Instrument2_Prob',
                      'MTG-Jamend0-Instrument3', 'Instrument3_Prob',
                      'essentia.Key_edma',
                      'essentia.key_krumhansl',
                      'essentia.key_temperley',
                      'librosa.key',
                      'essentia.rhythm.bpm',
                      'librosa.beat_track (BPM)',
                      'librosa.beat.tempo (BPM)',
                      'Lyrics',
                      'Sentiment Anaylsis']
            csv_writer.writerow(header)

    # Prepare results dictionary
    all_results = {}

    # Process each audio file in the directory
    for filename in os.listdir(audio_dir):
        # Skip if file already processed
        if filename in processed_files:
            print(f"Skipping already processed file: {filename}")
            continue

        if filename.endswith(('.wav', '.mp3', '.flac')):  # Add more extensions if needed
            file_path = os.path.join(audio_dir, filename)

            try:
                # Detect emotional breakpoints
                breakpoints = detect_emotional_breakpoints(file_path)
                
                if breakpoints:
                    # Split audio at breakpoints
                    segments = split_audio_at_breakpoints(file_path, breakpoints)
                else:
                    # If no breakpoints, treat entire file as one segment
                    segments = [{
                        'path': file_path,
                        'start_time': 0,
                        'end_time': librosa.get_duration(filename=file_path),
                        'transition_type': "full_song"
                    }]
    
                # Process each segment
                for i, segment in enumerate(segments):

                    # Get Top3 Genre, Mood and Instruments for the Music File
                    mtg_genre_predictions , essentia_tagging_predictions, fma_predictions  = genre_classifier.predict(segment['path'])
                    print(f'MTG-Genre Predictions for {filename}: {mtg_genre_predictions}')
                    print(f'Essentia-Genre Predictions for {filename}: {essentia_tagging_predictions}')
                    print(f'FMA TOP 3 GENRES PREDICTIONS for {filename}: {fma_predictions}')
                    
                    mood_predictions = mood_classifier.predict(segment['path'])
                    print(f'Mood Predictions for {filename}: {mood_predictions}')

                    instrument_predictions = Instrument_classifier.predict(segment['path'])
                    print(f'Instrument Predictions for {filename}: {instrument_predictions}')

                    # Get Key and BPM
                    key_n_bpm = KeyBPM.analyze(file_path=segment['path'])

                    # Sentiment Analysis
                    sentiment_results, has_lyrics = Sentiment_analyzer.analyze(
                        segment['path'],
                        moods=[mood for mood, prob in mood_predictions[:]]
                    )

                    # Process sentiment if lyrics exist
                    if has_lyrics:
                        result_token = "[RESULT]"
                        sentiment_results['sentiment'] = sentiment_results['sentiment'].split(result_token, 1)[1].strip()
                        sentiment_results['sentiment'] = sentiment_results['sentiment'].split(result_token)[1]

                    print(f"Sentiment for {filename}: {sentiment_results['sentiment']}")

                    # Prepare row for CSV
                    row = [
                        filename,
                        i,  # segment number
                        segment['start_time'],
                        segment['end_time'],
                        segment['transition_type']
                    ]

                    # Add MTG-genres
                    for genre, prob in mtg_genre_predictions:
                        row.extend([genre, prob])
                    # Pad with empty values if less than 3 genres
                    while len(row) < 7:
                        row.extend(['', ''])

                    # Add FMA-genres
                    for genre, prob in fma_predictions:
                        row.extend([genre, prob])
                    # Pad with empty values if less than 3 genres
                    while len(row) < 13:
                        row.extend(['', ''])

                    # Add Essentia-AutoTagging genres
                    for genre, prob in essentia_tagging_predictions:
                        row.extend([genre, prob])
                    # Pad with empty values if less than 3 genres
                    while len(row) < 19:
                        row.extend(['', ''])

                    # Add moods
                    for mood, prob in mood_predictions:
                        row.extend([mood, prob])
                    # Pad with empty values if less than 3 moods
                    while len(row) < 25:
                        row.extend(['', ''])

                    # Add instruments
                    for instrument, prob in instrument_predictions:
                        row.extend([instrument, prob])
                    # Pad with empty values if less than 3 instruments
                    while len(row) < 29:
                        row.extend(['', ''])

                    # Add keys
                    for key in key_n_bpm['Key']:
                        row.extend([key])
                    # Pad with empty values if less than 4 keys
                    while len(row) < 32:
                        row.append('')

                    for bpm in key_n_bpm['BPM']:
                        row.extend([bpm])
                    # Pad with empty values if less than 3 BPMs
                    while len(row) < 35:
                        row.append('')

                    # Add BPM, lyrics, and sentiment
                    row.extend([
                        sentiment_results['lyrics'],
                        sentiment_results['sentiment']
                    ])

                    # Append to CSV immediately after processing each file
                    with open(output_path, 'a', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(row)
                
                # Clean up temporary files if they were created
                if breakpoints:
                    for segment in segments:
                        if segment['path'] != file_path:  # Don't delete original file
                            os.remove(segment['path'])
                    os.rmdir(os.path.dirname(segments[0]['path']))  # Remove temp directory

                # Store results (optional)
                all_results[filename] = {
                    'mtg_genres': mtg_genre_predictions,
                    'fma_genres': fma_predictions,
                    'essentia_genres': essentia_tagging_predictions,
                    'moods': mood_predictions,
                    'instruments': instrument_predictions,
                    'key': key_n_bpm['Key'],
                    'bpm': key_n_bpm['BPM'],
                    'lyrics': sentiment_results['lyrics'],
                    'sentiment': sentiment_results['sentiment']
                }

                print(f"Processed {len(segments)} segments for {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

    return all_results



if __name__ == "__main__":
    # Get the directory where your script is located
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    Genreclassifier = AudioGenreClassifier(
        genre_model_path=os.path.join(BASE_DIR, "models/mtg_jamendo_genre-discogs-effnet-1.pb"),
        model_json_path=os.path.join(BASE_DIR, "metadata/mtg_jamendo_genre-discogs-effnet-1.json"),
        essentia_genre_model_path=os.path.join(BASE_DIR, "models/msd-musicnn-1.pb"),
        essentia_genre_json_path=os.path.join(BASE_DIR, "metadata/msd-musicnn-1.json"),
        embedding_model_path=os.path.join(BASE_DIR, "models/discogs-effnet-bs64-1.pb"),
        fma_model_path=os.path.join(BASE_DIR,'models/fma_models/xgb_model.pkl'),
        fma_scaler_path=os.path.join(BASE_DIR,'models/fma_models/standard_scaler.joblib'),
        fma_pca_path=os.path.join(BASE_DIR,'models/fma_models/pca_model.joblib')
    )

    # Mood Classifier
    MoodClassifier = AudioMoodClassifier(
        models_dir=os.path.join(BASE_DIR, "models/mood_detection_models"),
        metadatas_dir=os.path.join(BASE_DIR, "metadata/mood_detection_metadatas")
    )

    # Instrument Classifier
    Instrument_classifier = AudioInstrumentClassifier(
        instrument_model_path=os.path.join(BASE_DIR, "models/mtg_jamendo_instrument-discogs-effnet-1.pb"),
        model_json_path=os.path.join(BASE_DIR, "metadata/mtg_jamendo_instrument-discogs-effnet-1.json"),
        embedding_model_path=os.path.join(BASE_DIR, "models/discogs-effnet-bs64-1.pb")
    )

    #Create KeyBPMExtractor.
    KeyBPM_analyzer = KeyBPMExtractor()

    #Create LyricsExtractor Instance.
    Lyrics_analyzer = LyricsExtractor()

    # Create Sentiment analyzer
    Sentiment_analyzer = AudioSentimentAnalyzer(Lyrics_analyzer)

    #STORE RESULTS..

    #Store results of vocals dir in a CSV.
    process_directory(audio_dir='audio_files',
                    output_path='audio_files/output.csv',
                    genre_classifier=Genreclassifier,
                    mood_classifier=MoodClassifier,
                    Instrument_classifier=Instrument_classifier,
                    KeyBPM=KeyBPM_analyzer,
                    Sentiment_analyzer=Sentiment_analyzer)
