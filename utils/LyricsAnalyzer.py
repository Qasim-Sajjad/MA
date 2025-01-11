import re,os
from typing import Optional, Tuple, Dict
from collections import Counter
import string
import json
from dotenv import load_dotenv
from huggingface_hub import login, InferenceClient

class LyricsExtractor:
    def __init__(self, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        """
        Initializes the lyrics extractor with validation and sentiment analysis capabilities.

        Args:
            model_name (str): Hugging Face model name for LLM.
        """

        # ------------ Mixtral Model For Sentiment Analysis. UPDATE API TOKEN HERE -------------
        load_dotenv(dotenv_path='utils/.env')
        token = os.getenv(key="hf_mixtral") # PLACE API TOKEN.
        if token is None:
            raise ValueError("MIXTRAL LLM TOKEN DOES NOT EXIST, PLEASE CREATE A .ENV FILE AND PASTE token with name 'hf_mixtral'.")
        
        login(token=token, add_to_git_credential=True)
        self.llm_client = InferenceClient(
            model=model_name,
            timeout=200,
        )

    def is_valid_lyrics(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Performs rule-based validation of lyrics text.
        
        Args:
            text (str): Text to validate
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, reason_if_invalid)
        """
        if not text or not isinstance(text, str):
            return False, "Empty or invalid text"
            
        text = text.strip()
        
        if len(text) < 10:
            return False, "Text too short"
        
        cleaned_text = text.lower()
        words = cleaned_text.split()
        
        if not words:
            return False, "No words found after cleaning"
            
        # Check word repetition
        word_counts = Counter(words)
        most_common_word, count = word_counts.most_common(1)[0]
        repetition_ratio = count / len(words)
        
        if repetition_ratio > 0.3 and len(words) > 5:
            return False, f"Excessive repetition (word '{most_common_word}' appears {count} times)"
                            
        # Check word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length > 15:
            return False, "Average word length too high"
                
        # Check special characters
        special_chars = sum(1 for char in text if char in string.punctuation)
        if special_chars / len(text) > 0.3:
            return False, "Too many special characters"
            
        # Check timestamps
        if re.search(r'\d{1,2}:\d{2}', text):
            return False, "Contains timestamp-like patterns"
            
        return True, None

    def verify_lyrics_with_llm(self, text: str) -> Tuple[bool, str]:
        """
        Uses LLM to verify if the text appears to be valid lyrics.
        
        Args:
            text (str): Text to verify
            
        Returns:
            Tuple[bool, str]: (is_valid, explanation)
        """
        prompt = f"""
        Analyze the following text and determine if it represents valid song lyrics based on these key criteria:

        Purpose: Evaluate text for song lyric authenticity and provide clear validation.

        IMPORTANT VALIDATION RULES:
        - Must have coherent narrative or thematic flow
        - Should demonstrate rhythmic or musical structure
        - Contains repeating elements like chorus/hooks
        - Uses poetic devices (metaphors, rhymes, etc.)
        - Has appropriate line breaks and verses
        - Follows natural language patterns
        - Contains emotional or meaningful content

        Text to analyze: {text}

        OUTPUT FORMAT:
        Respond with:
        - 'VALID' or 'INVALID' status
        - Brief explanation (max 50 words)
        - Confidence score (0-100%)

        Example Valid Response:
        VALID: Clear verse-chorus structure, consistent rhyme scheme, emotionally resonant theme about love. Strong poetic imagery and natural flow. (Confidence: 92%)

        Example Invalid Response:
        INVALID: Random words without coherent structure or meaning. No musical elements or rhyme scheme present. Lacks any clear theme or purpose. (Confidence: 88%)

        Analyze the text and provide result after [RESULT] token.[RESULT]:
        """

        response = self.call_llm(self.llm_client, prompt, max_tokens=50)
        
        is_valid = False
        explanation = response.split('[RESULT]', 1)[1].strip() if '[RESULT]' in response else response
        if 'VALID' in explanation.upper():
            is_valid = True
        elif 'INVALID' in explanation.upper():
            is_valid = False
        return is_valid, explanation

    def call_llm(self, inference_client: InferenceClient, prompt: str, max_tokens: int) -> str:
        """
        Makes a call to the LLM.

        Args:
            inference_client: The inference client
            prompt (str): Input prompt
            max_tokens (int): Maximum tokens to generate

        Returns:
            str: Generated text
        """
        response = inference_client.post(
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": max_tokens},
                "task": "text-generation",
            },
        )
        return json.loads(response.decode())[0]["generated_text"]

    def analyze_sentiment(self, lyrics_text: str, max_new_tokens: int = 60) -> Optional[Dict]:
        """
        Validates and analyzes the sentiment of provided lyrics.

        Args:
            lyrics_text (str): Lyrics text to analyze
            max_new_tokens (int): Maximum number of tokens to generate

        Returns:
            Optional[Dict]: Analysis results including validation and sentiment
        """
        # First perform rule-based validation
        is_valid_rules, rule_reason = self.is_valid_lyrics(lyrics_text)
        
        if not is_valid_rules:
            return {
                'valid': False,
                'reason': f"Rule-based validation failed: {rule_reason}",
                'sentiment': None
            }

        # Then verify with LLM
        is_valid_llm, llm_explanation = self.verify_lyrics_with_llm(lyrics_text)
        
        if not is_valid_llm:
            return {
                'valid': False,
                'reason': f"LLM validation failed: {llm_explanation}",
                'sentiment': None
            }

        # If both validations pass, perform sentiment analysis
        sentiment_prompt = (
            f"Below are some song lyrics:\n\n{lyrics_text}\n\n"
            "Analyze the text above and describe the overall sentiment and mood of these lyrics in one short and concise sentence. "
            "Provide your response after the [RESULT] token.\n\n [RESULT]"
        )

        sentiment_result = self.call_llm(self.llm_client, sentiment_prompt, max_new_tokens)

        return {
            'valid': True,
            'validation_details': {
                'rule_based': rule_reason,
                'llm_based': llm_explanation
            },
            'sentiment': sentiment_result
        }