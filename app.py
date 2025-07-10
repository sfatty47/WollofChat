#!/usr/bin/env python3
"""
WolofChat - Wolof Educational Q&A App
A general-purpose educational chatbot for Wolof language and culture.

This app can be deployed on various platforms:
- Streamlit Cloud
- Flask/Django
- FastAPI
- Heroku
- Railway
- Vercel
- AWS/GCP/Azure
"""

import os
import json
import tempfile
import re
import math
import requests
from bs4 import BeautifulSoup
import urllib.parse
import speech_recognition as sr
from gtts import gTTS
from typing import Dict, List, Optional, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WolofChatConfig:
    """Configuration class for WolofChat app"""
    
    def __init__(self):
        # LLM Configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434')
        self.huggingface_token = os.getenv('HUGGINGFACE_TOKEN', '')
        
        # Speech Recognition Configuration
        self.speech_recognition_enabled = os.getenv('SPEECH_RECOGNITION_ENABLED', 'true').lower() == 'true'
        self.google_speech_api_key = os.getenv('GOOGLE_SPEECH_API_KEY', '')
        
        # Text-to-Speech Configuration
        self.tts_enabled = os.getenv('TTS_ENABLED', 'true').lower() == 'true'
        
        # Web Search Configuration
        self.web_search_enabled = os.getenv('WEB_SEARCH_ENABLED', 'true').lower() == 'true'
        
        # App Configuration
        self.debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        self.max_tokens = int(os.getenv('MAX_TOKENS', '1000'))
        self.temperature = float(os.getenv('TEMPERATURE', '0.7'))

class LLMService:
    """Abstract base class for LLM services"""
    
    def __init__(self, config: WolofChatConfig):
        self.config = config
        self.available = False
    
    def is_available(self) -> bool:
        return self.available
    
    def generate_response(self, question: str, subject: str, context: str = "", question_language: str = "Wolof") -> Optional[str]:
        raise NotImplementedError

class OpenAIService(LLMService):
    """OpenAI GPT service"""
    
    def __init__(self, config: WolofChatConfig):
        super().__init__(config)
        self.available = bool(config.openai_api_key)
        if self.available:
            try:
                import openai
                openai.api_key = config.openai_api_key
                # Test connection
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
                logger.info("OpenAI service available")
            except Exception as e:
                logger.warning(f"OpenAI service not available: {e}")
                self.available = False
    
    def generate_response(self, question: str, subject: str, context: str = "", question_language: str = "Wolof") -> Optional[str]:
        if not self.available:
            return None
        
        try:
            import openai
            
            if question_language == "English":
                system_prompt = """You are a helpful educational assistant specializing in Wolof language and African culture. The user asked a question in English, but please provide your answer in Wolof.

Your responses should:
1. Be detailed and informative
2. Use proper Wolof grammar and vocabulary
3. Include relevant historical or cultural context when appropriate
4. Be educational and suitable for learning
5. If the question is about a person, place, or concept, provide thorough information
6. Use Wolof expressions and cultural references when relevant
7. Be comprehensive and help the user understand the topic better

Always respond in Wolof language."""
                
                user_prompt = f"""English Question: {question}
Subject: {subject}
Context: {context}

Please provide a comprehensive, accurate, and educational answer in Wolof."""
            else:
                system_prompt = """You are a helpful educational assistant specializing in Wolof language and African culture. Answer questions in Wolof language.

Your responses should:
1. Be detailed and informative
2. Use proper Wolof grammar and vocabulary
3. Include relevant historical or cultural context when appropriate
4. Be educational and suitable for learning
5. If the question is about a person, place, or concept, provide thorough information
6. Use Wolof expressions and cultural references when relevant
7. Be comprehensive and help the user understand the topic better

Always respond in Wolof language."""
                
                user_prompt = f"""Question: {question}
Subject: {subject}
Context: {context}

Please provide a comprehensive, accurate, and educational answer in Wolof."""

            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return None

class OllamaService(LLMService):
    """Ollama Mistral service"""
    
    def __init__(self, config: WolofChatConfig):
        super().__init__(config)
        try:
            import ollama
            response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': 'test'}])
            self.available = True
            logger.info("Ollama service available")
        except Exception as e:
            logger.warning(f"Ollama service not available: {e}")
            self.available = False
    
    def generate_response(self, question: str, subject: str, context: str = "", question_language: str = "Wolof") -> Optional[str]:
        if not self.available:
            return None
        
        try:
            import ollama
            
            if question_language == "English":
                prompt = f"""You are a helpful educational assistant specializing in Wolof language and African culture. The user asked a question in English, but please provide your answer in Wolof.

English Question: {question}
Subject: {subject}
Context: {context}

Please provide a comprehensive, accurate, and educational answer in Wolof. Your response should:
1. Be detailed and informative
2. Use proper Wolof grammar and vocabulary
3. Include relevant historical or cultural context when appropriate
4. Be educational and suitable for learning
5. If the question is about a person, place, or concept, provide thorough information
6. Use Wolof expressions and cultural references when relevant

Make sure your answer is educational, informative, and helps the user understand the topic better.

Answer in Wolof:"""
            else:
                prompt = f"""You are a helpful educational assistant specializing in Wolof language and African culture. Answer the following question about {subject} in Wolof language.

Question: {question}
Context: {context}

Please provide a comprehensive, accurate, and educational answer in Wolof. Your response should:
1. Be detailed and informative
2. Use proper Wolof grammar and vocabulary
3. Include relevant historical or cultural context when appropriate
4. Be educational and suitable for learning
5. If the question is about a person, place, or concept, provide thorough information
6. Use Wolof expressions and cultural references when relevant

Make sure your answer is educational, informative, and helps the user understand the topic better.

Answer in Wolof:"""

            response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt}])
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return None

class SpeechRecognitionService:
    """Speech recognition service using Google Speech Recognition"""
    
    def __init__(self, config: WolofChatConfig):
        self.config = config
        self.available = config.speech_recognition_enabled
    
    def is_available(self) -> bool:
        return self.available
    
    def record_audio(self) -> Optional[sr.AudioData]:
        """Record audio from microphone"""
        if not self.available:
            return None
        
        try:
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                logger.info("Listening... Speak now!")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
                return audio
        except sr.WaitTimeoutError:
            logger.warning("No speech detected within timeout")
            return None
        except Exception as e:
            logger.error(f"Microphone error: {e}")
            return None
    
    def speech_to_text(self, audio_data: sr.AudioData, language: str = "en-US") -> Optional[str]:
        """Convert speech to text"""
        if not self.available or not audio_data:
            return None
        
        try:
            recognizer = sr.Recognizer()
            
            if language == "wo":
                try:
                    text = recognizer.recognize_google(audio_data, language="wo-SN")
                    return text
                except:
                    text = recognizer.recognize_google(audio_data, language="en-US")
                    return text
            else:
                text = recognizer.recognize_google(audio_data, language=language)
                return text
                
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return None

class TextToSpeechService:
    """Text-to-speech service using gTTS"""
    
    def __init__(self, config: WolofChatConfig):
        self.config = config
        self.available = config.tts_enabled
    
    def is_available(self) -> bool:
        return self.available
    
    def text_to_speech(self, text: str, lang: str = "wo") -> Optional[str]:
        """Convert text to speech and return file path"""
        if not self.available:
            return None
        
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            fp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(fp.name)
            return fp.name
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

class WebSearchService:
    """Web search service using Wikipedia and other sources"""
    
    def __init__(self, config: WolofChatConfig):
        self.config = config
        self.available = config.web_search_enabled
    
    def is_available(self) -> bool:
        return self.available
    
    def search(self, query: str, max_results: int = 2) -> List[Dict[str, str]]:
        """Search the web for information"""
        if not self.available:
            return []
        
        try:
            # Try Wikipedia API first
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(query)}"
            response = requests.get(wiki_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if 'extract' in data:
                    return [{
                        'title': data.get('title', query),
                        'body': data.get('extract', ''),
                        'link': data.get('content_urls', {}).get('desktop', {}).get('page', '')
                    }]
            
            # Fallback to structured knowledge
            return self.get_structured_knowledge(query)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return self.get_structured_knowledge(query)
    
    def get_structured_knowledge(self, query: str) -> List[Dict[str, str]]:
        """Get structured knowledge for common topics"""
        query_lower = query.lower()
        
        knowledge_base = {
            "sundiata keita": {
                "title": "Sundiata Keita",
                "body": "Sundiata Keita was the founder of the Mali Empire, ruling from 1235 to 1255. He is celebrated as a hero of the Mandinka people and is the subject of the Epic of Sundiata. He united the Mandinka kingdoms and established the Mali Empire as a major power in West Africa.",
                "link": "https://en.wikipedia.org/wiki/Sundiata_Keita"
            },
            "mansa musa": {
                "title": "Mansa Musa",
                "body": "Mansa Musa was the ninth Mansa of the Mali Empire, reigning from 1312 to 1337. He is famous for his wealth and his pilgrimage to Mecca in 1324, during which he distributed so much gold that it caused inflation in the regions he passed through.",
                "link": "https://en.wikipedia.org/wiki/Mansa_Musa"
            },
            "senegal": {
                "title": "Senegal",
                "body": "Senegal is a country in West Africa. It gained independence from France in 1960. Dakar is its capital and largest city. The country is known for its diverse culture and is home to the Wolof people and language.",
                "link": "https://en.wikipedia.org/wiki/Senegal"
            },
            "africa": {
                "title": "Africa",
                "body": "Africa is the world's second-largest continent, covering about 30.3 million square kilometers. It has 54 countries and is home to over 1.3 billion people. Africa has a rich history, diverse cultures, and abundant natural resources.",
                "link": "https://en.wikipedia.org/wiki/Africa"
            }
        }
        
        for key, info in knowledge_base.items():
            if key in query_lower:
                return [info]
        
        return []

class WolofChatApp:
    """Main WolofChat application class"""
    
    def __init__(self, config: WolofChatConfig = None):
        self.config = config or WolofChatConfig()
        
        # Initialize services
        self.openai_service = OpenAIService(self.config)
        self.ollama_service = OllamaService(self.config)
        self.speech_service = SpeechRecognitionService(self.config)
        self.tts_service = TextToSpeechService(self.config)
        self.web_search_service = WebSearchService(self.config)
        
        # Load knowledge base
        self.qa_pairs = self._load_qa_pairs()
        self.wolof_english_dict = self._load_wolof_english_dict()
    
    def _load_qa_pairs(self) -> Dict[str, Dict[str, str]]:
        """Load pre-defined Q&A pairs"""
        return {
            "history": {
                "kan moi mansa musa": "Mansa Musa mooy ab mansa bu gudd ci Mali Empire. Moom moo doon mansa ci 1312 ba 1337. Moom moo gudd ci wealth ak pilgrimage ci Mecca.",
                "sundiata keita": "Sundiata Keita mooy ab mansa bu gudd ci Mali Empire. Moom moo doon founder ci Mali Empire ci 1235. Sundiata Keita mooy ab mansa bu am military skills bu gudd."
            },
            "mathematics": {
                "lan la addition": "Addition mooy ab operation ci mathematics bu joxal sum of two numbers. Example: 2 + 3 = 5.",
                "lan la multiplication": "Multiplication mooy ab operation ci mathematics bu joxal product of two numbers. Example: 2 × 3 = 6."
            },
            "science": {
                "lan la photosynthesis": "Photosynthesis mooy ab process ci plants bu defar food using sunlight, carbon dioxide, ak water.",
                "lan la water": "Water mooy ab compound bu am hydrogen ak oxygen (H2O). Water mooy essential ci life."
            },
            "general": {
                "salamalekum": "Malekum salam! Naka nga? (Hello! How are you?)",
                "lan la wolof": "Wolof mooy ab language bu spoken ci Senegal ak Gambia. Wolof mooy am rich culture ak history."
            }
        }
    
    def _load_wolof_english_dict(self) -> Dict[str, str]:
        """Load Wolof-English dictionary"""
        return {
            "kan": "who",
            "lan": "what",
            "moi": "king",
            "mansa": "king",
            "xam": "know",
            "xam-xam": "knowledge",
            "laaj": "question",
            "tontu": "answer",
            "suujet": "subject",
            "bi": "the",
            "ci": "in",
            "ak": "and",
            "dina": "will",
            "ñëw": "come",
            "jox": "give",
            "ma": "me",
            "sa": "your",
            "nu": "we",
            "defar": "do",
            "waxal": "tell",
            "yónni": "send",
            "nataal": "feedback",
            "déglu": "listen",
            "wax": "speak",
            "soppaliku": "please",
            "tànn": "choose",
            "ab": "a",
            "bu": "that",
            "am": "have",
            "solo": "important"
        }
    
    def translate_wolof_to_english(self, text: str) -> str:
        """Translate Wolof to English for search"""
        text_lower = text.lower().strip()
        for wolof, english in self.wolof_english_dict.items():
            text_lower = text_lower.replace(wolof.lower(), english)
        
        # Handle common patterns
        text_lower = re.sub(r'\bkan\s+moi\b', 'who is king', text_lower)
        text_lower = re.sub(r'\bmansa\s+musa\b', 'King Moussa', text_lower)
        
        return text_lower.capitalize()
    
    def extract_math_expression(self, text: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        """Extract and calculate mathematical expressions"""
        text_lower = text.lower()
        
        # Look for mathematical patterns
        math_patterns = [
            r'(\d+)\s*plus\s*(\d+)',
            r'(\d+)\s*\+\s*(\d+)',
            r'(\d+)\s*minus\s*(\d+)',
            r'(\d+)\s*-\s*(\d+)',
            r'(\d+)\s*times\s*(\d+)',
            r'(\d+)\s*\*\s*(\d+)',
            r'(\d+)\s*divided\s*by\s*(\d+)',
            r'(\d+)\s*/\s*(\d+)'
        ]
        
        operations = ['plus', 'minus', 'times', 'divided']
        
        for i, pattern in enumerate(math_patterns):
            match = re.search(pattern, text_lower)
            if match:
                a, b = int(match.group(1)), int(match.group(2))
                operation = operations[i // 2]
                
                if operation == 'plus':
                    result = a + b
                elif operation == 'minus':
                    result = a - b
                elif operation == 'times':
                    result = a * b
                elif operation == 'divided':
                    result = a / b if b != 0 else None
                
                if result is not None:
                    return f"{a} {operation} {b}", result, operation
        
        return None, None, None
    
    def get_math_answer_in_wolof(self, expression: str, result: float, operation: str) -> str:
        """Generate Wolof answer for mathematical expressions"""
        operation_wolof = {
            'plus': 'addition',
            'minus': 'subtraction', 
            'times': 'multiplication',
            'divided': 'division'
        }.get(operation, operation)
        
        return f"Ci mathematics, {expression} mooy {result}. Bii mooy {operation_wolof}."
    
    def find_answer(self, question: str, subject: str, question_language: str = "Wolof") -> Tuple[str, Optional[List[str]]]:
        """Find answer using multiple services"""
        question_lower = question.lower().strip()
        
        # Check for mathematical expressions
        expression, result, operation = self.extract_math_expression(question_lower)
        if expression and result is not None:
            return self.get_math_answer_in_wolof(expression, result, operation), None
        
        # Try OpenAI first
        if self.openai_service.is_available():
            logger.info("Trying OpenAI...")
            answer = self.openai_service.generate_response(question, subject, "", question_language)
            if answer and len(answer) > 30:
                return answer, None
        
        # Try Ollama
        if self.ollama_service.is_available():
            logger.info("Trying Ollama...")
            answer = self.ollama_service.generate_response(question, subject, "", question_language)
            if answer and len(answer) > 30:
                return answer, None
        
        # Check pre-defined answers
        if subject.lower() in self.qa_pairs:
            for pattern, answer in self.qa_pairs[subject.lower()].items():
                if pattern in question_lower or question_lower in pattern:
                    return answer, None
        
        # Check general answers
        if "general" in self.qa_pairs:
            for pattern, answer in self.qa_pairs["general"].items():
                if pattern in question_lower or question_lower in pattern:
                    return answer, None
        
        # Try web search
        if self.web_search_service.is_available():
            if question_language == "Wolof":
                english_question = self.translate_wolof_to_english(question)
            else:
                english_question = question
            
            if english_question and len(english_question) > 3:
                logger.info("Searching the web...")
                search_results = self.web_search_service.search(english_question)
                if search_results:
                    answer = self._generate_answer_from_search(question, search_results, subject)
                    sources = [result.get('link', '') for result in search_results if result.get('link')]
                    return answer, sources
        
        # Generic answer
        generic_answers = {
            "history": "Ci history, nu mooy study past events ak human societies. History mooy help nu understand present ak future.",
            "mathematics": "Ci mathematics, nu mooy study numbers, quantities, shapes, ak patterns. Mathematics mooy am importance ci everyday life.",
            "science": "Ci science, nu mooy study natural world through observation ak experimentation. Science mooy help nu understand world around nu.",
            "geography": "Ci geography, nu mooy study Earth's physical features ak human societies. Geography mooy help nu understand different places ak cultures.",
            "technology": "Ci technology, nu mooy use tools, machines, ak systems bu solve problems. Technology mooy am impact on modern life.",
            "literature": "Ci literature, nu mooy study written works bu am artistic ak intellectual value. Literature mooy help nu understand human experience.",
            "general": "Ci general knowledge, nu mooy study various topics bu am importance ci life. Education mooy am importance ci personal development."
        }
        
        return generic_answers.get(subject.lower(), "Ci subject bii, nu mooy study various topics bu am importance ci education. Soppaliku, waxal ma question bu specific bu nu mooy answer."), None
    
    def _generate_answer_from_search(self, question: str, search_results: List[Dict[str, str]], subject: str) -> str:
        """Generate answer from search results"""
        if not search_results:
            return "No information found."
        
        relevant_info = []
        for result in search_results:
            if result.get('title') and result.get('body'):
                relevant_info.append(f"{result['title']}: {result['body']}")
        
        if not relevant_info:
            return "No relevant information found."
        
        combined_info = " ".join(relevant_info[:2])
        
        if subject.lower() == "history":
            return f"Ci history, {combined_info[:300]}... Bii mooy information bu nu am ci {question}."
        elif subject.lower() == "mathematics":
            return f"Ci mathematics, {combined_info[:300]}... Bii mooy mathematical concept bu nu am ci {question}."
        elif subject.lower() == "science":
            return f"Ci science, {combined_info[:300]}... Bii mooy scientific information bu nu am ci {question}."
        else:
            return f"Ci {subject.lower()}, {combined_info[:300]}... Bii mooy information bu nu am ci {question}."
    
    def get_service_status(self) -> Dict[str, bool]:
        """Get status of all services"""
        return {
            "openai": self.openai_service.is_available(),
            "ollama": self.ollama_service.is_available(),
            "speech_recognition": self.speech_service.is_available(),
            "text_to_speech": self.tts_service.is_available(),
            "web_search": self.web_search_service.is_available()
        }
    
    def process_voice_input(self, audio_data: sr.AudioData, input_language: str = "Wolof") -> Optional[str]:
        """Process voice input and return text"""
        if not self.speech_service.is_available():
            return None
        
        return self.speech_service.speech_to_text(audio_data, "en-US" if input_language == "English" else "wo")
    
    def generate_voice_output(self, text: str, lang: str = "wo") -> Optional[str]:
        """Generate voice output and return file path"""
        if not self.tts_service.is_available():
            return None
        
        return self.tts_service.text_to_speech(text, lang)

# Global app instance
app_instance = None

def get_app() -> WolofChatApp:
    """Get or create the global app instance"""
    global app_instance
    if app_instance is None:
        app_instance = WolofChatApp()
    return app_instance

if __name__ == "__main__":
    # Example usage
    app = WolofChatApp()
    print("WolofChat App initialized!")
    print("Service status:", app.get_service_status())
    
    # Example question
    question = "Kan moi Mansa Musa?"
    answer, sources = app.find_answer(question, "History", "Wolof")
    print(f"Q: {question}")
    print(f"A: {answer}") 