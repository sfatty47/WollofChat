import streamlit as st
import sys
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering
from llama_cpp import Llama
import speech_recognition as sr
from gtts import gTTS
import tempfile
import io
import base64

# --- LLM Model Path (Mistral) ---
# Download a Mistral GGUF model (e.g., mistral-7b-instruct-v0.2.Q4_K_M.gguf) from TheBloke on HuggingFace:
# https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
# Place the .gguf file in your project directory and set the path below:
MISTRAL_MODEL_PATH = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# --- Language Switch ---
language = st.sidebar.selectbox("Language / Langu (UI)", ["Wolof", "English"])
LOCALE = {
    "Wolof": {
        "title": "Wolof Edu: Jàngat ak Laaj",
        "desc": "Soppaliku laaj bi ci Wolof ci xam-xam (mathematics, science, history, geography, technology, literature). Tontu bi dina ñëw ci Wolof.",
        "ask": "Laaj sa laaj ci Wolof:",
        "subject": "Tànn ab suujet:",
        "get_answer": "Jox ma tontu",
        "warn_empty": "Soppaliku laaj bu am solo ci Wolof.",
        "processing": "Dinaa defar...",
        "answer": "Tontu bi:",
        "voice_input": "Wax sa laaj (Voice Input)",
        "voice_output": "Déglu tontu bi (Listen to Answer)",
        "record": "Bëgg nga wax?",
        "stop_record": "Bàyyi wax",
        "listening": "Déglu laa...",
        "error_voice": "Problème ci wax bi. Seetil ko.",
        "no_voice": "Wax bi du déglu. Seetil ko.",
        "feedback": "Nataal (Feedback)",
        "feedback_placeholder": "Waxal nu lan la defar...",
        "submit_feedback": "Yónni nataal",
        "feedback_sent": "Nataal bi yónni na!",
        "torch_error": "**Jàmm rekk!**\n\nApp bi dafa soxla torch version bu bees (2.6 walla siiw).\n\nMën nga def: `pip install --upgrade torch`\n\nWall bu ko gisul, war nga suqali Python 3.10 walla 3.9, walla xam ne version bu bees duñu ko defar ci sa system.\n\nApp bi du dox ba mu am torch 2.6+"
    },
    "English": {
        "title": "Wolof Edu: Learn & Ask",
        "desc": "Ask questions in Wolof about math, science, history, geography, technology, or literature. Get answers in Wolof.",
        "ask": "Ask your question in Wolof:",
        "subject": "Select a subject:",
        "get_answer": "Get Answer",
        "warn_empty": "Please enter a question in Wolof.",
        "processing": "Processing...",
        "answer": "Answer:",
        "voice_input": "Voice Input",
        "voice_output": "Listen to Answer",
        "record": "Click to speak",
        "stop_record": "Stop recording",
        "listening": "Listening...",
        "error_voice": "Voice input error. Please try again.",
        "no_voice": "No voice detected. Please try again.",
        "feedback": "Feedback",
        "feedback_placeholder": "Tell us how we can improve...",
        "submit_feedback": "Submit Feedback",
        "feedback_sent": "Feedback sent!",
        "torch_error": "**Error!**\n\nThis app requires torch version 2.6 or higher.\n\nTry: `pip install --upgrade torch`\n\nIf not available, you may need to upgrade Python to 3.10 or 3.9, or wait for torch 2.6+ to be available for your system.\n\nThe app cannot run without torch 2.6+"
    }
}

# --- Torch version check ---
REQUIRED_TORCH_VERSION = (2, 6)
torch_version_tuple = tuple(map(int, torch.__version__.split(".")[:2]))
if torch_version_tuple < REQUIRED_TORCH_VERSION:
    st.error(LOCALE[language]["torch_error"])
    st.stop()

# --- Model loading ---
@st.cache_resource(show_spinner=True)
def load_nllb():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource(show_spinner=True)
def load_qa_model():
    qa_model_name = "deepset/roberta-base-squad2"
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    return qa_tokenizer, qa_model

@st.cache_resource(show_spinner=True)
def load_llm():
    if os.path.exists(MISTRAL_MODEL_PATH):
        return Llama(
            model_path=MISTRAL_MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )
    else:
        return None

# --- Voice functions ---
def record_audio():
    """Record audio from microphone"""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info(LOCALE[language]["listening"])
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            return audio
        except sr.WaitTimeoutError:
            st.error(LOCALE[language]["no_voice"])
            return None
        except Exception as e:
            st.error(LOCALE[language]["error_voice"])
            return None

def speech_to_text(audio):
    """Convert speech to text"""
    recognizer = sr.Recognizer()
    try:
        # Try Wolof first, then English
        text = recognizer.recognize_google(audio, language="wo-SN")
        return text
    except:
        try:
            # Fallback to English
            text = recognizer.recognize_google(audio, language="en-US")
            return text
        except:
            st.error(LOCALE[language]["error_voice"])
            return None

def text_to_speech(text, lang="wo"):
    """Convert text to speech"""
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(fp.name)
        return fp.name
    except Exception as e:
        st.error(f"TTS error: {e}")
        return None

def get_audio_player(audio_file):
    """Create audio player for Streamlit"""
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/mp3")

# --- Translation functions ---
def translate(text, src_lang, tgt_lang, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    inputs = {k: v for k, v in inputs.items()}
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=256
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def answer_question(question, context, qa_tokenizer, qa_model):
    inputs = qa_tokenizer.encode_plus(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = qa_model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = qa_tokenizer.convert_tokens_to_string(
            qa_tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0][answer_start:answer_end]
            )
        )
    return answer

def get_llm_answer(question, llm_model):
    """Get answer from LLM"""
    if llm_model is None:
        return "LLM model not found. Please download the Mistral GGUF model."
    
    prompt = f"""You are a helpful educational assistant. Answer the following question clearly and accurately:

Question: {question}

Answer:"""
    
    try:
        response = llm_model(prompt, max_tokens=512, temperature=0.7, stop=["\n\n"])
        return response['choices'][0]['text'].strip()
    except Exception as e:
        return f"LLM error: {e}"

# --- Static contexts ---
CONTEXTS = {
    "Mathematics": "Mathematics is the study of numbers, quantities, shapes, and patterns. Basic operations include addition, subtraction, multiplication, and division. Algebra deals with variables and equations. Geometry studies shapes, sizes, and spatial relationships. Calculus involves rates of change and accumulation. Statistics deals with data collection, analysis, and interpretation.",
    "Science": "Science is the systematic study of the natural world through observation and experimentation. Physics studies matter, energy, and their interactions. Chemistry examines the composition and properties of substances. Biology studies living organisms and life processes. Earth science explores our planet's systems and processes.",
    "History": "History is the study of past events and human societies. Ancient civilizations include Egypt, Greece, Rome, and China. The Middle Ages saw feudalism and the rise of kingdoms. The Renaissance brought cultural and scientific advances. The Industrial Revolution transformed manufacturing and society. World Wars I and II shaped the modern world.",
    "Geography": "Geography studies Earth's physical features and human societies. Physical geography examines landforms, climate, and natural resources. Human geography studies population, culture, and economic activities. Continents include Africa, Asia, Europe, North America, South America, Australia, and Antarctica. Major oceans are Pacific, Atlantic, Indian, and Arctic.",
    "Technology": "Technology involves tools, machines, and systems that solve problems. Computers process information using hardware and software. The internet connects global networks for communication and information sharing. Artificial intelligence enables machines to learn and make decisions. Mobile devices provide portable computing and communication capabilities.",
    "Literature": "Literature includes written works of artistic and intellectual value. Poetry uses rhythm, rhyme, and imagery to express ideas and emotions. Prose includes novels, short stories, and essays. Drama presents stories through dialogue and performance. Different genres include fiction, non-fiction, mystery, romance, and science fiction."
}

# --- Main app ---
st.title(LOCALE[language]["title"])
st.write(LOCALE[language]["desc"])

# Load models
with st.spinner("Loading models..."):
    tokenizer, nllb_model = load_nllb()
    qa_tokenizer, qa_model = load_qa_model()
    llm_model = load_llm()

# Subject selection
subject = st.selectbox(LOCALE[language]["subject"], list(CONTEXTS.keys()))

# Input method selection
input_method = st.radio("Input method:", ["Text", "Voice"])

wolof_question = ""

if input_method == "Text":
    wolof_question = st.text_input(LOCALE[language]["ask"])
else:
    # Voice input
    col1, col2 = st.columns(2)
    with col1:
        if st.button(LOCALE[language]["record"]):
            audio = record_audio()
            if audio:
                text = speech_to_text(audio)
                if text:
                    wolof_question = text
                    st.success(f"Recognized: {text}")

# Process question
if st.button(LOCALE[language]["get_answer"]):
    if not wolof_question.strip():
        st.warning(LOCALE[language]["warn_empty"])
    else:
        with st.spinner(LOCALE[language]["processing"]):
            # Show original question
            st.write(f"**Question:** {wolof_question}")
            
            # Translate Wolof to English
            english_question = translate(wolof_question, "wol_Latn", "eng_Latn", tokenizer, nllb_model)
            st.write(f"**English:** {english_question}")
            
            # Get answer (try LLM first, fallback to QA)
            if llm_model:
                english_answer = get_llm_answer(english_question, llm_model)
                st.write(f"**LLM Answer:** {english_answer}")
            else:
                # Use QA model with context
                context = CONTEXTS[subject]
                english_answer = answer_question(english_question, context, qa_tokenizer, qa_model)
                st.write(f"**QA Answer:** {english_answer}")
            
            # Translate answer back to Wolof
            wolof_answer = translate(english_answer, "eng_Latn", "wol_Latn", tokenizer, nllb_model)
            
            # Display Wolof answer
            st.success(f"**{LOCALE[language]['answer']}** {wolof_answer}")
            
            # Voice output
            if st.button(LOCALE[language]["voice_output"]):
                audio_file = text_to_speech(wolof_answer, "wo")
                if audio_file:
                    get_audio_player(audio_file)
                    # Clean up temp file
                    os.unlink(audio_file)

# Feedback section
st.markdown("---")
st.subheader(LOCALE[language]["feedback"])
feedback = st.text_area(LOCALE[language]["feedback_placeholder"])
if st.button(LOCALE[language]["submit_feedback"]):
    if feedback.strip():
        # Save feedback to CSV
        feedback_data = pd.DataFrame({
            "feedback": [feedback],
            "language": [language],
            "timestamp": [pd.Timestamp.now()]
        })
        
        if os.path.exists("feedback.csv"):
            feedback_data.to_csv("feedback.csv", mode="a", header=False, index=False)
        else:
            feedback_data.to_csv("feedback.csv", index=False)
        
        st.success(LOCALE[language]["feedback_sent"]) 