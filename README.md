# WollofChat
Wolof educational question-answering app that can answer basic math, science, and history questions in Wollof

# Wolof Educational Question-Answering App

This app helps Wolof-speaking students get answers to math, science, and history questions in Wolof, using state-of-the-art translation and question-answering models.

## Features
- Ask questions in Wolof, get answers in Wolof
- **Voice input and output** - speak your questions and listen to answers
- Uses facebook/nllb-200-distilled-600M for translation (Wolof ↔ English)
- Uses deepset/roberta-base-squad2 for English question answering
- Optional: Local Mistral 7B LLM for more comprehensive answers
- Static context for math, science, history, geography, technology, and literature
- Language switch (Wolof/English UI)
- Feedback system for users

## Setup
1. **Clone the repo**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **For voice features (optional):**
   - Install system audio dependencies:
     - **macOS:** `brew install portaudio`
     - **Ubuntu/Debian:** `sudo apt-get install portaudio19-dev python3-pyaudio`
     - **Windows:** Usually works out of the box
4. **Run the app:**
   ```bash
   streamlit run wolof_edu_qa_app.py
   ```

## Requirements
- Python 3.9, 3.10, or 3.11 (recommended)
- torch >=2.6 required for transformers models
- Microphone access for voice input
- Speakers/headphones for voice output

## Voice Features
- **Voice Input:** Click "Click to speak" and ask your question verbally
- **Voice Output:** Click "Listen to Answer" to hear the Wolof answer spoken
- Supports both Wolof and English speech recognition
- Uses Google's text-to-speech for natural Wolof pronunciation

## Optional LLM Setup
For more comprehensive answers using Mistral 7B:
1. Download a Mistral GGUF model from [TheBloke on HuggingFace](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
2. Place the `.gguf` file in your project directory
3. The app will automatically use it for better answers

## Usage
1. Select your preferred language (Wolof/English) for the UI
2. Choose a subject (Mathematics, Science, History, etc.)
3. Select input method: Text or Voice
4. Ask your question in Wolof
5. Get your answer in Wolof (with optional voice output)

## Troubleshooting
- **Torch version error:** Upgrade to torch 2.6+ or use Python 3.10/3.11
- **Voice not working:** Check microphone permissions and install portaudio
- **Model download issues:** Check internet connection and try again

## Deployment
Ready for deployment on Streamlit Cloud, Heroku, or any Python hosting platform. All dependencies are pinned for compatibility.
