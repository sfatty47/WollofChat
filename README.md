# WolofChat - Wolof Educational Q&A App

A comprehensive educational chatbot for Wolof language and culture, designed for deployment on multiple platforms.

## Features

- **AI-Powered Responses** - OpenAI GPT, Ollama Mistral, and Hugging Face integration
- **Speech Recognition** - Voice input in Wolof and English
- **Text-to-Speech** - Listen to answers in Wolof
- **Multi-Language Support** - Wolof and English input/output
- **Web Search Integration** - Real-time information from the web
- **Educational Content** - History, Math, Science, Geography, Technology, Literature
- **Multi-Platform Deployment** - Streamlit, Flask, FastAPI, Docker support

## Quick Start

### Local Development
```bash
# Clone the repository
git clone <your-repo-url>
cd WollofEdu

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your_openai_api_key"

# Run the app
streamlit run streamlit_app.py
```

### Docker Deployment
```bash
# Build and run with Docker
docker build -t wolofchat .
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key wolofchat

# Or use docker-compose (includes Ollama)
docker-compose up
```

## Deployment Options

### Streamlit Cloud (Recommended for Beginners)
- Push to GitHub → Deploy on [share.streamlit.io](https://share.streamlit.io)
- Free tier available
- Automatic HTTPS

### Heroku
- Use `Procfile` and `runtime.txt`
- Free tier available
- Easy deployment

### Railway
- Modern platform with good free tier
- Automatic deployments
- Uses `app.json` configuration

### Vercel
- Fast deployment with global CDN
- Uses `vercel.json` configuration
- Good for API deployment

### Docker
- Consistent environment
- Works everywhere
- Includes Ollama service

## Configuration

### Environment Variables
```bash
# LLM Services
OPENAI_API_KEY=your_openai_api_key
OLLAMA_URL=http://localhost:11434
HUGGINGFACE_TOKEN=your_huggingface_token

# Features
SPEECH_RECOGNITION_ENABLED=true
TTS_ENABLED=true
WEB_SEARCH_ENABLED=true
DEBUG_MODE=false

# App Configuration
MAX_TOKENS=1000
TEMPERATURE=0.7
```

### Service Priority
1. **OpenAI GPT** (highest quality)
2. **Ollama Mistral** (local)
3. **Hugging Face** (cloud)
4. **Knowledge Base** (fallback)

## Project Structure

```
WollofEdu/
├── app.py                 # Core WolofChat application
├── streamlit_app.py       # Streamlit interface
├── flask_app.py          # Flask API
├── fastapi_app.py        # FastAPI interface
├── requirements.txt      # Python dependencies
├── Procfile             # Heroku deployment
├── dockerfile           # Docker configuration
├── docker-compose.yml   # Multi-service setup
├── app.json             # Railway configuration
├── vercel.json          # Vercel configuration
├── runtime.txt          # Python version
└── DEPLOYMENT_GUIDE.md  # Detailed deployment guide
```

## Usage Examples

### Text Questions
- **Wolof**: "Kan moi Mansa Musa?" (Who is King Mansa Musa?)
- **English**: "Who was Sundiata Keita?"
- **Math**: "Lan la 2 + 3?" (What is 2 + 3?)

### Voice Input
- Click "Click to Speak"
- Ask questions in Wolof or English
- Get real-time speech recognition

### Voice Output
- Click "Listen to Answer"
- Hear responses in Wolof
- Natural pronunciation

## API Endpoints (FastAPI/Flask)

```bash
# Get service status
GET /api/status

# Ask a question
POST /api/ask
{
  "question": "Kan moi Mansa Musa?",
  "subject": "History",
  "language": "Wolof"
}

# Speech to text
POST /api/speech-to-text

# Text to speech
POST /api/text-to-speech
{
  "text": "Mansa Musa mooy ab mansa bu gudd",
  "language": "wo"
}
```

## Development

### Running Different Versions
```bash
# Streamlit (default)
streamlit run streamlit_app.py

# Flask API
python flask_app.py

# FastAPI
python fastapi_app.py

# Core app only
python app.py
```

### Testing
```bash
# Install test dependencies
pip install pytest

# Run tests
pytest
```

## Service Status

The app shows real-time status of all services:
- **OpenAI GPT** - Premium responses
- **Ollama Mistral** - Local AI
- **Speech Recognition** - Voice input
- **Text-to-Speech** - Voice output
- **Web Search** - Real-time information

## Supported Languages

- **Wolof** - Primary language with cultural context
- **English** - Secondary language for accessibility

## Educational Subjects

- **History** - African history, Mali Empire, notable figures
- **Mathematics** - Basic operations, concepts, problem-solving
- **Science** - Natural world, physics, chemistry, biology
- **Geography** - Africa, countries, physical features
- **Technology** - Computers, internet, modern tools
- **Literature** - Wolof literature, storytelling, culture
- **General** - Everyday topics, greetings, conversation

## Troubleshooting

### Common Issues
1. **Speech Recognition**: Install portaudio (`brew install portaudio`)
2. **OpenAI Errors**: Check API key and quota
3. **Ollama Issues**: Ensure Ollama is running locally
4. **Deployment Failures**: Check environment variables

### Platform-Specific Help
- See `DEPLOYMENT_GUIDE.md` for detailed instructions
- Check platform documentation
- Review error logs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Wolof language community
- OpenAI for GPT integration
- Ollama for local AI
- Google for speech recognition
- Streamlit for the web framework

---

**Jàngat ak Laaj! (Learn and Ask!)**
