# WolofChat Deployment Guide

WolofChat is designed to be deployed on multiple platforms. This guide covers deployment options for various hosting services.

## Quick Start

### Local Development
```bash
# Clone the repository
git clone <your-repo-url>
cd WollofEdu

# Install dependencies
pip install -r requirements.txt

# Run with Streamlit
streamlit run streamlit_app.py

# Or run with Flask
python flask_app.py

# Or run with FastAPI
python fastapi_app.py
```

## Prerequisites

### Environment Variables
Set these environment variables for full functionality:

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

## Deployment Platforms

### 1. Streamlit Cloud (Recommended for Beginners)

**Pros:**
- Easy deployment
- Free tier available
- Built-in support for Streamlit apps
- Automatic HTTPS

**Steps:**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set environment variables in the dashboard
5. Deploy!

**Configuration:**
- Uses `streamlit_app.py` as entry point
- No additional configuration needed

### 2. Heroku

**Pros:**
- Free tier available
- Easy deployment
- Good for small to medium apps

**Steps:**
1. Install Heroku CLI
2. Create Heroku app: `heroku create your-app-name`
3. Set environment variables:
   ```bash
   heroku config:set OPENAI_API_KEY=your_key
   heroku config:set SPEECH_RECOGNITION_ENABLED=true
   ```
4. Deploy: `git push heroku main`

**Configuration:**
- Uses `Procfile` for deployment
- Uses `runtime.txt` for Python version
- Uses `requirements.txt` for dependencies

### 3. Railway

**Pros:**
- Modern platform
- Easy deployment
- Good free tier
- Automatic deployments

**Steps:**
1. Connect your GitHub repository to Railway
2. Railway will auto-detect the Python app
3. Set environment variables in the dashboard
4. Deploy automatically

**Configuration:**
- Uses `app.json` for configuration
- Automatic Python detection

### 4. Vercel

**Pros:**
- Fast deployment
- Global CDN
- Good for static + API apps

**Steps:**
1. Install Vercel CLI: `npm i -g vercel`
2. Run: `vercel`
3. Set environment variables in dashboard
4. Deploy

**Configuration:**
- Uses `vercel.json` for configuration
- Supports Python functions

### 5. Docker Deployment

**Pros:**
- Consistent environment
- Easy scaling
- Works everywhere

**Steps:**
```bash
# Build image
docker build -t wolofchat .

# Run container
docker run -p 8501:8501 -e OPENAI_API_KEY=your_key wolofchat

# Or use docker-compose
docker-compose up
```

**Configuration:**
- Uses `Dockerfile` for containerization
- Uses `docker-compose.yml` for multi-service setup

### 6. AWS/GCP/Azure

**Pros:**
- Enterprise-grade
- Highly scalable
- Full control

**Steps:**
1. Choose your service (EC2, App Engine, etc.)
2. Follow platform-specific deployment guides
3. Set environment variables
4. Deploy

## Platform-Specific Configurations

### For Flask Apps
Use `flask_app.py` for platforms that prefer Flask:
- Heroku (with Flask buildpack)
- AWS Elastic Beanstalk
- Google App Engine

### For FastAPI Apps
Use `fastapi_app.py` for modern API deployment:
- Railway
- Vercel
- AWS Lambda
- Google Cloud Functions

### For Streamlit Apps
Use `streamlit_app.py` for data science platforms:
- Streamlit Cloud
- Hugging Face Spaces
- Local development

## Recommended Deployment Strategy

### For Beginners
1. **Start with Streamlit Cloud** - Easiest to deploy
2. **Use free tier** - No cost to get started
3. **Add OpenAI API key** - For best responses

### For Production
1. **Use Railway or Heroku** - Good balance of features
2. **Set up proper environment variables**
3. **Enable all services** - Full functionality
4. **Monitor usage** - Track API costs

### For Enterprise
1. **Use Docker** - Consistent deployment
2. **Deploy on AWS/GCP/Azure** - Scalable infrastructure
3. **Set up CI/CD** - Automated deployments
4. **Monitor and log** - Production monitoring

## Troubleshooting

### Common Issues

**Speech Recognition Not Working:**
- Check microphone permissions
- Install portaudio: `brew install portaudio` (macOS)
- Ensure PyAudio is installed correctly

**OpenAI API Errors:**
- Check API key is set correctly
- Verify API quota and billing
- Check network connectivity

**Ollama Not Available:**
- Ensure Ollama is running locally
- Check OLLAMA_URL environment variable
- Verify Ollama service is accessible

**Deployment Failures:**
- Check Python version compatibility
- Verify all dependencies in requirements.txt
- Check environment variables are set

### Platform-Specific Issues

**Heroku:**
- Check build logs for dependency issues
- Ensure Procfile is correct
- Verify environment variables

**Streamlit Cloud:**
- Check for import errors
- Verify file paths are correct
- Check environment variables in dashboard

**Docker:**
- Check Dockerfile syntax
- Verify port mappings
- Check container logs

## Monitoring and Maintenance

### Health Checks
- Use `/health` endpoint (FastAPI)
- Monitor service status
- Check API response times

### Logging
- Enable debug mode for development
- Monitor error logs
- Track API usage

### Updates
- Keep dependencies updated
- Monitor security patches
- Test new features locally first

## Success Metrics

Your deployment is successful when:
- App loads without errors
- All services show as available
- Questions return answers in Wolof
- Speech recognition works
- Text-to-speech generates audio
- Web search provides sources

## Support

For deployment issues:
1. Check the troubleshooting section
2. Review platform-specific documentation
3. Check GitHub issues
4. Contact platform support

---

**Happy Deploying!** 