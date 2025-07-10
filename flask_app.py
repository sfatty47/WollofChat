#!/usr/bin/env python3
"""
Flask version of WolofChat for deployment on various platforms
"""

from flask import Flask, render_template, request, jsonify, send_file
from app import WolofChatApp, WolofChatConfig
import tempfile
import os
import speech_recognition as sr
import io
import base64

app = Flask(__name__)

# Initialize WolofChat
wolofchat = WolofChatApp()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get service status"""
    return jsonify(wolofchat.get_service_status())

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Ask a question and get answer"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        subject = data.get('subject', 'General')
        question_language = data.get('language', 'Wolof')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        answer, sources = wolofchat.find_answer(question, subject, question_language)
        
        return jsonify({
            'question': question,
            'answer': answer,
            'sources': sources,
            'subject': subject,
            'language': question_language
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/speech-to-text', methods=['POST'])
def speech_to_text():
    """Convert speech to text"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        language = request.form.get('language', 'Wolof')
        
        # Convert audio file to AudioData
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        
        text = wolofchat.process_voice_input(audio_data, language)
        
        if text:
            return jsonify({'text': text})
        else:
            return jsonify({'error': 'Could not recognize speech'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    """Convert text to speech"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        language = data.get('language', 'wo')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        audio_file_path = wolofchat.generate_voice_output(text, language)
        
        if audio_file_path:
            return send_file(audio_file_path, mimetype='audio/mp3')
        else:
            return jsonify({'error': 'Could not generate speech'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/record-audio', methods=['POST'])
def record_audio():
    """Record audio from microphone"""
    try:
        if not wolofchat.speech_service.is_available():
            return jsonify({'error': 'Speech recognition not available'}), 400
        
        audio_data = wolofchat.speech_service.record_audio()
        
        if audio_data:
            # Convert AudioData to base64 for transmission
            # This is a simplified version - in production you'd want to handle this differently
            return jsonify({'success': True, 'message': 'Audio recorded successfully'})
        else:
            return jsonify({'error': 'Could not record audio'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG_MODE', 'false').lower() == 'true') 