# 🎤📚 PDF Voice Assistant

A Streamlit-based application that allows you to upload PDF documents and interact with them using voice commands. Ask questions about your PDF content and receive both text and audio responses powered by Google's Gemini AI.

## 🎥 Demo Video

<!-- Add your project implementation video here -->
[![PDF Voice Assistant Demo](https://img.shields.io/badge/▶️%20Watch-Demo%20Video-red?style=for-the-badge&logo=youtube)](

https://github.com/user-attachments/assets/b7b1a5f4-5bcb-4add-b337-23d39adbca6a

)

*Click the badge above to watch the full implementation and usage demo*

---

## ✨ Features

- **PDF Text Extraction**: Upload and process PDF documents with automatic text extraction
- **Voice Input**: Ask questions using your microphone with speech-to-text conversion
- **AI-Powered Responses**: Get intelligent answers using Google's Gemini AI model
- **Text-to-Speech**: Listen to answers with AI-generated voice responses
- **Semantic Search**: Advanced document retrieval using ChromaDB and sentence transformers
- **Dual Input Methods**: Support for both voice and text input
- **Real-time Processing**: Live status updates and interactive interface

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **Google Gemini AI**: Large language model for question answering
- **ChromaDB**: Vector database for document storage and retrieval
- **SentenceTransformers**: Text embedding for semantic search
- **PyPDF2**: PDF text extraction
- **SpeechRecognition**: Voice-to-text conversion
- **gTTS (Google Text-to-Speech)**: Text-to-voice conversion
- **Python-dotenv**: Environment variable management

## 📋 Prerequisites

- Python 3.7 or higher
- Microphone access for voice input
- Internet connection for AI services
- Google Gemini API key

## 🚀 Installation

1. **Clone or download the repository**
   ```bash
   git clone https://github.com/AbhishekDongre14/pdf-voice-assistant.git
   cd pdf-voice-assistant
   ```

2. **Install required dependencies**
   ```bash
   pip install streamlit PyPDF2 chromadb sentence-transformers speechrecognition gtts google-generativeai python-dotenv
   ```

3. **Install additional system dependencies** (for speech recognition)
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install python3-pyaudio portaudio19-dev

   # On macOS
   brew install portaudio

   # On Windows
   pip install pyaudio
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

5. **Get your Gemini API key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add it to your `.env` file

## 🎯 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Upload a PDF**
   - Click "Choose a PDF file" in the left column
   - Select your PDF document
   - Click "Process PDF" to extract and index the text

3. **Ask questions using voice**
   - Click "Start Voice Input" button
   - Wait for the "Listening..." indicator
   - Speak your question clearly
   - Wait for the AI response and audio playback

4. **Alternative text input**
   - Type your question in the text input field
   - Click "Submit Question" for processing

## 🎛️ Configuration Options

### Chunking Parameters
- **Chunk Size**: 1000 characters (configurable in `build_chroma_index`)
- **Overlap**: 200 characters for context preservation
- **Top-K Retrieval**: 3-4 most relevant chunks

### AI Model Settings
- **Model**: Gemini-1.5-Flash (configurable in `query_gemini`)
- **Temperature**: 0.3 for consistent responses
- **Max Output Tokens**: 300 for concise answers

### Audio Settings
- **TTS Language**: English (hardcoded as 'en')
- **Speech Timeout**: 10 seconds
- **Phrase Time Limit**: 10 seconds

## 📁 Project Structure

```
pdf-voice-assistant/
├── app.py                 # Main application file
├── .env                   # Environment variables (create this)
├── requirements.txt       # Python dependencies (optional)
└── README.md             # This file
```

## 🔧 Key Functions

### Core Components

- **`extract_pdf_text()`**: Extracts text from PDF files with encryption handling
- **`build_chroma_index()`**: Creates vector embeddings and stores in ChromaDB
- **`retrieve_relevant_chunks()`**: Semantic search with query variants
- **`query_gemini()`**: Integrates with Google's Gemini AI for responses
- **`speech_to_text()`**: Converts voice input to text using Google Speech Recognition
- **`text_to_speech()`**: Generates audio responses using Google TTS

### Session Management
- Maintains PDF processing state
- Handles audio playback with unique keys
- Caches embedding model for performance

## 🚨 Troubleshooting

### Common Issues

1. **"Gemini API key not found"**
   - Ensure `.env` file exists with correct `GEMINI_API_KEY`
   - Verify API key is valid and has proper permissions

2. **"Could not understand audio"**
   - Speak more clearly and reduce background noise
   - Check microphone permissions and functionality
   - Ensure stable internet connection

3. **"Error reading PDF"**
   - Try a different PDF file
   - Ensure PDF is not corrupted or heavily encrypted
   - Check file size limitations

4. **Audio playback issues**
   - Refresh the browser page
   - Check browser audio permissions
   - Try a different browser (Chrome recommended)

### Performance Tips

- Use PDFs with clear, extractable text (avoid scanned images)
- Keep questions concise and specific
- Process smaller PDF files for faster performance
- Ensure stable internet connection for AI services

## 📊 System Requirements

- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space for models and cache
- **Network**: Stable internet for AI API calls
- **Browser**: Modern browser with audio support (Chrome, Firefox, Safari)

## 🔒 Privacy & Security

- PDF content is processed locally for text extraction
- Voice data is sent to Google's speech recognition service
- Text queries are sent to Google's Gemini AI service
- No data is permanently stored on external servers
- API keys are loaded from local environment variables

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Review Google AI Studio documentation for API issues
- Ensure all dependencies are properly installed
- Verify microphone and audio permissions in your browser

## 🔮 Future Enhancements

- Support for multiple languages in TTS
- Batch PDF processing
- Chat history and conversation memory
- Export functionality for Q&A sessions
- Advanced PDF parsing for tables and images
- Custom voice models and accents
