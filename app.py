import streamlit as st
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
import speech_recognition as sr
from gtts import gTTS
import google.generativeai as genai
import os
import time
import base64
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="PDF Voice Assistant",
    page_icon="🎤📚",
    layout="wide"
)

# Initialize session state
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False
if 'embedder' not in st.session_state:
    st.session_state.embedder = None
if 'current_audio_key' not in st.session_state:
    st.session_state.current_audio_key = 0

# Load embedder (cached)
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# PDF extraction function
def extract_pdf_text(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        
        # Handle encrypted PDFs
        if pdf_reader.is_encrypted:
            try:
                pdf_reader.decrypt("")
            except Exception:
                st.error("❌ PDF is encrypted and requires a password.")
                return ""
        
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
        
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

# Build ChromaDB index
def build_chroma_index(text, collection_name="pdf_collection", chunk_size=1000, overlap=200):
    client = chromadb.Client()
    
    # Reset collection if it exists
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    collection = client.create_collection(name=collection_name)
    
    # Split into chunks with overlap
    chunks, start, ids = [], 0, []
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        ids.append(str(len(ids)))
        start += chunk_size - overlap
    
    # Embed chunks
    embeddings = st.session_state.embedder.encode(chunks).tolist()
    
    # Add to Chroma
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )
    
    return collection, chunks

# Retrieval function
def retrieve_relevant_chunks(question, collection, top_k=3):
    query_variants = [
        question,
        f"Explain: {question}",
        f"What does the document say about {question}?",
        f"Details related to {question}"
    ]
    
    retrieved = []
    for q in query_variants:
        q_emb = st.session_state.embedder.encode([q]).tolist()[0]
        results = collection.query(query_embeddings=[q_emb], n_results=top_k)
        retrieved.extend(results["documents"][0])
    
    # Deduplicate
    final_chunks = []
    seen = set()
    for ch in retrieved:
        if ch not in seen:
            final_chunks.append(ch)
            seen.add(ch)
    
    return "\n".join(final_chunks[:top_k])

# Query Gemini function
def query_gemini(question, collection, model="gemini-1.5-flash", top_k=4):
    try:
        # Retrieve relevant chunks from Chroma
        context = retrieve_relevant_chunks(question, collection, top_k=top_k)
        
        # Build prompt
        prompt = f"""Based on the following document excerpts, answer the question accurately:

        Document Context:
        {context}

        Question: {question}

        Provide a clear and concise answer:"""
        
        # Call Gemini model
        response = genai.GenerativeModel(model).generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 300
            }
        )
        
        if response and response.candidates:
            return response.candidates[0].content.parts[0].text.strip()
        else:
            return "No response generated"
    
    except Exception as e:
        return f"Error querying Gemini: {str(e)}"

# Speech recognition function
def speech_to_text():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    try:
        with microphone as source:
            st.info("🎤 Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source)
            st.info("🎤 Listening... Speak now!")
            
            # Listen for speech
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            
        st.info("🔄 Processing speech...")
        text = recognizer.recognize_google(audio)
        return text
    
    except sr.WaitTimeoutError:
        return "Timeout: No speech detected"
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Speech recognition error: {e}"
    except Exception as e:
        return f"Error: {str(e)}"

# Text to speech function
def text_to_speech(text, lang='en'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        # Increment audio key to force new audio generation
        st.session_state.current_audio_key += 1
        return fp
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

# Create audio player with unique key
def create_audio_player(audio_bytes, key=None):
    audio_bytes.seek(0)  # Reset position to beginning
    b64 = base64.b64encode(audio_bytes.read()).decode()
    # Use current timestamp to ensure unique audio element
    unique_id = f"audio_{st.session_state.current_audio_key}_{int(time.time() * 1000)}"
    audio_html = f"""
    <audio id="{unique_id}" controls autoplay key="{key}">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    <script>
        // Force reload of audio element
        document.getElementById('{unique_id}').load();
    </script>
    """
    return audio_html

# Main app
def main():
    # Configure Gemini API from environment variable
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        genai.configure(api_key=api_key)
    
    # Load embedder
    if st.session_state.embedder is None:
        with st.spinner("Loading embedding model..."):
            st.session_state.embedder = load_embedder()
    
    # Title and header
    st.title("🎤📚 PDF Voice Assistant")
    st.markdown("Upload a PDF and ask questions using your voice!")
    
    # Sidebar for status
    with st.sidebar:
        st.header("📊 Status")
        
        if api_key:
            st.success("✅ Gemini API configured")
        else:
            st.error("❌ Gemini API key not found in .env file")
        
        if st.session_state.pdf_processed:
            st.success("✅ PDF processed and ready")
        else:
            st.info("ℹ️ Upload a PDF to get started")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 PDF Upload")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            if st.button("📚 Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    # Extract text
                    pdf_text = extract_pdf_text(uploaded_file)
                    
                    if pdf_text:
                        # Build index
                        collection, chunks = build_chroma_index(pdf_text)
                        st.session_state.collection = collection
                        st.session_state.pdf_processed = True
                        
                        st.success(f"✅ PDF processed! Created {len(chunks)} text chunks.")
                        
                        # Show preview
                        with st.expander("📖 Preview extracted text"):
                            st.text_area("First 500 characters:", 
                                       value=pdf_text[:500], height=150)
                    else:
                        st.error("❌ Failed to extract text from PDF")
    
    with col2:
        st.header("🎤 Voice Interaction")
        
        if not api_key:
            st.warning("⚠️ Please add GEMINI_API_KEY to your .env file")
        elif not st.session_state.pdf_processed:
            st.warning("⚠️ Please upload and process a PDF first")
        else:
            # Voice input section
            st.subheader("🗣️ Ask a Question")
            
            col2a, col2b = st.columns([1, 1])
            
            with col2a:
                if st.button("🎤 Start Voice Input", type="primary"):
                    with st.spinner("Listening..."):
                        question = speech_to_text()
                        
                    if question and not question.startswith(("Timeout", "Could not", "Speech recognition error", "Error")):
                        st.success(f"🗣️ You asked: {question}")
                        
                        # Get answer
                        with st.spinner("Getting answer..."):
                            answer = query_gemini(question, st.session_state.collection)
                        
                        st.subheader("📝 Answer:")
                        st.write(answer)
                        
                        # Generate speech with English language (hardcoded)
                        with st.spinner("Generating speech..."):
                            audio_fp = text_to_speech(answer, 'en')
                        
                        if audio_fp:
                            st.subheader("🔊 Listen to Answer:")
                            # Use unique key based on current audio key
                            audio_html = create_audio_player(audio_fp, key=f"voice_audio_{st.session_state.current_audio_key}")
                            st.markdown(audio_html, unsafe_allow_html=True)
                    else:
                        st.error(f"❌ {question}")
            
            with col2b:
                # Manual text input as alternative
                st.subheader("⌨️ Or Type Your Question")
                manual_question = st.text_input("Enter your question:")
                
                if st.button("📤 Submit Question") and manual_question:
                    with st.spinner("Getting answer..."):
                        answer = query_gemini(manual_question, st.session_state.collection)
                    
                    st.subheader("📝 Answer:")
                    st.write(answer)
                    
                    # Generate speech with English language (hardcoded)
                    with st.spinner("Generating speech..."):
                        audio_fp = text_to_speech(answer, 'en')
                    
                    if audio_fp:
                        st.subheader("🔊 Listen to Answer:")
                        # Use unique key based on current audio key
                        audio_html = create_audio_player(audio_fp, key=f"text_audio_{st.session_state.current_audio_key}")
                        st.markdown(audio_html, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("### 📋 Instructions:")
    st.markdown("""
    1. **Upload a PDF** and click 'Process PDF'
    2. **Click 'Start Voice Input'** and speak your question
    3. **Listen to the answer** or read the text response
    4. You can also type questions manually as an alternative
    """)
    
    # Usage tips
    with st.expander("💡 Usage Tips"):
        st.markdown("""
        - **Speak clearly** and avoid background noise
        - **Keep questions concise** and specific
        - **Wait for the 'Listening...' indicator** before speaking
        - **Voice output** is in English by default
        """)

if __name__ == "__main__":
    main()