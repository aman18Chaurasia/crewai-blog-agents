import streamlit as st

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="AI Summarizer",
    page_icon="🤖",
    layout="wide"
)

import requests
import re
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# Import check function
def check_and_import_transformers():
    """Check transformers installation and import."""
    try:
        # Try importing
        from transformers import pipeline, AutoTokenizer
        import transformers
        
        # Check version compatibility
        transformers_version = transformers.__version__
        
        try:
            import tokenizers
            tokenizers_version = tokenizers.__version__
        except ImportError:
            return None, None, "Tokenizers not installed"
        
        # Version compatibility check
        if transformers_version.startswith('4.35') and not tokenizers_version.startswith('0.14'):
            return None, None, f"Version mismatch: transformers {transformers_version} needs tokenizers 0.14.x, but found {tokenizers_version}"
        
        return pipeline, AutoTokenizer, None
        
    except ImportError as e:
        return None, None, f"Import failed: {str(e)}"

# Check other imports
def check_other_imports():
    """Check other required imports."""
    status = {}
    
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        status['youtube'] = True
    except ImportError:
        status['youtube'] = False
    
    try:
        from bs4 import BeautifulSoup
        status['bs4'] = True
    except ImportError:
        status['bs4'] = False
    
    try:
        import nltk
        status['nltk'] = True
    except ImportError:
        status['nltk'] = False
    
    return status

# Check imports
pipeline_func, AutoTokenizer, transformers_error = check_and_import_transformers()
import_status = check_other_imports()

class FixedSummarizer:
    def __init__(self):
        self.model_loaded = False
        self.summarizer = None
        self.model_name = None
        
    def load_model(self, model_name="sshleifer/distilbart-cnn-12-6"):
        """Load model with proper error handling."""
        if not pipeline_func:
            return False, "Transformers not available"
            
        try:
            st.info(f"🤖 Loading model: {model_name}")
            
            # Load model
            self.summarizer = pipeline_func(
                "summarization",
                model=model_name,
                device=-1,  # Force CPU
                clean_up_tokenization_spaces=True
            )
            
            self.model_loaded = True
            self.model_name = model_name
            return True, f"✅ Model {model_name} loaded successfully!"
            
        except Exception as e:
            error_msg = str(e)
            
            # Try fallback models
            fallbacks = [
                "sshleifer/distilbart-cnn-12-6",
                "facebook/bart-large-cnn", 
                "t5-base"
            ]
            
            for fallback in fallbacks:
                if fallback != model_name:
                    try:
                        st.info(f"Trying fallback: {fallback}")
                        self.summarizer = pipeline_func(
                            "summarization",
                            model=fallback,
                            device=-1
                        )
                        self.model_loaded = True
                        self.model_name = fallback
                        return True, f"✅ Loaded fallback model: {fallback}"
                    except:
                        continue
            
            return False, f"❌ All models failed: {error_msg}"
    
    def extract_youtube_id(self, url):
        """Extract YouTube video ID."""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def get_youtube_transcript(self, url):
        """Get YouTube transcript."""
        if not import_status['youtube']:
            return None, "YouTube Transcript API not available"
            
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            
            video_id = self.extract_youtube_id(url)
            if not video_id:
                return None, "Invalid YouTube URL"
            
            # Get available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try English first
            try:
                transcript = transcript_list.find_transcript(['en'])
            except:
                # Get first available transcript
                available = list(transcript_list._transcript_dict.keys())
                if available:
                    transcript = transcript_list.find_transcript([available[0]])
                else:
                    return None, "No transcripts available"
            
            # Fetch transcript data
            transcript_data = transcript.fetch()
            text = " ".join([item['text'] for item in transcript_data])
            
            return text, "Success"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def scrape_website(self, url): 
        """Scrape website content."""
        if not import_status['bs4']:
            return None, "BeautifulSoup not available"
            
        try:
            from bs4 import BeautifulSoup
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                element.extract()
            
            # Try to find main content
            content_selectors = ['main', 'article', '.content', '.post', '.entry-content']
            main_content = None
            
            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if not main_content:
                main_content = soup
            
            # Extract and clean text
            text = main_content.get_text()
            lines = (line.strip() for line in text.splitlines())
            clean_text = ' '.join(line for line in lines if line)
            
            return clean_text, "Success"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def summarize_text(self, text):
        """Summarize text."""
        if not self.model_loaded:
            return "❌ Model not loaded"
            
        try:
            # Limit text length for stability
            if len(text) > 2000:  # Conservative limit
                text = text[:2000]
            
            # Generate summary
            result = self.summarizer(
                text,
                max_length=130,
                min_length=30,
                do_sample=False,
                truncation=True
            )
            
            return result[0]['summary_text']
            
        except Exception as e:
            return f"❌ Summarization failed: {str(e)}"

def main():
    # Title and description
    st.title("🤖 AI Content Summarizer")
    st.markdown("### Fixed Version - Handles Installation Issues")
    
    # Show dependency status
    st.markdown("### 🔍 System Check")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if pipeline_func:
            st.success("✅ Transformers")
        else:
            st.error("❌ Transformers")
    
    with col2:
        if import_status['youtube']:
            st.success("✅ YouTube API")
        else:
            st.error("❌ YouTube API")
    
    with col3:
        if import_status['bs4']:
            st.success("✅ BeautifulSoup")
        else:
            st.error("❌ BeautifulSoup")
    
    with col4:
        if import_status['nltk']:
            st.success("✅ NLTK")
        else:
            st.warning("⚠️ NLTK")
    
    # Show transformers error if exists
    if transformers_error:
        st.error(f"🚨 Transformers Error: {transformers_error}")
        
        st.markdown("### 🔧 Fix Command:")
        st.code("""
# Run this exact command:
pip uninstall transformers tokenizers -y
pip install transformers==4.35.0 tokenizers==0.14.2
        """)
        
        if st.button("🔄 Refresh Page After Fix"):
            st.rerun()
        
        st.stop()
    
    # Initialize summarizer
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = FixedSummarizer()
        st.session_state.model_loaded = False
    
    # Model loading section
    st.markdown("### 🤖 AI Model")
    
    model_options = {
        "DistilBART (Recommended)": "sshleifer/distilbart-cnn-12-6",
        "BART Large (Better Quality)": "facebook/bart-large-cnn",
        "T5 Base (Alternative)": "t5-base"
    }
    
    selected_model = st.selectbox(
        "Choose Model:",
        options=list(model_options.keys()),
        help="DistilBART is fastest and most reliable"
    )
    
    model_name = model_options[selected_model]
    
    # Load model button
    if not st.session_state.model_loaded:
        if st.button("🚀 Load Model", type="primary"):
            with st.spinner("Loading AI model..."):
                success, message = st.session_state.summarizer.load_model(model_name)
                
            if success:
                st.success(message)
                st.session_state.model_loaded = True
                st.rerun()
            else:
                st.error(message)
    else:
        st.success(f"✅ Model loaded: {st.session_state.summarizer.model_name}")
        
        # Main functionality
        st.markdown("### 🔗 Summarize Content")
        
        # URL input
        url = st.text_input(
            "Paste URL here:",
            placeholder="https://youtube.com/watch?v=... or https://example.com/article",
            help="YouTube videos need captions/transcripts"
        )
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            summarize_btn = st.button("📝 Summarize", type="primary")
        
        with col2:
            example_btn = st.button("📋 Load Example")
        
        with col3:
            clear_btn = st.button("🗑️ Clear")
        
        # Handle example button
        if example_btn:
            url = "https://www.youtube.com/watch?v=kJQP7kiw5Fk"
            st.rerun()
        
        # Handle clear button  
        if clear_btn:
            if 'last_result' in st.session_state:
                del st.session_state.last_result
            st.rerun()
        
        # Handle summarize button
        if summarize_btn and url:
            if not url.startswith(('http://', 'https://')):
                st.error("❌ Please enter a valid URL")
            else:
                # Process based on URL type
                if "youtube.com" in url or "youtu.be" in url:
                    process_youtube_url(url)
                else:
                    process_website_url(url)
        
        elif summarize_btn:
            st.warning("⚠️ Please enter a URL first")
        
        # Show previous result
        if 'last_result' in st.session_state:
            show_result(st.session_state.last_result)
    
    # Instructions
    st.markdown("---")
    st.markdown("### 💡 How to Use")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎥 YouTube Videos:**
        - Must have captions/transcripts
        - Works with most popular videos
        - Auto-generated captions OK
        
        **Example:**
        `https://youtube.com/watch?v=kJQP7kiw5Fk`
        """)
    
    with col2:
        st.markdown("""
        **🌐 Websites:**
        - News articles work best
        - Blog posts and documentation
        - May not work with some sites
        
        **Example:**
        `https://blog.google/technology/ai/`
        """)

def process_youtube_url(url):
    """Process YouTube URL."""
    with st.spinner("🎥 Getting YouTube transcript..."):
        text, status = st.session_state.summarizer.get_youtube_transcript(url)
    
    if text:
        st.success(f"✅ Transcript: {len(text):,} characters")
        
        with st.spinner("🤖 Generating summary..."):
            summary = st.session_state.summarizer.summarize_text(text)
        
        result = {
            'success': True,
            'type': 'YouTube Video',
            'url': url,
            'text_length': len(text),
            'word_count': len(text.split()),
            'summary': summary
        }
        
        st.session_state.last_result = result
        show_result(result)
        
    else:
        st.error(f"❌ Failed: {status}")
        result = {
            'success': False,
            'type': 'YouTube Video', 
            'url': url,
            'error': status
        }
        st.session_state.last_result = result

def process_website_url(url):
    """Process website URL."""
    with st.spinner("🌐 Scraping website..."):
        text, status = st.session_state.summarizer.scrape_website(url)
    
    if text:
        st.success(f"✅ Content: {len(text):,} characters")
        
        with st.spinner("🤖 Generating summary..."):
            summary = st.session_state.summarizer.summarize_text(text)
        
        result = {
            'success': True,
            'type': 'Website',
            'url': url,
            'text_length': len(text),
            'word_count': len(text.split()),
            'summary': summary
        }
        
        st.session_state.last_result = result
        show_result(result)
        
    else:
        st.error(f"❌ Failed: {status}")
        result = {
            'success': False,
            'type': 'Website',
            'url': url, 
            'error': status
        }
        st.session_state.last_result = result

def show_result(result):
    """Display summarization result."""
    st.markdown("### 📋 Summary Result")
    
    if result['success']:
        # Success case
        st.markdown(f"**🔗 Source:** {result['url']}")
        st.markdown(f"**📊 Type:** {result['type']}")
        
        # Show summary
        st.markdown("**📝 Summary:**")
        st.markdown(f"""
        <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50; margin: 10px 0;'>
            {result['summary']}
        </div>
        """, unsafe_allow_html=True)
        
        # Show stats
        with st.expander("📊 Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Characters", f"{result['text_length']:,}")
                st.metric("Words", f"{result['word_count']:,}")
            with col2:
                st.metric("Model", st.session_state.summarizer.model_name.split('/')[-1])
                st.metric("Type", result['type'])
    
    else:
        # Error case
        st.error(f"❌ Failed to process {result['type']}")
        st.error(f"URL: {result['url']}")
        st.error(f"Error: {result['error']}")
        
        # Show troubleshooting
        with st.expander("🔧 Troubleshooting"):
            if result['type'] == 'YouTube Video':
                st.markdown("""
                **YouTube Issues:**
                - Video must have captions/transcripts
                - Try a different popular video
                - Check if video is public
                """)
            else:
                st.markdown("""
                **Website Issues:**
                - Some sites block scraping
                - Try a news article URL
                - Check if URL is accessible
                """)

if __name__ == "__main__":
    main()