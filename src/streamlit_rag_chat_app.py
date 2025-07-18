import streamlit as st
import sys
import os
import re
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
import zipfile
import requests

# Add the current directory to Python path to import rag_vf
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_vf import ChemistryRAG
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.media import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env")

# Available models for selection
AVAILABLE_MODELS = {
    "Qwen2.5-VL-7B": "qwen/qwen-2.5-vl-7b-instruct",
    "Qwen2.5-VL-14B": "qwen/qwen-2.5-vl-14b-instruct", 
    "Gemini Pro 1.5": "google/gemini-pro-1.5",
    "Gemini Flash": "google/gemini-flash-1.5",
    "GPT-4o": "openai/gpt-4o",
    "GPT-4o Mini": "openai/gpt-4o-mini",
    "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
    "Claude 3.5 Haiku": "anthropic/claude-3.5-haiku"
}

def extract_video_id_and_timestamp(youtube_url):
    """Extract video ID and timestamp from YouTube URL."""
    if not youtube_url:
        return None, None
    
    # Extract video ID
    video_id_match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]+)', youtube_url)
    if not video_id_match:
        return None, None
    
    video_id = video_id_match.group(1)
    
    # Extract timestamp
    timestamp_match = re.search(r't=(\d+)s', youtube_url)
    timestamp = int(timestamp_match.group(1)) if timestamp_match else 0
    
    return video_id, timestamp

def create_youtube_embed_url(video_id, timestamp=0):
    """Create YouTube embed URL with timestamp."""
    return f"https://www.youtube.com/embed/{video_id}?start={timestamp}"

def filter_neighboring_timestamps(items, time_window=60):
    """
    Filter out neighboring timestamps within a time window.
    Returns items sorted by score, keeping only non-neighboring ones.
    """
    if not items:
        return []
    
    # Sort by score (descending)
    sorted_items = sorted(items, key=lambda x: x['score'] if x['score'] else 0, reverse=True)
    
    filtered_items = []
    for item in sorted_items:
        quadruplet = item['quadruplet']
        current_timestamp = quadruplet['timestamp']
        
        # Check if this timestamp is too close to any already selected timestamp
        is_neighbor = False
        for selected_item in filtered_items:
            selected_timestamp = selected_item['quadruplet']['timestamp']
            if abs(current_timestamp - selected_timestamp) < time_window:
                is_neighbor = True
                break
        
        if not is_neighbor:
            filtered_items.append(item)
    
    return filtered_items

def group_and_filter_results(retrievals, max_per_speaker=3, time_window=60):
    """
    Group results by speaker and return top 3 non-neighboring timestamps per speaker.
    """
    # Group by speaker
    speaker_groups = defaultdict(list)
    for item in retrievals:
        speaker = item['quadruplet']['speaker']
        speaker_groups[speaker].append(item)
    
    # Filter each speaker's results
    filtered_results = []
    for speaker, items in speaker_groups.items():
        # Filter out neighboring timestamps and take top 3
        filtered_items = filter_neighboring_timestamps(items, time_window)[:max_per_speaker]
        filtered_results.extend(filtered_items)
    
    return filtered_results

def create_agent(model_id: str, api_key: str) -> Agent:
    """Create an agent with the specified model and API key."""
    try:
        # Set the API key in environment
        os.environ["OPENROUTER_API_KEY"] = api_key
        
        agent = Agent(
            model=OpenRouter(id=model_id),
            markdown=True
        )
        return agent
    except Exception as e:
        st.error(f"Failed to create agent: {e}")
        return None

def generate_chat_response(agent: Agent, question: str, retrieved_slides: List[Dict], chat_history: List[Dict]) -> str:
    """Generate a response using the agent based on retrieved slides and chat history."""
    if not agent:
        return "Error: Agent not initialized properly."
    
    try:
        # Build context from retrieved slides
        context = "Based on the following slide images and captions:\n\n"
        images = []
        
        for i, slide in enumerate(retrieved_slides):
            quadruplet = slide['quadruplet']
            context += f"Slide {i+1}: {quadruplet['caption']} <|image|>"
            images.append(Image(filepath=quadruplet['img_path']))
        
        # Build the full prompt
        system_prompt = (
            "You are an expert chemistry tutor. You have access to slide images and captions from chemistry lectures. "
            "Answer questions based on the provided content. Be helpful, accurate, and concise."
        )
        
        # Include chat history for context
        conversation_history = ""
        if chat_history:
            conversation_history = "\n\nPrevious conversation:\n"
            for msg in chat_history[-3:]:  # Last 3 messages for context
                conversation_history += f"{msg['role']}: {msg['content']}\n"
        
        full_prompt = f"{system_prompt}\n\n{context}\n\n{conversation_history}\n\nUser: {question}\n\nAssistant:"
        
        # Generate response with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = agent.run(full_prompt, images=images)
                content = response.content
                
                if not content or content.strip() == "":
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    else:
                        return "I apologize, but I couldn't generate a response. Please try again."
                
                return content
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return f"Error generating response: {str(e)}"
                    
    except Exception as e:
        return f"Error: {str(e)}"

def download_and_extract_indexes(url, extract_to='saved_indexes'):
    zip_path = 'saved_indexes.zip'
    if not os.path.exists(extract_to):
        print('Downloading saved_indexes...')

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        # Get Hugging Face token from secrets if available
        hf_token = st.secrets.get("HF_TOKEN", None)
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        # Download the ZIP file
        r = requests.get(url, headers=headers)

        # Safety check: verify it's a ZIP
        if r.headers.get("Content-Type") != "application/zip":
            raise ValueError("Downloaded content is not a ZIP file. Check the URL or token.zip")

        with open(zip_path, 'wb') as f:
            f.write(r.content)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')

        os.remove(zip_path)
        print('saved_indexes ready!')


# Replace with your actual Hugging Face download link:
download_and_extract_indexes('https://huggingface.co/datasets/ines-epfl-ethz/SW4retrieval/resolve/main/saved_indexes.zip')


def main():
    st.set_page_config(
        page_title="Chemistry RAG Chat with YouTube Videos",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Chemistry RAG Agent with YouTube Videos")
    st.markdown("Ask questions about chemistry and chat with an AI about the retrieved video content!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Provider selection
        provider = st.selectbox(
            "Select Provider:",
            ["OpenAI", "OpenRouter"]
        )
        
        # API key input based on provider
        import os
        api_key = ""
        if provider == "OpenAI":
            api_key = st.text_input(
                "OpenAI API Key:",
                type="password",
                help="Enter your OpenAI API key if using OpenAI models"
            )
        else:
            api_key = st.text_input(
                "OpenRouter API Key:",
                type="password",
                help="Enter your OpenRouter API key to use OpenRouter models"
            )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENROUTER_API_KEY"] = api_key
        
        # Model selection based on provider
        OPENAI_MODELS = {
            "GPT-4o": "openai/gpt-4o",
            "GPT-4o Mini": "openai/gpt-4o-mini"
        }
        OPENROUTER_MODELS = {
            "Qwen2.5-VL-7B": "qwen/qwen-2.5-vl-7b-instruct",
            "Qwen2.5-VL-14B": "qwen/qwen-2.5-vl-14b-instruct",
            "Gemini Pro 1.5": "google/gemini-pro-1.5",
            "Gemini Flash": "google/gemini-flash-1.5",
            "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
            "Claude 3.5 Haiku": "anthropic/claude-3.5-haiku"
        }
        if provider == "OpenAI":
            model_dict = OPENAI_MODELS
        else:
            model_dict = OPENROUTER_MODELS
        selected_model_name = st.selectbox(
            "Choose Model:",
            list(model_dict.keys()),
            index=0
        )
        selected_model_id = model_dict[selected_model_name]
        
        # Initialize agent
        agent = None
        # Use the correct key for initialization
        key_to_use = api_key
        if key_to_use:
            if st.button("🔗 Initialize Agent"):
                with st.spinner("Initializing agent..."):
                    agent = create_agent(selected_model_id, key_to_use)
                    if agent:
                        st.success("✅ Agent initialized successfully!")
                        st.session_state.agent = agent
                    else:
                        st.error("❌ Failed to initialize agent")
        
        # Show current agent status
        if 'agent' in st.session_state:
            st.success("✅ Agent ready")
        else:
            st.warning("⚠️ Please initialize agent with API key")
        
        st.divider()
        
        # Retrieval parameters
        st.header("🔍 Retrieval Parameters")
        top_k = st.slider("Workshops retrieved", 1, 20, 6)
        time_window = st.slider("Time window (seconds)", 30, 300, 60, 30)
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            if 'chat_history' in st.session_state:
                del st.session_state.chat_history
            if 'retrieved_slides' in st.session_state:
                del st.session_state.retrieved_slides
            st.success("Chat history cleared!")
            st.rerun()
    
    # Initialize RAG system
    @st.cache_resource
    def load_rag_system(file_mtime):
        return ChemistryRAG(
            quadruplet_json_path="cookbook/agent_concepts/rag/triplets_with_youtube.json",
            save_dir="./saved_indexes",
            load_saved=True
        )
    
    # Get file modification time to invalidate cache when JSON changes
    json_path = "triplets_with_youtube.json"
    file_mtime = os.path.getmtime(json_path) if os.path.exists(json_path) else 0
    
    try:
        rag = load_rag_system(file_mtime)
        st.success("✅ RAG system loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading RAG system: {e}")
        return
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'retrieved_slides' not in st.session_state:
        st.session_state.retrieved_slides = []
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Content Search")
        
        # Search input
        search_question = st.text_input(
            "Search for chemistry content:",
            placeholder="e.g., What are examples of Buchwald Hartwig coupling?",
            key="search_input"
        )
        
        if st.button("🔍 Search", key="search_button"):
            if search_question:
                with st.spinner("Searching for relevant content..."):
                    try:
                        # Get retrievals
                        retrievals, total = rag.retrieve(
                            query=search_question,
                            top_k=top_k,
                            hybrid_weight=0.5,
                            min_score_threshold=0.3
                        )
                        
                        # Group and filter results
                        filtered_results = group_and_filter_results(retrievals, max_per_speaker=3, time_window=time_window)
                        
                        # Store retrieved slides for chat
                        st.session_state.retrieved_slides = filtered_results
                        
                        st.success(f"Found {len(filtered_results)} relevant results!")
                    except Exception as e:
                        st.error(f"❌ Error during retrieval: {e}")
            else:
                st.warning("Please enter a search question.")
        
        # Always display videos if there are retrieved slides
        if st.session_state.get('retrieved_slides'):
            st.subheader("📺 Retrieved Video Clips")
            # Group filtered results by speaker
            speaker_groups = defaultdict(list)
            for item in st.session_state.retrieved_slides:
                speaker = item['quadruplet']['speaker']
                speaker_groups[speaker].append(item)
            # Display one row per speaker
            for speaker, items in speaker_groups.items():
                st.markdown(f"### 🎤 **{speaker}**")
                # Create columns for this speaker's videos (max 3)
                cols = st.columns(min(3, len(items)))
                for i, item in enumerate(items):
                    quadruplet = item['quadruplet']
                    with cols[i]:
                        # Extract video info
                        video_id, timestamp = extract_video_id_and_timestamp(quadruplet['youtube_url'])
                        if video_id:
                            # Create embed URL with timestamp
                            embed_url = create_youtube_embed_url(video_id, timestamp)
                            # Display video preview
                            st.components.v1.iframe(
                                embed_url,
                                height=150,
                                scrolling=False
                            )
                            # Display metadata
                            st.markdown(f"**Caption:** {quadruplet['caption'][:60]}...")
                            st.markdown(f"**Timestamp:** {timestamp}s")
                            if item['score']:
                                st.markdown(f"**Score:** {item['score']:.3f}")
                            # Link to full video
                            if quadruplet['youtube_url']:
                                st.markdown(f"[Watch Full Video]({quadruplet['youtube_url']})")
                        else:
                            st.warning("No YouTube URL available")
                st.divider()
    
    with col2:
        st.subheader("💬 Chat Interface")
        
        # Chat history display
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**Assistant:** {message['content']}")
                st.divider()
        
        # Chat input
        user_question = st.text_input(
            "Ask a question about the retrieved content:",
            placeholder="e.g., Can you explain what's happening in the first video?",
            key="chat_input"
        )
        
        if st.button("💬 Send", key="send_chat"):
            if user_question and 'agent' in st.session_state and st.session_state.retrieved_slides:
                # Add user message to history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_question
                })
                
                # Generate response
                with st.spinner("Generating response..."):
                    response = generate_chat_response(
                        st.session_state.agent,
                        user_question,
                        st.session_state.retrieved_slides,
                        st.session_state.chat_history
                    )
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response
                })
                
                st.rerun()
            elif not 'agent' in st.session_state:
                st.error("Please initialize the agent first!")
            elif not st.session_state.retrieved_slides:
                st.error("Please search for content first!")
            else:
                st.error("Please enter a question!")
    
    # Information section
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        **Step 1: Configure the Agent**
        - Select a model from the dropdown
        - Enter your OpenRouter API key
        - Click "Initialize Agent"
        
        **Step 2: Search for Content**
        - Enter a chemistry question in the search box
        - Adjust retrieval parameters if needed
        - Click "Search" to find relevant video clips
        
        **Step 3: Chat with the AI**
        - Ask questions about the retrieved content
        - The AI will use the slide images and captions to answer
        - Chat history is maintained for context
        
        **Features:**
        - 🔍 Semantic search across lecture content
        - 📺 YouTube video previews at specific timestamps
        - 💬 Interactive chat with AI about retrieved content
        - 🎯 Hybrid text and image retrieval
        - 📊 Relevance scoring
        - 👥 Grouped by speaker with top 3 non-neighboring timestamps
        """)

if __name__ == "__main__":
    main() 