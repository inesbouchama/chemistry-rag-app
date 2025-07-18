import streamlit as st
import sys
import os
import re
from pathlib import Path
from collections import defaultdict

# Add the current directory to Python path to import rag_vf
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_vf import ChemistryRAG

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

def main():
    st.set_page_config(
        page_title="Chemistry RAG with YouTube Previews",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Chemistry RAG with YouTube Previews")
    st.markdown("Ask questions about chemistry and get relevant video clips with timestamps!")
    
    # Clear cache button
    if st.button("🔄 Clear Cache & Reload Data"):
        st.cache_resource.clear()
        st.success("Cache cleared! Reloading data...")
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
    import os
    json_path = "triplets_with_youtube.json"
    file_mtime = os.path.getmtime(json_path) if os.path.exists(json_path) else 0
    
    try:
        rag = load_rag_system(file_mtime)
        st.success("✅ RAG system loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading RAG system: {e}")
        return
    
    # User input
    st.subheader("Ask a Chemistry Question")
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are examples of Buchwald Hartwig coupling?"
    )
    
    # Retrieval parameters
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        top_k = st.slider("Number of results", 1, 20, 10)
    with col2:
        hybrid_weight = st.slider("Hybrid weight", 0.0, 1.0, 0.5, 0.1)
    with col3:
        min_score_threshold = st.slider("Min score threshold", 0.0, 1.0, 0.3, 0.1)
    with col4:
        time_window = st.slider("Time window (seconds)", 30, 300, 60, 30)
    
    # Search button
    if st.button("🔍 Search", type="primary"):
        if question:
            with st.spinner("Searching for relevant content..."):
                try:
                    # Get retrievals
                    retrievals, total = rag.retrieve(
                        query=question,
                        top_k=top_k,
                        hybrid_weight=hybrid_weight,
                        min_score_threshold=min_score_threshold
                    )
                    
                    # Group and filter results
                    filtered_results = group_and_filter_results(retrievals, max_per_speaker=3, time_window=time_window)
                    
                    st.success(f"Found {len(filtered_results)} relevant results from {len(set(item['quadruplet']['speaker'] for item in filtered_results))} speakers!")
                    
                    # Display results grouped by speaker
                    st.subheader("📺 Retrieved Video Clips (Grouped by Speaker)")
                    
                    # Group filtered results by speaker
                    speaker_groups = defaultdict(list)
                    for item in filtered_results:
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
                                        height=200,
                                        scrolling=False
                                    )
                                    
                                    # Display metadata
                                    st.markdown(f"**Caption:** {quadruplet['caption'][:80]}...")
                                    st.markdown(f"**Timestamp:** {timestamp}s")
                                    if item['score']:
                                        st.markdown(f"**Score:** {item['score']:.3f}")
                                    
                                    # Link to full video
                                    if quadruplet['youtube_url']:
                                        st.markdown(f"[Watch Full Video]({quadruplet['youtube_url']})")
                                else:
                                    st.warning("No YouTube URL available")
                        
                        st.divider()
                    
                    # Display detailed results in expandable section
                    with st.expander("📋 Detailed Results"):
                        for i, item in enumerate(filtered_results):
                            quadruplet = item['quadruplet']
                            st.markdown(f"**Result {i+1}:**")
                            st.markdown(f"- **Speaker:** {quadruplet['speaker']}")
                            st.markdown(f"- **Caption:** {quadruplet['caption']}")
                            st.markdown(f"- **Timestamp:** {quadruplet['timestamp']}")
                            st.markdown(f"- **YouTube URL:** {quadruplet['youtube_url']}")
                            if item['score']:
                                st.markdown(f"- **Score:** {item['score']:.3f}")
                            st.markdown("---")
                
                except Exception as e:
                    st.error(f"❌ Error during retrieval: {e}")
        else:
            st.warning("Please enter a question to search.")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app uses a RAG (Retrieval Augmented Generation) system to find relevant chemistry content from video lectures.
        
        **Features:**
        - 🔍 Semantic search across lecture content
        - 📺 YouTube video previews at specific timestamps
        - 🎯 Hybrid text and image retrieval
        - 📊 Relevance scoring
        - 👥 Grouped by speaker with top 3 non-neighboring timestamps
        
        **How to use:**
        1. Enter your chemistry question
        2. Adjust retrieval parameters if needed
        3. Click "Search" to find relevant video clips
        4. Watch the previews or click to view full videos
        """)
        
        st.header("⚙️ Parameters")
        st.markdown(f"""
        - **Top K:** Number of results to retrieve
        - **Hybrid Weight:** Balance between text and image similarity
        - **Min Score:** Minimum relevance threshold
        - **Time Window:** Minimum seconds between timestamps
        """)

if __name__ == "__main__":
    main() 