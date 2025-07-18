# Chemistry RAG Chat App

An enhanced Streamlit application that combines RAG (Retrieval Augmented Generation) with interactive chat functionality for chemistry content.

## Features

### 🔍 **Content Search**
- Semantic search across chemistry lecture content
- Hybrid text and image retrieval using BGE embeddings and OCR
- YouTube video previews at specific timestamps
- Relevance scoring and filtering
- Grouped results by speaker with non-neighboring timestamps

### 💬 **Interactive Chat**
- Chat with AI about retrieved content
- Multiple model support (Qwen, Gemini, GPT-4, Claude)
- Chat history maintenance for context
- Multimodal responses using slide images and captions

### ⚙️ **Model Configuration**
- Model selection from popular providers
- API key management
- Retrieval parameter tuning
- Real-time agent initialization

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements_chat_app.txt
```

### 2. Prepare Data
Ensure you have the following files in the directory:
- `triplets_with_youtube.json` - Contains slide quadruplets with YouTube URLs
- `saved_indexes/` - Directory with pre-built FAISS indexes (or set `load_saved=False`)

### 3. Get API Key
- Sign up at [OpenRouter](https://openrouter.ai/)
- Get your API key
- The app supports multiple models through OpenRouter

## Usage

### Running the App
```bash
streamlit run streamlit_rag_chat_app.py
```

### Step-by-Step Guide

1. **Configure the Agent**
   - Select a model from the dropdown (Qwen2.5-VL-7B recommended for multimodal)
   - Enter your OpenRouter API key
   - Click "Initialize Agent"

2. **Search for Content**
   - Enter a chemistry question in the search box
   - Adjust retrieval parameters if needed:
     - **Number of results**: How many slides to retrieve
     - **Hybrid weight**: Balance between text and image similarity
     - **Min score threshold**: Minimum relevance score
     - **Time window**: Minimum seconds between timestamps
   - Click "Search" to find relevant video clips

3. **Chat with the AI**
   - Ask questions about the retrieved content
   - The AI will use the slide images and captions to answer
   - Chat history is maintained for context
   - Use "Clear Chat History" to start fresh

## Available Models

### Multimodal Models (Recommended)
- **Qwen2.5-VL-7B**: Best for image + text understanding
- **Qwen2.5-VL-14B**: Higher quality but slower
- **Gemini Pro 1.5**: Good multimodal capabilities

### Text-Only Models
- **GPT-4o**: High quality text responses
- **GPT-4o Mini**: Faster, more affordable
- **Claude 3.5 Sonnet**: Good reasoning capabilities
- **Claude 3.5 Haiku**: Fast and efficient

## Architecture

### RAG System
- **Text Encoder**: BGE-large-en-v1.5 for semantic search
- **Image Processing**: EasyOCR + BGE for slide content
- **Index**: FAISS for efficient similarity search
- **Hybrid Scoring**: Combines text and image similarity

### Chat System
- **Agent**: Agnostic agent with OpenRouter integration
- **Multimodal**: Processes both text and images
- **Context**: Uses retrieved slides as context
- **History**: Maintains conversation context

## Example Workflow

1. **Search**: "What is the mechanism of the aldol reaction?"
2. **Retrieve**: System finds relevant slides with timestamps
3. **Chat**: "Can you explain what's happening in the first video?"
4. **Response**: AI analyzes the slide images and explains the content

## Troubleshooting

### Common Issues

1. **Agent Initialization Fails**
   - Check your API key is correct
   - Ensure you have sufficient credits on OpenRouter
   - Try a different model

2. **Empty Responses**
   - Reduce the number of images (try 3 instead of 10)
   - Check if the model supports multimodal input
   - Try a different model

3. **Slow Performance**
   - Use smaller models (Qwen2.5-VL-7B instead of 14B)
   - Reduce the number of retrieved slides
   - Clear chat history to reduce context length

4. **No Results Found**
   - Lower the minimum score threshold
   - Increase the number of results
   - Try different search terms

### Performance Tips

- Use Qwen2.5-VL-7B for best multimodal performance
- Keep retrieved slides to 3-5 for faster responses
- Clear chat history periodically to maintain performance
- Use specific chemistry terms for better retrieval

## File Structure

```
rag/
├── streamlit_rag_chat_app.py      # Enhanced chat app
├── streamlit_rag_app.py           # Original search-only app
├── rag_vf.py                      # Core RAG system
├── requirements_chat_app.txt       # Dependencies
├── README_chat_app.md             # This file
├── triplets_with_youtube.json     # Data file
└── saved_indexes/                 # Pre-built indexes
```

## Contributing

To extend the app:

1. **Add New Models**: Update `AVAILABLE_MODELS` dictionary
2. **Enhance Chat**: Modify `generate_chat_response()` function
3. **Improve Retrieval**: Adjust parameters in the sidebar
4. **Add Features**: Extend the Streamlit interface

## License

This project is part of the ChemRAG system for chemistry education. 