# Chemistry Lecture RAG Pipeline
# This pipeline retrieves relevant slide-caption pairs and uses them to answer chemistry questions

import os
import torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq
from typing import List, Dict, Tuple, Optional
import faiss
import pickle

class ChemistryRAG:
    def __init__(self, 
             train_caption_dir: str,
             train_image_dir: str,
             test_caption_dir: str,
             test_image_dir: str,
             model_path: Optional[str] = None,
             save_dir: str = "./saved",
             load_saved: bool = False):
        """
        Initialize the Chemistry RAG system
        ...
        save_dir: directory to save/load indexes and embeddings
        load_saved: if True, loads saved embeddings and indexes instead of recomputing
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.load_saved = load_saved
        self.model_path = model_path  # 🔵 Save model path for later lazy loading

        # Light models
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Heavy model placeholders (lazy loaded later)
        self.tokenizer = None
        self.llm = None

        # Load data
        self.train_slides, self.train_captions = self._load_dataset(train_image_dir, train_caption_dir)
        self.test_slides, self.test_captions = self._load_dataset(test_image_dir, test_caption_dir)
        
        self.slides = self.test_slides
        self.captions = self.test_captions

        # Build or load indexes
        if load_saved and self._check_saved_files():
            print("🔵 Loading saved indexes and embeddings...")
            self._load_indexes()
        else:
            print("🟠 Building new indexes and embeddings...")
            self.text_index, self.text_embeddings = self._build_text_index()
            self.image_index, self.image_embeddings = self._build_image_index()
            self._save_indexes()


    def _check_saved_files(self):
        return (os.path.exists(os.path.join(self.save_dir, "text_embeddings.npy")) and
                os.path.exists(os.path.join(self.save_dir, "image_embeddings.npy")) and
                os.path.exists(os.path.join(self.save_dir, "text_index.faiss")) and
                os.path.exists(os.path.join(self.save_dir, "image_index.faiss")))

    def _save_indexes(self):
        # Save embeddings
        np.save(os.path.join(self.save_dir, "text_embeddings.npy"), self.text_embeddings)
        np.save(os.path.join(self.save_dir, "image_embeddings.npy"), self.image_embeddings)

        # Save FAISS indexes
        faiss.write_index(self.text_index, os.path.join(self.save_dir, "text_index.faiss"))
        faiss.write_index(self.image_index, os.path.join(self.save_dir, "image_index.faiss"))

        # Save captions and slides metadata
        with open(os.path.join(self.save_dir, "metadata.pkl"), "wb") as f:
            pickle.dump((self.captions, self.slides), f)

        print("✅ Saved embeddings, indexes, and metadata.")

    def _load_indexes(self):
        # Load embeddings
        self.text_embeddings = np.load(os.path.join(self.save_dir, "text_embeddings.npy"))
        self.image_embeddings = np.load(os.path.join(self.save_dir, "image_embeddings.npy"))

        # 🔵 Normalize loaded embeddings
        self.text_embeddings = self.text_embeddings / np.linalg.norm(self.text_embeddings, axis=1, keepdims=True)
        self.image_embeddings = self.image_embeddings / np.linalg.norm(self.image_embeddings, axis=1, keepdims=True)

        # Load FAISS indexes
        self.text_index = faiss.read_index(os.path.join(self.save_dir, "text_index.faiss"))
        self.image_index = faiss.read_index(os.path.join(self.save_dir, "image_index.faiss"))

        # Load captions and slides
        with open(os.path.join(self.save_dir, "metadata.pkl"), "rb") as f:
            self.captions, self.slides = pickle.load(f)

        print("✅ Loaded and normalized embeddings, indexes, and metadata.")


    def _load_llm(self):
        """Lazy-load the VLM (Qwen2.5-VL) when needed."""
        if self.llm is None or self.tokenizer is None:
            print("🔵 Loading VLM model (Qwen2.5-VL)...")
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
            self.llm = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
            print("✅ VLM model loaded.")

        
    def _load_dataset(self, image_dir: str, caption_dir: str) -> Tuple[List[Image.Image], List[str]]:
        """
        Load slides and their captions from directories organized by speaker subfolders.

        Args:
            image_dir: Directory containing image subfolders
            caption_dir: Directory containing caption subfolders

        Returns:
            List of images and corresponding captions
        """
        slides = []
        captions = []

        # List all speaker subfolders
        speaker_folders = sorted(os.listdir(caption_dir))

        for speaker in speaker_folders:
            speaker_caption_dir = os.path.join(caption_dir, speaker)
            speaker_image_dir = os.path.join(image_dir, speaker)

            # Make sure both caption and image folder exist
            if not os.path.isdir(speaker_caption_dir) or not os.path.isdir(speaker_image_dir):
                continue  # skip if one is missing

            # List files inside speaker subfolder
            caption_files = sorted([f for f in os.listdir(speaker_caption_dir) if f.endswith('.txt')])
            image_files = sorted([f for f in os.listdir(speaker_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

            for caption_file in caption_files:
                base_name = os.path.splitext(caption_file)[0]
                image_file = f"{base_name}.png"

                # Check if corresponding image exists
                if image_file not in image_files:
                    # Maybe image is .jpg instead of .png
                    image_file = f"{base_name}.jpg"
                    if image_file not in image_files:
                        continue  # skip if no matching image

                cap_path = os.path.join(speaker_caption_dir, caption_file)
                img_path = os.path.join(speaker_image_dir, image_file)

                try:
                    image = Image.open(img_path).convert('RGB')
                    with open(cap_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()

                    slides.append(image)
                    captions.append((speaker, caption))
                except Exception as e:
                    print(f"Error loading {img_path} or {cap_path}: {e}")

        print(f"Loaded {len(slides)} slide-caption pairs")
        return slides, captions

    
    def _build_text_index(self):
        """Build FAISS index for text embeddings"""
        embeddings = self.text_encoder.encode([cap[1] for cap in self.captions])
        dimension = embeddings.shape[1]
        
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        return index, embeddings
    
    def _build_image_index(self):
        """Build FAISS index for image embeddings"""
        embeddings = []
        
        for image in self.slides:
            inputs = self.clip_processor(images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            # Normalize embeddings
            image_embedding = image_features / image_features.norm(dim=1, keepdim=True)
            embeddings.append(image_embedding.squeeze().numpy())
        
        embeddings = np.array(embeddings).astype('float32')
        dimension = embeddings.shape[1]
        
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        return index, embeddings
    
    def retrieve(self, query: str, top_k: int = 3, hybrid_weight: float = 0.5, min_score_threshold: float = 0.3) -> List[Dict]:
        """
        Retrieve relevant slide-caption pairs using hybrid retrieval, plus 2 neighbors before and after each (only if from the same speaker).
        """
        text_query_embedding = self.text_encoder.encode([query], normalize_embeddings=True)[0].astype('float32')

        # Text retrieval
        text_distances, text_indices = self.text_index.search(np.array([text_query_embedding]), top_k * 2)

        # Image retrieval
        inputs = self.clip_processor(text=query, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_query_features = self.clip_model.get_text_features(**inputs)
        image_query_embedding = (image_query_features / image_query_features.norm(dim=1, keepdim=True)).squeeze().numpy().astype('float32')

        image_distances, image_indices = self.image_index.search(np.array([image_query_embedding]), top_k * 2)

        # Hybrid scores
        hybrid_scores = {}
        for i, idx in enumerate(text_indices[0]):
            score = (1 - hybrid_weight) * (1 - text_distances[0][i])
            hybrid_scores[idx] = score
        for i, idx in enumerate(image_indices[0]):
            score = hybrid_weight * (1 - image_distances[0][i])
            if idx in hybrid_scores:
                hybrid_scores[idx] += score
            else:
                hybrid_scores[idx] = score

        sorted_indices = sorted(hybrid_scores.keys(), key=lambda idx: hybrid_scores[idx], reverse=True)

        # Filter main matches by min_score_threshold
        main_indices = [idx for idx in sorted_indices if hybrid_scores[idx] >= min_score_threshold][:top_k]

        # 🔵 Add neighbors ONLY if same speaker
        neighbor_indices = set()
        for idx in main_indices:
            main_speaker = self.captions[idx][0]
            for offset in [-2, -1, 1, 2]:
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(self.slides):
                    neighbor_speaker = self.captions[neighbor_idx][0]
                    if neighbor_speaker == main_speaker:  # ✅ Check same speaker
                        neighbor_indices.add(neighbor_idx)

        # Combine: main retrievals first, then neighbors (no duplicate)
        all_indices = list(main_indices) + list(neighbor_indices - set(main_indices))

        # Build final retrieval list
        results = []
        for idx in all_indices:
            slide_type = "main" if idx in main_indices else "neighbor"
            results.append({
                'slide': self.slides[idx],
                'caption': self.captions[idx][1],
                'speaker': self.captions[idx][0],
                'score': hybrid_scores.get(idx, None),  # Neighbor slides may not have a score
                'type': slide_type  # 🔵 Add the type here!
            })

        return results


    
    def answer_question(self, query: str, top_k: int = 3, hybrid_weight: float = 0.0, min_score_threshold: float = 0.4) -> str:
        """Answer a chemistry question using retrieved images and captions"""
        self._load_llm()
        
        retrievals = self.retrieve(query, top_k=top_k, hybrid_weight=hybrid_weight, min_score_threshold=min_score_threshold)

        images = []
        #prompt = "You are an expert chemistry tutor. Carefully examine each slide image and its caption. Then answer the multiple choice question."
        prompt = "You are an expert chemistry tutor. Answer the following question based on the provided image and text context.\n\n"
        for i, item in enumerate(retrievals):
            images.append(item['slide'])
            prompt += f"\nSlide {i+1}: {item['caption']} <|image|>"

        #prompt += f"\n\nQuestion:\n{query}\n\nInstructions:\n- Think carefully.\n- At the end, answer clearly by writing 'Answer: a', 'Answer: b', 'Answer: c', or 'Answer: d'."
        prompt += f"\n\nQuestion:\n{query}\n\nInstructions:\n- Think carefully.\n"
        inputs = self.tokenizer(prompt, images=images, return_tensors="pt")

        generation_output = self.llm.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=3,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        answer = self.tokenizer.decode(generation_output[0], skip_special_tokens=True)
        return answer.strip()


# Usage example
if __name__ == "__main__":
    # Initialize the RAG system with the specified paths
    rag = ChemistryRAG(
        train_caption_dir="data/4m_data/english/train/caption",
        train_image_dir="data/4m_data/english/train/rgb",
        test_caption_dir="data/4m_data/english/test/caption",
        test_image_dir="data/4m_data/english/test/rgb",
        save_dir="./saved_indexes",
        load_saved=False  # ⬅️ new
    )
    # Question to test retrieval score
    #test_question = """In the First Generation Synthesis of di-tert-Butyl Acridinium slide, can you give examples of the reaction times?"""
    test_question = "Can you give a quick example on the divergent total synthesis of sescitebanoid lactone family?"
    retrievals = rag.retrieve(
        test_question,
        top_k=2,
        hybrid_weight=0.0,  
        min_score_threshold=0
    )
    print(f"Retrieved {len(retrievals)} items.")

    print(f"Top retrieved slides:")
    for i, item in enumerate(retrievals):
        slide_type_emoji = "🔵" if item['type'] == 'main' else "🟢"
        score_str = f"{item['score']:.4f}" if item['score'] is not None else "N/A"
        print(f"{slide_type_emoji} Slide {i+1} (Speaker: {item['speaker']}): (score: {score_str})")
        print(f"Caption: {item['caption'][:300]}...")  # Print the first 300 characters of caption
        print("-" * 80)


        # Now, generate the answer
    print("\nGenerating answer...")
    final_answer = rag.answer_question(test_question, top_k=2, hybrid_weight=1.0, min_score_threshold=0)

    # print("\nGenerated Answer:")
    # print(final_answer)