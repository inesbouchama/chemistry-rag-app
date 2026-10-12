# Chemistry Lecture RAG Pipeline
# This pipeline retrieves relevant slide-caption pairs and uses them to answer chemistry questions
#retrieval with  BGE text encoder + OCR on images & text encoder 

import os
import torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import faiss
import pickle
import pandas as pd
import re
import easyocr



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
        self.text_encoder = SentenceTransformer('BAAI/bge-large-en-v1.5')

        self.ocr = easyocr.Reader(['en'], gpu=True)  # Uses GPU if available

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
                    captions.append((speaker, caption, img_path))
                except Exception as e:
                    print(f"Error loading {img_path} or {cap_path}: {e}")

        print(f"Loaded {len(slides)} slide-caption pairs")
        return slides, captions


    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Wrapper for BGE text encoder.
        """
        embeddings = self.text_encoder.encode(
            texts,
            normalize_embeddings=True
        )
        return np.array(embeddings)


    def _build_text_index(self):
        """Build FAISS index for text embeddings"""
        #embeddings = self.text_encoder.encode([cap[1] for cap in self.captions])
        #Normalize embeddings
        embeddings = self.text_encoder.encode(
    [cap[1] for cap in self.captions],
    normalize_embeddings=True  # ✅ BGE works best normalized!
)

        dimension = embeddings.shape[1]
        
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        return index, embeddings
    
    def _build_image_index(self):
        """Build FAISS index for slide images using OCR + BGE"""
        ocr_texts = []

        for image in self.slides:
            # Convert image to text with OCR
            result = self.ocr.readtext(np.array(image))  # EasyOCR expects numpy array
            text = "\n".join([line[1] for line in result])  # Combine detected lines

            ocr_texts.append(text)

        # Embed extracted text with BGE
        embeddings = self.text_encoder.encode(
            ocr_texts,
            normalize_embeddings=True
        )

        embeddings = np.array(embeddings).astype('float32')
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return index, embeddings

    
    def retrieve(self, query: str, top_k: int, hybrid_weight: float, min_score_threshold: float) -> List[Dict]:
        """
        Retrieve relevant slide-caption pairs using hybrid retrieval.
        This version dynamically adjusts the hybrid_weight based on retrieval results.
        """
        # Reset hybrid_weight to 0 for each question
        #original_hybrid_weight = hybrid_weight
        retrievals = []

        # First retrieval attempt
        retrievals = self._perform_retrieval(query, top_k, hybrid_weight, min_score_threshold)
        # if len(retrievals) == 0:
        #     # If no slides retrieved, set hybrid_weight to 0.5 and try again
        #     hybrid_weight = 0.5
        #     print("🔵 No slides retrieved with hybrid_weight=0.0, trying hybrid_weight=0.5")
        #     retrievals = self._perform_retrieval(query, top_k, hybrid_weight, min_score_threshold)
            
        # if len(retrievals) == 0:
        #     # If still no slides retrieved, set hybrid_weight to 1 and try again
        #     print("🔵 No slides retrieved with hybrid_weight=0.5, trying hybrid_weight=1.0")
        #     hybrid_weight = 1.0
        #     retrievals = self._perform_retrieval(query, top_k, hybrid_weight, min_score_threshold)
        
        # Always revert hybrid_weight to its original value after processing
        #hybrid_weight = original_hybrid_weight
        print(f"🔍 Number of retrieved slides: {len(retrievals)}")
        total_slides = len(self.slides)
        print(f"Total slides in lecture: {total_slides}")
        return retrievals, total_slides

    def _perform_retrieval(self, query: str, top_k: int, hybrid_weight: float, min_score_threshold: float) -> List[Dict]:
        """
        This function actually performs the retrieval based on the given hybrid_weight and other parameters.
        It's extracted to avoid repetition.
        """
        text_query_embedding = self.encode_text([query])[0].astype('float32')
        text_query_embedding = text_query_embedding / np.linalg.norm(text_query_embedding)

        # Text distances
        text_distances, text_indices = self.text_index.search(
            np.array([text_query_embedding]), top_k * 2
        )

        # OCR slide distances (uses same query)
        image_distances, image_indices = self.image_index.search(
            np.array([text_query_embedding]), top_k * 2
        )


        # Normalize text and image scores
        text_scores = [1 - (d / 2) for d in text_distances[0]]
        image_scores = [1 - (d / 2) for d in image_distances[0]]

        # Min-max normalization to scale scores to [0, 1]
        text_scores = (text_scores - np.min(text_scores)) / (np.max(text_scores) - np.min(text_scores) + 1e-9)
        image_scores = (image_scores - np.min(image_scores)) / (np.max(image_scores) - np.min(image_scores) + 1e-9)

        # Apply hybrid scoring
        # hybrid_scores = {}
        # for i, idx in enumerate(text_indices[0]):
        #     score = (1 - hybrid_weight) * text_scores[i]
        #     hybrid_scores[idx] = score

        # for i, idx in enumerate(image_indices[0]):
        #     score = hybrid_weight * image_scores[i]
        #     if idx in hybrid_scores:
        #         hybrid_scores[idx] += score
        #     else:
        #         hybrid_scores[idx] = score

        
        # Hybrid scores
        hybrid_scores = {}
        for i, idx in enumerate(text_indices[0]):
            # Convert L2 distance to cosine similarity
            cosine_similarity = 1 - (text_distances[0][i] / 2)
            score = (1 - hybrid_weight) * cosine_similarity  # Text contribution
            hybrid_scores[idx] = score

        for i, idx in enumerate(image_indices[0]):
            # Convert L2 distance to cosine similarity
            cosine_similarity = 1 - (image_distances[0][i] / 2)
            score = hybrid_weight * cosine_similarity  # Image contribution
            if idx in hybrid_scores:
                hybrid_scores[idx] += score
            else:
                hybrid_scores[idx] = score
        print(f"Hybrid scores: {hybrid_scores}")
        sorted_indices = sorted(hybrid_scores.keys(), key=lambda idx: hybrid_scores[idx], reverse=True)
        #print(hybrid_scores)
        # Filter main matches by min_score_threshold
        filtered_indices = [idx for idx in sorted_indices if hybrid_scores[idx] >= min_score_threshold][:top_k]
        main_indices = filtered_indices[:top_k]
        # Add neighbors ONLY if same speaker
        neighbor_indices = set()
        for idx in main_indices:
            main_speaker = self.captions[idx][0]
            for offset in [-3, -2, -1, 1, 2, 3]:
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(self.slides):
                    neighbor_speaker = self.captions[neighbor_idx][0]
                    if neighbor_speaker == main_speaker:  # Check same speaker
                        neighbor_indices.add(neighbor_idx)

        # Combine: main retrievals first, then neighbors (no duplicate)
        all_indices = list(main_indices) + list(neighbor_indices - set(main_indices))

        # Build final retrieval list
        # ...existing code...

# Build final retrieval list
        results = []
        for idx in all_indices:
            #print(self.captions)
            slide_type = "main" if idx in main_indices else "neighbor"
            results.append({
                'slide': self.slides[idx],
                'caption': self.captions[idx][1],
                'speaker': self.captions[idx][0],
                'slide_path': self.captions[idx][2],  # Add slide path here
                'score': hybrid_scores.get(idx, None),  # Neighbor slides may not have a score
                'type': slide_type  # Add the type here!
            })

        return results

        
    def evaluate_rag_accuracy(self, benchmark_path: str, output_path: str = "./benchmark/misclassified-rag.csv"):
        """
        Evaluate the RAG accuracy by checking if the true slide is retrieved.

        Args:
            benchmark_path: Path to the benchmark CSV file (e.g., mcq_labelled.csv).
            output_path: Path to save the misclassified results.
        """
        df = pd.read_csv(benchmark_path)
        misclassified = []
        total_questions = len(df)
        correct_retrievals = 0
        total_retrievals = 0

        for index, row in df.iterrows():
            question_id = row['Index']
            question = row['Question']
            true_slide_path = row['slide_path']
            speaker = row['Speaker']

            # Retrieve slides for the question
            retrievals, total_slides = self.retrieve(
                query=question,
                top_k=6,  # Adjust as needed
                hybrid_weight=0.5,
                min_score_threshold=0.3
            )
            #print(retrievals)
            total_retrievals += len(retrievals)
            # Extract retrieved slide paths and their types
            retrieved_slide_paths = [item['slide_path'] for item in retrievals]
            retrieved_slide_types = {item['slide_path']: item['type'] for item in retrievals}

            # Check if the true slide is in the retrieved slides
            if true_slide_path in retrieved_slide_paths:
                correct_retrievals += 1
                slide_type = retrieved_slide_types[true_slide_path]
            else:
                slide_type = "not_retrieved"

            # Record misclassifications
            if slide_type == "not_retrieved":
                misclassified.append({
                    "Index": question_id,
                    "Speaker": speaker,
                    "Question": question,
                    "True Slide": true_slide_path,
                    "Retrieved Slides": retrieved_slide_paths
                })

        # Calculate RAG accuracy
        rag_accuracy = correct_retrievals / total_questions if total_questions > 0 else 0.0
        print(f"✅ RAG Accuracy: {rag_accuracy:.2%} ({correct_retrievals}/{total_questions})")

        # Save misclassifications to a CSV file
        misclassified_df = pd.DataFrame(misclassified)
        misclassified_df.to_csv(output_path, index=False)
        print(f"✅ Misclassified results saved to {output_path}")
        print("Average number of retrievals per question:", total_retrievals / total_questions if total_questions > 0 else 0)
        print(f"Average number of slides: {total_slides}")


# Usage example
if __name__ == "__main__":
    # Initialize the RAG system with the specified paths
    rag = ChemistryRAG(
        train_caption_dir="data/4m_data/english/train/caption",
        train_image_dir="data/4m_data/english/train/rgb",
        test_caption_dir="data/4m_data/english/test/caption",
        test_image_dir="data/4m_data/english/test/rgb",
        save_dir="./saved_indexes",
        load_saved=True # False to rebuild indexes
    )

    #Test the RAG system: get accuracy
    acc = rag.evaluate_rag_accuracy(benchmark_path="./benchmark/mcq_labelled.csv",)