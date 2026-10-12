# Chemistry Lecture RAG Pipeline
# This pipeline retrieves relevant slide-caption pairs and uses them to answer chemistry questions

import os
import torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor
from typing import List, Dict, Tuple, Optional
import faiss
import pickle
import pandas as pd
import re

class ChemistryRAG:
    def __init__(self, 
             train_caption_dir: str,
             train_image_dir: str,
             test_caption_dir: str,
             test_image_dir: str,
             model_path="models/fine-web-modernbert-base-8192-multi-tok-new-procedure-60-epochs-notags",
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
        # self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

        from transformers import AutoTokenizer, AutoModel

        # Ensure model_path is a local path, e.g. "./nbert-base-8192-multi-tok-new-procedure-60-epochs-notags"
        model_path_abs = os.path.abspath(self.model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path_abs, local_files_only=True)
        self.text_encoder = AutoModel.from_pretrained(model_path_abs, local_files_only=True).eval()
        self.text_encoder.to("cuda" if torch.cuda.is_available() else "cpu")


        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Heavy model placeholders (lazy loaded later)
        #self.tokenizer = None
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
            self.text_index, self.text_embeddings = self._build_text_index_MB()
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
        #self.text_embeddings = self.text_embeddings / np.linalg.norm(self.text_embeddings, axis=1, keepdims=True)
        #self.image_embeddings = self.image_embeddings / np.linalg.norm(self.image_embeddings, axis=1, keepdims=True)

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
            self.qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
            #self.llm = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
            model_id = "Qwen/Qwen-VL-Chat"
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self.llm = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
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
                    captions.append((speaker, caption, img_path))
                except Exception as e:
                    print(f"Error loading {img_path} or {cap_path}: {e}")

        print(f"Loaded {len(slides)} slide-caption pairs")
        return slides, captions

    def _build_text_index_MB(self):
        """Build FAISS index for text embeddings using ModernBERT"""
        embeddings = []
        
        for cap in self.captions:
            text = cap[1]
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}

            attention_mask = inputs['attention_mask']
            with torch.no_grad():
                outputs = self.text_encoder(**inputs)
                last_hidden = outputs.last_hidden_state

                # Mean Pooling
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask

                # Normalize
                embedding = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

            embeddings.append(embedding[0].cpu().numpy())  # Append inside the loop

        embeddings = np.array(embeddings).astype('float32')
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return index, embeddings


    
    def _build_text_index(self):
        """Build FAISS index for text embeddings"""
        #embeddings = self.text_encoder.encode([cap[1] for cap in self.captions])
        #Normalize embeddings
        embeddings = self.text_encoder.encode([cap[1] for cap in self.captions], normalize_embeddings=True)

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
    
    def retrieve(self, query: str, top_k: int = 3, hybrid_weight: float = 0.0, min_score_threshold: float = 0.3) -> List[Dict]:
        """
        Retrieve relevant slide-caption pairs using hybrid retrieval.
        This version dynamically adjusts the hybrid_weight based on retrieval results.
        """
        # Reset hybrid_weight to 0 for each question
        original_hybrid_weight = hybrid_weight
        retrievals = []

        # First retrieval attempt
        retrievals = self._perform_retrieval(query, top_k, hybrid_weight, min_score_threshold)

        if len(retrievals) == 0:
            # If no slides retrieved, set hybrid_weight to 0.5 and try again
            hybrid_weight = 0.5
            print("🔵 No slides retrieved with hybrid_weight=0.0, trying hybrid_weight=0.5")
            retrievals = self._perform_retrieval(query, top_k, hybrid_weight, min_score_threshold)
            
        if len(retrievals) == 0:
            # If still no slides retrieved, set hybrid_weight to 1 and try again
            print("🔵 No slides retrieved with hybrid_weight=0.5, trying hybrid_weight=1.0")
            hybrid_weight = 1.0
            retrievals = self._perform_retrieval(query, top_k, hybrid_weight, min_score_threshold)
        
        # Always revert hybrid_weight to its original value after processing
        hybrid_weight = original_hybrid_weight

        return retrievals


    def _perform_retrieval(self, query: str, top_k: int, hybrid_weight: float, min_score_threshold: float) -> List[Dict]:
        """
        This function actually performs the retrieval based on the given hybrid_weight and other parameters.
        It's extracted to avoid repetition.
        """
        #text_query_embedding = self.text_encoder.encode([query], normalize_embeddings=True)[0].astype('float32')

        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            last_hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
            attention_mask = inputs['attention_mask']

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_embeddings = torch.sum(last_hidden * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            query_embedding = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

        text_query_embedding = query_embedding.squeeze().cpu().numpy().astype('float32')

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
            cosine_similarity = 1 - (text_distances[0][i] / 2) # Convert distance to similarity
            score = (1 - hybrid_weight) * cosine_similarity
            hybrid_scores[idx] = score
        for i, idx in enumerate(image_indices[0]):
            cosine_similarity = 1 - (image_distances[0][i] / 2)
            score = hybrid_weight * cosine_similarity
            if idx in hybrid_scores:
                hybrid_scores[idx] += score
            else:
                hybrid_scores[idx] = score

        sorted_indices = sorted(hybrid_scores.keys(), key=lambda idx: hybrid_scores[idx], reverse=True)

        # Filter main matches by min_score_threshold
        #main_indices = [idx for idx in sorted_indices if hybrid_scores[idx] >= min_score_threshold][:top_k]
        main_indices = sorted_indices[:top_k]
        # Add neighbors ONLY if same speaker
        neighbor_indices = set()
        for idx in main_indices:
            main_speaker = self.captions[idx][0]
            for offset in [-2, -1, 1, 2]:
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


    
    def answer_question(self, query: str, a: str, b: str, c: str, d: str, top_k: int = 3, hybrid_weight: float = 0.3, min_score_threshold: float = 0.3) -> str:
        """Answer a chemistry question using retrieved images and captions."""
        self._load_llm()

        retrievals = self.retrieve(query, top_k=top_k, hybrid_weight=hybrid_weight, min_score_threshold=min_score_threshold)

        images = []
        prompt = (
            "You are an expert chemistry tutor. Based only on the following image-caption pairs, answer the multiple-choice question below.\n"
        )

        for i, item in enumerate(retrievals):
            images.append(item['slide'])
            prompt += f"\nSlide {i+1}: {item['caption']} <|image|>"

        prompt += (
            f"\n\nQuestion:\n{query}\n"
            f"a. {a}\n"
            f"b. {b}\n"
            f"c. {c}\n"
            f"d. {d}\n\n"
            "Start by choosing the single most relevant slide to answer this question. Then think through the image-caption context carefully and explain your reasoning. Then, based on your reasoning, choose the correct answer. Only reply with one of the following single letters: a, b, c, or d.\n"
            "There is only one correct answer.\n"
            "Do not include anything but the correct letter in your answer.\n"
            "Answer:\n"
        )

        inputs = self.processor(prompt, images=images, return_tensors="pt").to(self.llm.device)

        with torch.no_grad():
            output = self.llm.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.9,
                top_p=0.9,
                no_repeat_ngram_size=2,
                early_stopping=True
            )

        answer = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        return retrievals, answer.strip()


    def extract_answer(self, answer: str) -> str:
        # Look for patterns like: Answer: d or Answer:\n d
        match = re.search(r"Answer:\s*([a-dA-D])\b", answer)
        if match:
            return match.group(1).lower()
        else:
            print("⚠️ Invalid answer format:", answer[:200])  # Print only the first 200 chars for debugging
            return None
        
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

        for index, row in df.iterrows():
            question_id = row['Index']
            question = row['Question']
            true_slide_path = row['slide_path']
            speaker = row['Speaker']

            # Retrieve slides for the question
            retrievals = self.retrieve(
                query=question,
                top_k=3,  # Adjust as needed
                hybrid_weight=0.3,
                min_score_threshold=0
            )
            #print(retrievals)

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


# Usage example
if __name__ == "__main__":
    # Initialize the RAG system with the specified paths
    rag = ChemistryRAG(
        train_caption_dir="data/4m_data/english/train/caption",
        train_image_dir="data/4m_data/english/train/rgb",
        test_caption_dir="data/4m_data/english/test/caption",
        test_image_dir="data/4m_data/english/test/rgb",
        save_dir="./saved_indexes",
        load_saved=False # ⬅️ new
    )

    #Test the RAG system: get accuracy
    acc = rag.evaluate_rag_accuracy(benchmark_path="./benchmark/mcq_labelled.csv",)

#     # Question to test retrieval score
#     #test_question = """In the First Generation Synthesis of di-tert-Butyl Acridinium slide, can you give examples of the reaction times?"""
#     #test_question = "Can you give a quick example on the divergent total synthesis of sescitebanoid lactone family?"
#     #In the slide titled Multi-Gram Scale Preparation, we can see in how many chemical steps is acridinium salt synthesized. Starting material and a product of the step are separated by arrow How many steps are needed?
#     #In the slide titled Multi-Gram Scale Preparation, in how many many chemical steps is acridinium salt synthesized. Starting material and a product of the step are separated by arrow.
# #     test_question = (
# #     """In multi-gram scale preparation, in how many many chemical steps is acridinium salt synthesized. Starting material and a product of the step are separated by arrow.
# #     a. Two
# #     b. Four
# #     c. Three
# #     d. Five"""
# # )

###########
    count = 54
    df = pd.read_csv('./benchmark/mcq_labelled.csv')
    results = []
    correct = 0
    for index, row in df.iterrows():
        # Extract question and other details
        question = row['Question']
        a = row['a']
        b = row['b']
        c = row['c']
        d = row['d']
        options = [a, b, c, d]
        true_label = row['Answer']


        retrievals, final_answer = rag.answer_question(question, a, b, c, d, top_k=3, hybrid_weight=0.3, min_score_threshold=0)
    #   retrievals = rag.retrieve(query=question)
    
        generated_answer = rag.extract_answer(final_answer)
        if generated_answer not in {"a", "b", "c", "d"}:
            print(f"⚠️ Invalid answer format: {generated_answer}")
            count -= 1
            continue  # skip invalid predictions
        elif generated_answer == true_label:
            correct += 1

    accuracy = correct / count
    print(accuracy)
#############

