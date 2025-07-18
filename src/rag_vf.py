# Chemistry Lecture RAG Pipeline
# This pipeline retrieves relevant slide-caption pairs and uses them to answer chemistry questions
# retrieval with  BGE text encoder + OCR on images & text encoder

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
import json
import requests

from transformers import CLIPProcessor, CLIPModel, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor
from pathlib import Path

json_path = str(Path(__file__).parent.parent / "triplets_with_hf_paths.json")
HF_JSON_URL = "https://huggingface.co/datasets/ines-epfl-ethz/SW4retrieval/resolve/main/triplets_with_hf_paths.json"

# Download the JSON from Hugging Face if not present
if not Path(json_path).exists():
    print(f"Downloading {Path(json_path).name} from Hugging Face...")
    headers = {"User-Agent": "Mozilla/5.0"}
    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    response = requests.get(HF_JSON_URL, headers=headers)
    response.raise_for_status()
    with open(json_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {Path(json_path).name}!")


def get_speaker_and_filename(path, pos1, pos2):
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) < 3:
        return path  # fallback
    return os.path.join(parts[-pos1], parts[-pos2])


class ChemistryRAG:
    def __init__(self,
                 quadruplet_json_path: str = json_path,
                 model_path: Optional[str] = None,
                 save_dir: str = "./saved",
                 load_saved: bool = False):
        """
        Initialize the Chemistry RAG system
        quadruplet_json_path: path to the quadruplets JSON file
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

        # Load quadruplets from JSON
        with open(str(quadruplet_json_path), 'r') as f:
            self.quadruplets = json.load(f)

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
                os.path.exists(os.path.join(self.save_dir, "image_index.faiss")) and
                os.path.exists(os.path.join(self.save_dir, "quadruplets_metadata.pkl")))

    def _save_indexes(self):
        # Save embeddings
        np.save(os.path.join(self.save_dir, "text_embeddings.npy"),
                self.text_embeddings)
        np.save(os.path.join(self.save_dir, "image_embeddings.npy"),
                self.image_embeddings)

        # Save FAISS indexes
        faiss.write_index(self.text_index, os.path.join(
            self.save_dir, "text_index.faiss"))
        faiss.write_index(self.image_index, os.path.join(
            self.save_dir, "image_index.faiss"))

        # Save quadruplets metadata
        with open(os.path.join(self.save_dir, "quadruplets_metadata.pkl"), "wb") as f:
            pickle.dump(self.quadruplets, f)

        print("✅ Saved embeddings, indexes, and quadruplet metadata.")

    def _load_indexes(self):
        # Load embeddings
        self.text_embeddings = np.load(os.path.join(
            self.save_dir, "text_embeddings.npy"))
        self.image_embeddings = np.load(os.path.join(
            self.save_dir, "image_embeddings.npy"))

        # 🔵 Normalize loaded embeddings
        self.text_embeddings = self.text_embeddings / \
            np.linalg.norm(self.text_embeddings, axis=1, keepdims=True)
        self.image_embeddings = self.image_embeddings / \
            np.linalg.norm(self.image_embeddings, axis=1, keepdims=True)

        # Load FAISS indexes
        self.text_index = faiss.read_index(
            os.path.join(self.save_dir, "text_index.faiss"))
        self.image_index = faiss.read_index(
            os.path.join(self.save_dir, "image_index.faiss"))

        # Load quadruplets metadata
        with open(os.path.join(self.save_dir, "quadruplets_metadata.pkl"), "rb") as f:
            self.quadruplets = pickle.load(f)

        print("✅ Loaded and normalized embeddings, indexes, and quadruplet metadata.")

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
        """Build FAISS index for text embeddings using quadruplets captions"""
        captions = [q["caption"] for q in self.quadruplets]
        embeddings = self.text_encoder.encode(
            captions,
            normalize_embeddings=True
        )
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        return index, embeddings

    def _build_image_index(self):
        """Build FAISS index for slide images using OCR + BGE, using quadruplets img_path"""
        ocr_texts = []
        for q in self.quadruplets:
            img_path = q["img_path"]
            try:
                image = Image.open(img_path).convert('RGB')
                result = self.ocr.readtext(np.array(image))
                text = "\n".join([line[1] for line in result])
            except Exception as e:
                print(f"Error OCR on {img_path}: {e}")
                text = ""
            ocr_texts.append(text)
        embeddings = self.text_encoder.encode(
            ocr_texts,
            normalize_embeddings=True
        )
        embeddings = np.array(embeddings).astype('float32')
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index, embeddings

    def retrieve(self, query: str, top_k: int, hybrid_weight: float, min_score_threshold: float, neighbor_window: int = 3) -> List[Dict]:
        """
        Retrieve relevant quadruplets using hybrid retrieval.
        Returns a list of dicts, each containing the full quadruplet and retrieval metadata.

        Args:
            query: The search query
            top_k: Number of main slides to retrieve
            hybrid_weight: Weight for hybrid scoring (0-1)
            min_score_threshold: Minimum score threshold for retrieval
            neighbor_window: Number of neighboring slides to include (default: 3)
        """
        retrievals = self._perform_retrieval(
            query, top_k, hybrid_weight, min_score_threshold, neighbor_window)
        total_items = len(self.quadruplets)
        # print(f"🔍 Number of retrieved items: {len(retrievals)}")
        # print(f"Total items in index: {total_items}")
        return retrievals, total_items

    def _perform_retrieval(self, query: str, top_k: int, hybrid_weight: float, min_score_threshold: float, neighbor_window) -> List[Dict]:
        text_query_embedding = self.encode_text([query])[0].astype('float32')
        text_query_embedding = text_query_embedding / \
            np.linalg.norm(text_query_embedding)

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
        text_scores = (text_scores - np.min(text_scores)) / \
            (np.max(text_scores) - np.min(text_scores) + 1e-9)
        image_scores = (image_scores - np.min(image_scores)) / \
            (np.max(image_scores) - np.min(image_scores) + 1e-9)
        # Hybrid scores
        hybrid_scores = {}
        for i, idx in enumerate(text_indices[0]):
            cosine_similarity = 1 - (text_distances[0][i] / 2)
            score = (1 - hybrid_weight) * cosine_similarity
            hybrid_scores[idx] = score
        for i, idx in enumerate(image_indices[0]):
            cosine_similarity = 1 - (image_distances[0][i] / 2)
            score = hybrid_weight * cosine_similarity
            if idx in hybrid_scores:
                hybrid_scores[idx] += score
            else:
                hybrid_scores[idx] = score
        sorted_indices = sorted(hybrid_scores.keys(
        ), key=lambda idx: hybrid_scores[idx], reverse=True)
        filtered_indices = [
            idx for idx in sorted_indices if hybrid_scores[idx] >= min_score_threshold][:top_k]
        main_indices = filtered_indices[:top_k]
        neighbor_indices = set()
        for idx in main_indices:
            main_speaker = self.quadruplets[idx]["speaker"]
            # Use neighbor_window to determine the range of neighboring slides
            for offset in range(-neighbor_window, neighbor_window + 1):
                if offset == 0:  # Skip the main slide itself
                    continue
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(self.quadruplets):
                    neighbor_speaker = self.quadruplets[neighbor_idx]["speaker"]
                    if neighbor_speaker == main_speaker:
                        neighbor_indices.add(neighbor_idx)
        all_indices = list(main_indices) + \
            list(neighbor_indices - set(main_indices))
        results = []
        for idx in all_indices:
            slide_type = "main" if idx in main_indices else "neighbor"
            quadruplet = self.quadruplets[idx]
            results.append({
                'quadruplet': quadruplet,
                'score': hybrid_scores.get(idx, None),
                'type': slide_type
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

            # Normalize the true slide path
            true_key = get_speaker_and_filename(true_slide_path, 2, 1)

            retrievals, total_slides = self.retrieve(
                query=question,
                top_k=6,
                hybrid_weight=0.5,
                min_score_threshold=0.3
            )
            total_retrievals += len(retrievals)

            # Normalize the retrieved paths
            retrieved_keys = [get_speaker_and_filename(
                item['quadruplet']['img_path'], 3, 1) for item in retrievals]
            retrieved_types = {get_speaker_and_filename(
                item['quadruplet']['img_path'], 3, 1): item['type'] for item in retrievals}

            # Print true vs predicted for this question
            print(f"Question {index}: True img path: {true_key}")
            print(f"  Predicted img paths: {retrieved_keys}")

            if true_key in retrieved_keys:
                correct_retrievals += 1
                slide_type = retrieved_types[true_key]
            else:
                slide_type = "not_retrieved"

            if slide_type == "not_retrieved":
                misclassified.append({
                    "Index": question_id,
                    "Speaker": speaker,
                    "Question": question,
                    "True Slide": true_slide_path,
                    "Retrieved Slides": retrieved_keys
                })

        rag_accuracy = correct_retrievals / \
            total_questions if total_questions > 0 else 0.0
        print(
            f"✅ RAG Accuracy: {rag_accuracy:.2%} ({correct_retrievals}/{total_questions})")

        misclassified_df = pd.DataFrame(misclassified)
        misclassified_df.to_csv(output_path, index=False)
        print(f"✅ Misclassified results saved to {output_path}")
        print("Average number of retrievals per question:",
              total_retrievals / total_questions if total_questions > 0 else 0)
        print(f"Average number of slides: {total_slides}")

    def _load_llm(self):
        """Lazy-load the VLM (Qwen2.5-VL) when needed."""
        if self.llm is None or self.tokenizer is None:
            print("🔵 Loading VLM model (Qwen2.5-VL)...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen-VL-Chat", trust_remote_code=True)
            # self.llm = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
            model_id = "Qwen/Qwen-VL-Chat"
            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True)
            self.llm = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(
                "cuda" if torch.cuda.is_available() else "cpu")
            print("✅ VLM model loaded.")

    def answer_question(self, query: str, a: str, b: str, c: str, d: str, top_k: int = 6, hybrid_weight: float = 0.3, min_score_threshold: float = 0.3, neighbor_window: int = 3) -> str:
        """Answer a chemistry question using retrieved images and captions."""
        self._load_llm()

        retrievals, total_slides = self.retrieve(
            query, top_k=top_k, hybrid_weight=hybrid_weight, min_score_threshold=min_score_threshold, neighbor_window=neighbor_window)

        images = []
        prompt = (
            "You are an expert chemistry tutor. Based only on the following image-caption pairs, answer the multiple-choice question below.\n"
            "The question is about one of the image-caption pairs. Start by finding the single pair the question is asking about. Then think through the image-caption context carefully and explain your reasoning. Then, based on your reasoning, choose the correct answer between options a, b, c, or d. There is only one correct answer.\n"
            "Your answer should have the following format:\n"
            "Answer: a, b, c, or d.\n"
            "Explanation: <your reasoning here>\n"
            "Do not include any other text.\n\n"

        )

        for i, item in enumerate(retrievals):
            # Use img_path from quadruplet
            images.append(item['quadruplet']['img_path'])
            prompt += f"\nSlide {i+1}: {item['quadruplet']['caption']} <|image|>"

        prompt += (
            f"\n\nQuestion:\n{query}\n"
            f"a. {a}\n"
            f"b. {b}\n"
            f"c. {c}\n"
            f"d. {d}\n"
        )

        inputs = self.processor(prompt, images=images,
                                return_tensors="pt").to(self.llm.device)

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

        answer = self.processor.batch_decode(
            output, skip_special_tokens=True)[0]
        return retrievals, answer.strip()

    def extract_answer(self, answer: str) -> str:
        # Look for patterns like: Answer: d or Answer:\n d
        match = re.search(r"Answer:\s*([a-dA-D])\b", answer)
        if match:
            return match.group(1).lower()
        else:
            # Print only the first 200 chars for debugging
            print("⚠️ Invalid answer format:", answer[:200])
            return None

    def call_benchmark(self, benchmark_path: str):
        acc = rag.evaluate_rag_accuracy(
            benchmark_path="/home/iboucham/repo/mmlm/src/benchmark/mcq_labelled.csv",)
        print(f"RAG Accuracy: {acc:.2%}")

    # Example usage code (commented out to avoid indentation issues)
    # count = 54
    # df = pd.read_csv('/home/iboucham/repo/mmlm/src/benchmark/mcq_labelled.csv')
    # results = []
    # correct = 0
    # for index, row in df.iterrows():
    #     # Extract question and other details
    #     question = row['Question']
    #     a = row['a']
    #     b = row['b']
    #     c = row['c']
    #     d = row['d']
    #     options = [a, b, c, d]
    #     true_label = row['Answer']
# Usage example
if __name__ == "__main__":
    # Initialize the RAG system with the specified paths
    rag = ChemistryRAG(
        quadruplet_json_path="triplets_with_youtube.json",
        save_dir="./saved_indexes",
        load_saved=False  # ⬅️ # False to rebuild indexes
    )

    # Test the RAG system: get accuracy
    # rag.call_benchmark(benchmark_path="/home/iboucham/repo/mmlm/src/benchmark/mcq_labelled.csv")

    question = "What is the mechanism of the aldol reaction?"
    retrievals, total = rag.retrieve(
        query=question,
        top_k=6,
        hybrid_weight=0.5,
        min_score_threshold=0.3
    )

    # Extract YouTube URLs from retrieved quadruplets
    youtube_urls = []
    # Only 5 first quadruplets for now
    for item in retrievals:  # [:5]
        quadruplet = item['quadruplet']
        if quadruplet['youtube_url']:  # Check if URL exists
            youtube_urls.append(quadruplet['youtube_url'])

    print("Retrieved YouTube URLs:")
    for url in youtube_urls:
        print(url)
        # TODO: get the video and play it
        # retrievals, final_answer = rag.answer_question(question, a, b, c, d, top_k=6, hybrid_weight=0.5, min_score_threshold=0.3)
    #   retrievals = rag.retrieve(query=question)

        # generated_answer = rag.extract_answer(final_answer)
        # if generated_answer not in {"a", "b", "c", "d"}:
        #     print(f"⚠️ Invalid answer format: {generated_answer}")
        #     count -= 1
        #     continue  # skip invalid predictions
        # elif generated_answer == true_label:
        #     correct += 1
