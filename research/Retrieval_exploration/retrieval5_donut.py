import os
import torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import faiss
import pickle
import pandas as pd
from transformers import DonutProcessor, VisionEncoderDecoderModel
import re
from tqdm import tqdm
import time


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
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.load_saved = load_saved
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ✅ Text encoder
        self.text_encoder = SentenceTransformer('BAAI/bge-large-en-v1.5', device=self.device)

        # ✅ DONUT
        self.processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        self.donut = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        
        self.donut.to(self.device)

        # Optional: use half precision for speed if possible
        self.donut = self.donut.half().to(self.device)

        

        self.tokenizer = None
        self.llm = None

        self.train_slides, self.train_captions = self._load_dataset(train_image_dir, train_caption_dir)
        self.test_slides, self.test_captions = self._load_dataset(test_image_dir, test_caption_dir)

        self.slides = self.test_slides
        self.captions = self.test_captions

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
        np.save(os.path.join(self.save_dir, "text_embeddings.npy"), self.text_embeddings)
        np.save(os.path.join(self.save_dir, "image_embeddings.npy"), self.image_embeddings)
        faiss.write_index(self.text_index, os.path.join(self.save_dir, "text_index.faiss"))
        faiss.write_index(self.image_index, os.path.join(self.save_dir, "image_index.faiss"))
        with open(os.path.join(self.save_dir, "metadata.pkl"), "wb") as f:
            pickle.dump((self.captions, self.slides), f)
        print("✅ Saved embeddings, indexes, and metadata.")

    def _load_indexes(self):
        self.text_embeddings = np.load(os.path.join(self.save_dir, "text_embeddings.npy"))
        self.image_embeddings = np.load(os.path.join(self.save_dir, "image_embeddings.npy"))
        self.text_embeddings = self.text_embeddings / np.linalg.norm(self.text_embeddings, axis=1, keepdims=True)
        self.image_embeddings = self.image_embeddings / np.linalg.norm(self.image_embeddings, axis=1, keepdims=True)
        self.text_index = faiss.read_index(os.path.join(self.save_dir, "text_index.faiss"))
        self.image_index = faiss.read_index(os.path.join(self.save_dir, "image_index.faiss"))
        with open(os.path.join(self.save_dir, "metadata.pkl"), "rb") as f:
            self.captions, self.slides = pickle.load(f)
        print("✅ Loaded and normalized embeddings, indexes, and metadata.")

    def _load_dataset(self, image_dir: str, caption_dir: str) -> Tuple[List[Image.Image], List[str]]:
        slides = []
        captions = []
        speaker_folders = sorted(os.listdir(caption_dir))

        for speaker in speaker_folders:
            speaker_caption_dir = os.path.join(caption_dir, speaker)
            speaker_image_dir = os.path.join(image_dir, speaker)
            if not os.path.isdir(speaker_caption_dir) or not os.path.isdir(speaker_image_dir):
                continue

            caption_files = sorted([f for f in os.listdir(speaker_caption_dir) if f.endswith('.txt')])
            image_files = sorted([f for f in os.listdir(speaker_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

            for caption_file in caption_files:
                base_name = os.path.splitext(caption_file)[0]
                image_file = f"{base_name}.png"
                if image_file not in image_files:
                    image_file = f"{base_name}.jpg"
                    if image_file not in image_files:
                        continue

                cap_path = os.path.join(speaker_caption_dir, caption_file)
                img_path = os.path.join(speaker_image_dir, image_file)

                try:
                    image = Image.open(img_path).convert('RGB')

                    #improve speed by resizing images
                    image = image.resize((1280, 1280))

                    with open(cap_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    
                    slides.append(image)
                    captions.append((speaker, caption, img_path))
                except Exception as e:
                    print(f"Error loading {img_path} or {cap_path}: {e}")

        print(f"Loaded {len(slides)} slide-caption pairs")
        return slides, captions

    def encode_text(self, texts: List[str]) -> np.ndarray:
        embeddings = self.text_encoder.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)

    def _build_text_index(self):
        embeddings = self.text_encoder.encode(
            [cap[1] for cap in self.captions],
            normalize_embeddings=True
        )
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        return index, embeddings


    def _build_image_index(self):
        """
        Build FAISS index for slide images using DONUT + BGE.
        Fully GPU + robust fallback.
        """
        donut_outputs = []

        for idx, image in tqdm(enumerate(self.slides), total=len(self.slides)):

            try:
                # ✅ 1 — Encode image
                pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
                assert pixel_values is not None and pixel_values.shape[0] > 0, f"Empty pixel_values at slide {idx}"
                pixel_values = pixel_values.half()
                # ✅ 2 — Prepare prompt: PROMPT TUNING 
                prompt = "Answer the question based on the image: What is the title of the slide? What is it describing and how?"
                decoder_input_ids = self.processor.tokenizer(
                    prompt,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).input_ids.to(self.device)
                assert decoder_input_ids is not None, f"No decoder ids at slide {idx}"

                # ✅ 3 — Generate
                outputs = self.donut.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=512,
                    bad_words_ids=[[self.processor.tokenizer.unk_token_id]]
                )
                assert outputs is not None, f"No output generated at slide {idx}"

                # ✅ 4 — Decode
                decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
                assert decoded is not None and len(decoded) > 0, f"Decoded is empty at slide {idx}"

                clean_decoded = re.sub(r"[^\w\s\-\=\+\>\<\[\]\(\)\{\}\.\,]", " ", decoded)
                if not clean_decoded or len(clean_decoded) < 5:
                    raise ValueError("Too short or empty")

                donut_outputs.append(clean_decoded)
                print(f"✅ DONUT succeeded on slide {idx} : {clean_decoded[:40]}")


            except Exception as e:
                print(f"❌ DONUT failed on slide {idx}: {repr(e)}")
                donut_outputs.append("no_text_detected")


        print("✅ DONUT done — embedding text...")
        embeddings = self.text_encoder.encode(
            donut_outputs,
            normalize_embeddings=True
        )
        embeddings = np.array(embeddings).astype('float32')

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return index, embeddings  



    def retrieve(self, query: str, top_k: int, hybrid_weight: float, min_score_threshold: float) -> List[Dict]:
        retrievals = self._perform_retrieval(query, top_k, hybrid_weight, min_score_threshold)
        print(f"🔍 Number of retrieved slides: {len(retrievals)}")
        print(f"Total slides in lecture: {len(self.slides)}")
        return retrievals, len(self.slides)

    def _perform_retrieval(self, query: str, top_k: int, hybrid_weight: float, min_score_threshold: float) -> List[Dict]:
        text_query_embedding = self.encode_text([query])[0].astype('float32')
        text_query_embedding = text_query_embedding / np.linalg.norm(text_query_embedding)
        text_distances, text_indices = self.text_index.search(np.array([text_query_embedding]), top_k * 2)
        image_distances, image_indices = self.image_index.search(np.array([text_query_embedding]), top_k * 2)

        text_scores = [1 - (d / 2) for d in text_distances[0]]
        image_scores = [1 - (d / 2) for d in image_distances[0]]
        text_scores = (text_scores - np.min(text_scores)) / (np.max(text_scores) - np.min(text_scores) + 1e-9)
        image_scores = (image_scores - np.min(image_scores)) / (np.max(image_scores) - np.min(image_scores) + 1e-9)

        hybrid_scores = {}
        for i, idx in enumerate(text_indices[0]):
            score = (1 - hybrid_weight) * text_scores[i]
            hybrid_scores[idx] = score
        for i, idx in enumerate(image_indices[0]):
            score = hybrid_weight * image_scores[i]
            hybrid_scores[idx] = hybrid_scores.get(idx, 0) + score

        sorted_indices = sorted(hybrid_scores.keys(), key=lambda idx: hybrid_scores[idx], reverse=True)
        filtered_indices = [idx for idx in sorted_indices if hybrid_scores[idx] >= min_score_threshold][:top_k]
        main_indices = filtered_indices
        neighbor_indices = set()
        for idx in main_indices:
            main_speaker = self.captions[idx][0]
            for offset in [-3, -2, -1, 1, 2, 3]:
                neighbor_idx = idx + offset
                if 0 <= neighbor_idx < len(self.slides) and self.captions[neighbor_idx][0] == main_speaker:
                    neighbor_indices.add(neighbor_idx)
        all_indices = list(main_indices) + list(neighbor_indices - set(main_indices))
        results = []
        for idx in all_indices:
            slide_type = "main" if idx in main_indices else "neighbor"
            results.append({
                'slide': self.slides[idx],
                'caption': self.captions[idx][1],
                'speaker': self.captions[idx][0],
                'slide_path': self.captions[idx][2],
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

    start = time.time()
    # Initialize the RAG system with the specified paths
    rag = ChemistryRAG(
        train_caption_dir="data/4m_data/english/train/caption",
        train_image_dir="data/4m_data/english/train/rgb",
        test_caption_dir="data/4m_data/english/test/caption",
        test_image_dir="data/4m_data/english/test/rgb",
        save_dir="./saved_indexes",
        load_saved=False # False to rebuild indexes
    )

    #Test the RAG system: get accuracy
    acc = rag.evaluate_rag_accuracy(benchmark_path="./benchmark/mcq_labelled.csv",)
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")