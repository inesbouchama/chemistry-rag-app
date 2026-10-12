#!/usr/bin/env python3
"""
Fresh Benchmarking Script for Chemistry RAG with OpenRouter Qwen
Uses rag_vf.py retrieval method with optimized parameters for best performance.
"""

import pandas as pd
import sys
import os
import re
import time
import csv
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict
import requests
import tempfile
from agno.media import Image

# Add the parent directory to the path to import ChemistryRAG
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

from rag_vf import ChemistryRAG, get_speaker_and_filename
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=".env")

def get_hf_image_local_path(hf_url, hf_token):
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    response = requests.get(hf_url, headers=headers)
    response.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(hf_url)[-1])
    tmp.write(response.content)
    tmp.close()
    return tmp.name

class ChemistryRAGBenchmark:
    def __init__(self, api_key: str):
        """Initialize the benchmarking system."""
        self.api_key = api_key
        os.environ["OPENROUTER_API_KEY"] = api_key
        
        # Initialize RAG system
        print("🔵 Loading RAG system...")
        self.rag = ChemistryRAG(
            quadruplet_json_path="triplets_with_youtube.json",
            save_dir="./saved_indexes",
            load_saved=True
        )
        
        # Initialize OpenRouter agent with Gemini 1.5 Flash (multimodal)
        print("🔵 Initializing OpenRouter agent...")
        self.agent = Agent(
            #model=OpenRouter(id="google/gemini-pro-1.5"),
            #model=OpenRouter(id="google/gemini-flash-1.5"),
            model=OpenRouter(id="qwen/qwen-2.5-vl-7b-instruct"),
            markdown=True
        )
        
        print("✅ Benchmark system initialized!")
    
    def craft_optimized_prompt(self, question: str, options: List[str], retrieved_slides: List[Dict]) -> tuple:
        """
        Craft an improved prompt for chemistry multiple-choice questions.
        
        Key improvements:
        1. Shorter, more focused prompt
        2. Clearer format instructions
        3. Reduced complexity
        4. Better structure
        """
        # Simplified system prompt optimized for more slides
        system_prompt = (
            "You are a chemistry expert. Answer the multiple-choice question based on the slide images and captions.\n\n"
            "IMPORTANT: Focus on the slides that are most relevant to the question.\n"
            "You may not need all slides - identify the key ones and ignore irrelevant ones.\n\n"
            "IMPORTANT: Respond ONLY with:\n"
            "Answer: <letter>\n"
            "Explanation: <brief reasoning>\n\n"
            "Do not include any other text or formatting."
        )
        
        # Build context from retrieved slides (limit to 3 for better performance)
        context = "SLIDES:\n"
        images = []
        
        # Limit to 3 slides to reduce complexity
        limited_slides = retrieved_slides[:6]
        
        for i, slide in enumerate(limited_slides):
            quadruplet = slide['quadruplet']
            context += f"Slide {i+1}: {quadruplet['caption']} <|image|>\n"
            images.append(Image(filepath=quadruplet['img_path']))
        
        # Build the question section
        question_section = (
            f"QUESTION: {question}\n"
            f"a) {options[0]}\n"
            f"b) {options[1]}\n"
            f"c) {options[2]}\n"
            f"d) {options[3]}\n\n"
            f"Answer:"
        )
        
        full_prompt = system_prompt + context + question_section + "IMPORTANT: Respond ONLY with:\n" + "Answer: <letter>\n" + "Explanation: <brief reasoning>\n\n" + "Do not include any other text or formatting."
        
        return full_prompt, images


    def craft_no_slide_prompt(self, question: str, options: List[str]) -> tuple:
        """
        Craft a prompt and image list using only the true slide path from the benchmark file.
        The image is loaded directly from the local path given in true_slide_path.
        """
        from agno.media import Image

        system_prompt = (
            "You are a chemistry expert. Answer the multiple-choice question based on your knowledge.\n\n"
            "IMPORTANT: Respond ONLY with:\n"
            "Answer: <letter>\n"
            "Explanation: <brief reasoning>\n\n"
            "Do not include any other text or formatting."
        )
        
        
        question_section = (
            f"QUESTION: {question}\n"
            f"a) {options[0]}\n"
            f"b) {options[1]}\n"
            f"c) {options[2]}\n"
            f"d) {options[3]}\n\n"
            f"Answer:"
        )
        full_prompt = system_prompt + question_section 
        return full_prompt
    
        

    def craft_true_slide_prompt(self, question: str, options: List[str], true_slide_path: str) -> tuple:
        """
        Craft a prompt and image list using only the true slide path from the benchmark file.
        The image is loaded directly from the local path given in true_slide_path.
        """
        from agno.media import Image
        
        # Compose prompt (no caption from quadruplets, just use the image)
        system_prompt = (
            "You are a chemistry expert. Answer the multiple-choice question based on your knowledge and the slide image.\n\n"
            "IMPORTANT: Respond ONLY with:\n"
            "Answer: <letter>\n"
            "Explanation: <brief reasoning>\n\n"
            "Do not include any other text or formatting."
        )
        context = "SLIDE: <|image|>\n"
        
        # Try to load the image, if not found replace "test" with "train"
        try:
            print(f"Trying to load image at {true_slide_path}")
            images = [Image(filepath=true_slide_path)]
        except FileNotFoundError:
            print(f"Warning: Could not find image at {true_slide_path}")

        
        question_section = (
            f"QUESTION: {question}\n"
            f"a) {options[0]}\n"
            f"b) {options[1]}\n"
            f"c) {options[2]}\n"
            f"d) {options[3]}\n\n"
            f"Answer:"
        )
        full_prompt = system_prompt + context + question_section + "IMPORTANT: Respond ONLY with:\n" + "Answer: <letter>\n" + "Explanation: <brief reasoning>\n\n" + "Do not include any other text or formatting."
        return full_prompt, images
    
    
    
    def extract_answer(self, response: str) -> str:
        """Improved answer extraction with multiple patterns."""
        if not response or response.strip() == "":
            return None
        
        # Try multiple patterns
        patterns = [
            r"Answer:\s*([a-dA-D])\b",
            r"Answer:\s*([a-dA-D])\)",
            r"([a-dA-D])\s*\)",
            r"([a-dA-D])\b"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).lower()
        
        print(f"⚠️ Could not extract answer from: {response[:200]}")
        return None
    
    def is_retrieval_correct(self, true_slide_path: str, retrievals: List[Dict]) -> bool:
        """Check if the true slide is among the retrieved slides."""
        true_key = get_speaker_and_filename(true_slide_path, 2, 1)
        retrieved_keys = [get_speaker_and_filename(item['quadruplet']['img_path'], 3, 1) for item in retrievals]
        return true_key in retrieved_keys
    
    def smart_filter_retrievals(self, retrievals: List[Dict], max_slides: int = 6) -> List[Dict]:
        """
        Keep more slides strategy.
        
        Strategy:
        1. Keep more slides (5-6) to maintain retrieval accuracy
        2. Use better prompt structure to handle more context
        3. Let the model focus on the most relevant slides
        """
        if len(retrievals) <= max_slides:
            return retrievals
        
        # Separate main and neighbor slides
        main_slides = [r for r in retrievals if r['score'] is not None]
        neighbor_slides = [r for r in retrievals if r['score'] is None]
        
        # Sort main slides by score
        main_slides.sort(key=lambda x: x['score'], reverse=True)
        
        # Take top main slides (keep more)
        selected = main_slides[:max_slides]
        
        # If we need more slides, add some neighbors
        if len(selected) < max_slides:
            remaining = max_slides - len(selected)
            selected.extend(neighbor_slides[:remaining])
        
        return selected
    
    def run_benchmark(self, benchmark_path: str, output_path: str = "gemini_flash_benchmark_results.csv"):
        """
        Run the complete benchmark evaluation.
        
        Optimized Parameters:
        - top_k=6: Good balance between coverage and precision
        - hybrid_weight=0.5: Equal weight for text and image similarity
        - min_score_threshold=0.3: Moderate threshold to ensure quality
        - neighbor_window=2: Include some context without overwhelming
        - max_slides=5: Optimal number for multimodal generation
        """
        
        # Load benchmark data
        df = pd.read_csv(benchmark_path)
        total_questions = len(df)
        
        # Initialize counters
        correct_retrievals = 0
        correct_generations = 0
        correct_generations_on_correct_retrieval = 0  # <-- new counter
        invalid_responses = 0
        empty_responses = 0
        total_response_time = 0
    
        
        for index, row in df.iterrows():
            print(f"\n--- Question {index + 1}/{total_questions} ---")
            
            # Extract question data
            question = row['Question']
            options = [row['a'], row['b'], row['c'], row['d']]
            true_answer = row['Answer']
            true_slide_path = row['slide_path']
            
            print(f"Question: {question[:100]}...")
            print(f"True answer: {true_answer}")
            
            # Step 1: Retrieve relevant slides with high accuracy parameters
            start_time = time.time()
            retrievals, total_slides = self.rag.retrieve(
                query=question,
                top_k=6,  # Back to original for high retrieval accuracy
                hybrid_weight=0.5,  # Balanced text and image similarity
                min_score_threshold=0.3,  # Original threshold
                neighbor_window=1  # Original window
            )
            
            # Step 2: Smart filtering
            filtered_retrievals = self.smart_filter_retrievals(retrievals, max_slides=6)
            
            retrieval_time = time.time() - start_time
            print(f"Retrieved {len(retrievals)} slides, filtered to {len(filtered_retrievals)}")
            
            # Step 3: Check retrieval accuracy
            retrieval_correct = self.is_retrieval_correct(true_slide_path, filtered_retrievals)
            if retrieval_correct:
                correct_retrievals += 1
                print("✅ Retrieval: Correct")
            else:
                print("❌ Retrieval: Incorrect")
            

            
            generated_answer = None
            response_content = ""
            generation_time = 0
            
            
            #prompt, images = self.craft_optimized_prompt(question, options, filtered_retrievals)
            #Try giving always the true value
            prompt, images = self.craft_true_slide_prompt(question, options, true_slide_path)
            #prompt = self.craft_no_slide_prompt(question, options)
            
            generation_start = time.time()
            response = self.agent.run(prompt, images=images)
            #response = self.agent.run(prompt)
            generation_time = time.time() - generation_start
            
            response_content = response.content
            print(f"Response: {response_content}")
            
            # Check if response is empty
            if not response_content or response_content.strip() == "":
                print("  ❌ Empty response ")
                
            
            # Extract answer
            generated_answer = self.extract_answer(response_content)
            
            total_response_time += generation_time
            
            # Evaluate generation accuracy
            if generated_answer in {"a", "b", "c", "d"}:
                if generated_answer == true_answer:
                    correct_generations += 1
                    print(f"✅ Generation: Correct ({generated_answer})")
                    if retrieval_correct:
                        correct_generations_on_correct_retrieval += 1  # <-- increment only if retrieval was correct
                else:
                    print(f"❌ Generation: Wrong (expected {true_answer}, got {generated_answer})")
            else:
                if not response_content or response_content.strip() == "":
                    empty_responses += 1
                    print("❌ Generation: Empty response")
                else:
                    invalid_responses += 1
                    print("❌ Generation: Invalid format")
                
                # Store detailed results
        
        # Calculate final metrics
        retrieval_accuracy = correct_retrievals / total_questions
        generation_accuracy = correct_generations / total_questions
        # New metric: generation accuracy on correctly retrieved slides
        if correct_retrievals > 0:
            generation_accuracy_on_correct_retrieval = correct_generations_on_correct_retrieval / correct_retrievals
        else:
            generation_accuracy_on_correct_retrieval = 0.0
        avg_response_time = total_response_time / total_questions
        empty_rate = empty_responses / total_questions
        invalid_rate = invalid_responses / total_questions
        
        # Print final results
        print("\n" + "=" * 60)
        print("🎉 BENCHMARK COMPLETED!")
        print("=" * 60)
        print(f"📊 Final Results:")
        print(f"   Total Questions: {total_questions}")
        print(f"   Retrieval Accuracy: {retrieval_accuracy:.2%} ({correct_retrievals}/{total_questions})")
        print(f"   Generation Accuracy: {generation_accuracy:.2%} ({correct_generations}/{total_questions})")
        print(f"   Generation Accuracy (on correct retrievals): {generation_accuracy_on_correct_retrieval:.2%} ({correct_generations_on_correct_retrieval}/{correct_retrievals})")
        print(f"   Empty Responses: {empty_responses} ({empty_rate:.2%})")
        print(f"   Invalid Responses: {invalid_responses} ({invalid_rate:.2%})")
        print(f"   Average Response Time: {avg_response_time:.2f}s")
        print(f"   Total Time: {total_response_time/60:.1f} minutes")
        

    
        

def main():
    """Main function to run the benchmark with Gemini Flash."""
    print("🧪 Chemistry RAG Benchmark with Gemini Pro")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not found in environment variables")
        print("Please set your OpenRouter API key:")
        print("export OPENROUTER_API_KEY='your-api-key-here'")
        return
    
    # Initialize benchmark system
    benchmark = ChemistryRAGBenchmark(api_key)
    
    # Run benchmark
    #benchmark_path = "benchmark/data/mcq_labelled.csv"

    #Synthetic benchmark
    benchmark_path = "chemistry_mcq_true_slide_updated_paths.csv"
    
    if not os.path.exists(benchmark_path):
        print(f"❌ Error: Benchmark file not found at {benchmark_path}")
        return
    
    # Run the benchmark
    results = benchmark.run_benchmark(benchmark_path)
    
    print("\n🎯 More Slides Strategy:")
    print("=" * 50)
    print("1. High-accuracy retrieval (85%+) with original parameters")
    print("   - top_k=6, hybrid_weight=0.5, threshold=0.3, window=1")
    print("   - Gets 10-15 slides with high retrieval accuracy")
    print()
    print("2. Keep more slides (6) to maintain retrieval accuracy:")
    print("   - Don't aggressively filter down to 3")
    print("   - Let the model see more context")
    print("   - Better chance of including true slide")
    print()
    print("3. Improved prompt for handling more slides:")
    print("   - Instructs model to focus on relevant slides")
    print("   - Allows ignoring irrelevant slides")
    print("   - Better context management")
    print()
    print("4. Balance between retrieval and generation:")
    print("   - Maintain high retrieval accuracy")
    print("   - Give model enough context to work with")
    print("   - Avoid overwhelming with too many slides")

if __name__ == "__main__":
    main() 