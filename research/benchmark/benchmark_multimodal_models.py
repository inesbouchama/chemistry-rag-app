#!/usr/bin/env python3
"""
Comprehensive benchmarking script for multimodal models on Chemistry RAG pipeline.
Tests multiple models with different configurations and saves results to CSV.
"""

import pandas as pd
import sys
import os
import csv
from datetime import datetime
import time
import random

# Add the parent directory to the path to import ChemistryRAG
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

from rag_vf import ChemistryRAG, get_speaker_and_filename
from agno.agent import Agent, RunResponse
from agno.models.openrouter import OpenRouter
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path=".env")

# Initialize the ChemistryRAG system
rag = ChemistryRAG(
    quadruplet_json_path="triplets_with_youtube.json",
    save_dir="./saved_indexes",
    load_saved=True
)

# Define models to test
MODELS = [
    "qwen/qwen-2.5-vl-7b-instruct",
    "google/gemini-flash-1.5",
    "google/gemini-pro-1.5",
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct"
]

# Define configurations
TOP_K_VALUES = [6, 10]
SLIDE_LIMITS = ["all", 5, 10]  # all retrievals, top 5, top 10
RUNS_PER_CONFIG = 3

def create_agent(model_id):
    """Create an agent with the specified model."""
    return Agent(
        model=OpenRouter(id=model_id),
        markdown=True
    )

def generate_answer_with_model(agent, question, options, retrieved_slides, system_prompt, suffix_prompt):
    """Generate answer using the specified model."""
    # Import Image class for multimodal processing
    from agno.media import Image
    
    # Build the full prompt in the correct order:
    # system prompt, question, options, retrieved images and captions, suffix prompt
    
    # Start with system prompt
    full_prompt = f"{system_prompt}\n\n"
    
    # Add question and options
    full_prompt += f"Question: {question}\n"
    full_prompt += f"a. {options[0]}\n"
    full_prompt += f"b. {options[1]}\n"
    full_prompt += f"c. {options[2]}\n"
    full_prompt += f"d. {options[3]}\n\n"
    
    # Add retrieved images and their captions
    full_prompt += "Based on the following slide images and captions:\n\n"
    images = []  # Collect Image objects for multimodal processing
    
    for i, slide in enumerate(retrieved_slides):
        quadruplet = slide['quadruplet']
        full_prompt += f"Slide {i+1}: {quadruplet['caption']} <|image|>"
        # Create Image object from file path
        images.append(Image(filepath=quadruplet['img_path']))
    
    # End with suffix prompt
    full_prompt += f"\n{suffix_prompt}"
    
    # Generate answer using the model with both text and images
    response = agent.run(full_prompt, images=images)
    return response.content

def extract_answer(answer):
    """Extract the answer letter from the model response."""
    import re
    # Look for patterns like: Answer: d or Answer:\n d
    match = re.search(r"Answer:\s*([a-dA-D])\b", answer)
    if match:
        return match.group(1).lower()
    else:
        print("⚠️ Invalid answer format:", answer[:200])
        return None

def is_retrieval_correct(true_slide_path, retrievals):
    """Check if the true slide is among the retrieved slides."""
    true_key = get_speaker_and_filename(true_slide_path, 2, 1)
    retrieved_keys = [get_speaker_and_filename(item['quadruplet']['img_path'], 3, 1) for item in retrievals]
    return true_key in retrieved_keys

def save_benchmark_results(results, filename="benchmark_results.csv"):
    """Save benchmark results to CSV."""
    fieldnames = [
        "timestamp", "model_id", "top_k", "slide_limit", "run_number",
        "retrieval_accuracy", "generation_accuracy", "total_questions",
        "correct_retrievals", "correct_generations", "avg_response_time",
        "notes"
    ]
    
    try:
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        print(f"✅ Results saved to {filename}")
    except Exception as e:
        print(f"⚠️ Failed to save results: {e}")

def run_benchmark():
    """Run comprehensive benchmark across all models and configurations."""
    
    # Load benchmark data
    df = pd.read_csv('benchmark/data/mcq_labelled.csv')
    
    # System prompts
    system_prompt = (
        "You are an expert chemistry tutor. Based on the provided slide images and captions, answer the multiple-choice question.\n"
        "You MUST first list the pros and cons for each option (a, b, c, d).\n"
        "Then choose ONE option and output the final answer in EXACT format:\n"
        "Answer: <letter>\n"
        "Explanation: <your reasoning>\n"
        "Only the final answer must follow the format.\n"
    )
    
    suffix_prompt = ("There is only one correct answer.\n"
                    "Strictly follow the format: Answer: <letter>. Then on the next line, Explanation: ...\n"
                    "Do not include any other text.\n\n")
    
    results = []
    
    for model_id in MODELS:
        print(f"\n🔍 Testing model: {model_id}")
        
        try:
            agent = create_agent(model_id)
        except Exception as e:
            print(f"⚠️ Failed to create agent for {model_id}: {e}")
            continue
        
        for top_k in TOP_K_VALUES:
            print(f"  📊 Testing with top_k={top_k}")
            
            for slide_limit in SLIDE_LIMITS:
                print(f"    📄 Testing with slide_limit={slide_limit}")
                
                for run in range(RUNS_PER_CONFIG):
                    print(f"      🏃 Run {run + 1}/{RUNS_PER_CONFIG}")
                    
                    # Reset counters for this run
                    correct_retrievals = 0
                    correct_generations = 0
                    total_questions = len(df)
                    response_times = []
                    
                    for index, row in df.iterrows():
                        # Extract question details
                        question = row['Question']
                        options = [row['a'], row['b'], row['c'], row['d']]
                        true_label = row['Answer']
                        true_slide_path = row['slide_path'] if 'slide_path' in row else None
                        
                        # Retrieve slides
                        retrievals, total_slides = rag.retrieve(
                            query=question,
                            top_k=top_k,
                            hybrid_weight=0.5,
                            min_score_threshold=0.3,
                            neighbor_window=3
                        )
                        
                        # Apply slide limit
                        if slide_limit == "all":
                            top_x_retrievals = retrievals
                        else:
                            top_x_retrievals = retrievals[:slide_limit]
                        
                        # Check retrieval accuracy
                        if true_slide_path is not None and is_retrieval_correct(true_slide_path, top_x_retrievals):
                            correct_retrievals += 1
                        
                        # Generate answer
                        start_time = time.time()
                        try:
                            final_answer = generate_answer_with_model(
                                agent=agent,
                                question=question,
                                options=options,
                                retrieved_slides=top_x_retrievals,
                                system_prompt=system_prompt,
                                suffix_prompt=suffix_prompt
                            )
                            response_time = time.time() - start_time
                            response_times.append(response_time)
                            
                            # Extract and check answer
                            generated_answer = extract_answer(final_answer)
                            
                            if generated_answer in {"a", "b", "c", "d"}:
                                if generated_answer == true_label:
                                    correct_generations += 1
                                    print(f"        ✅ Q{index+1}: Correct ({generated_answer})")
                                else:
                                    print(f"        ❌ Q{index+1}: Wrong (expected {true_label}, got {generated_answer})")
                            else:
                                print(f"        ⚠️ Q{index+1}: Invalid format")
                                
                        except Exception as e:
                            print(f"        💥 Q{index+1}: Error - {e}")
                            response_times.append(0)  # Add 0 time for failed requests
                    
                    # Calculate accuracies
                    retrieval_accuracy = correct_retrievals / total_questions if total_questions > 0 else 0.0
                    generation_accuracy = correct_generations / total_questions if total_questions > 0 else 0.0
                    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
                    
                    # Store results
                    result = {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "model_id": model_id,
                        "top_k": top_k,
                        "slide_limit": slide_limit,
                        "run_number": run + 1,
                        "retrieval_accuracy": retrieval_accuracy,
                        "generation_accuracy": generation_accuracy,
                        "total_questions": total_questions,
                        "correct_retrievals": correct_retrievals,
                        "correct_generations": correct_generations,
                        "avg_response_time": avg_response_time,
                        "notes": f"Retrieval: {retrieval_accuracy:.2%}, Generation: {generation_accuracy:.2%}"
                    }
                    
                    results.append(result)
                    
                    print(f"        📈 Run {run + 1} Results:")
                    print(f"          Retrieval: {retrieval_accuracy:.2%} ({correct_retrievals}/{total_questions})")
                    print(f"          Generation: {generation_accuracy:.2%} ({correct_generations}/{total_questions})")
                    print(f"          Avg Response Time: {avg_response_time:.2f}s")
                    
                    # Add delay between runs to avoid rate limiting
                    time.sleep(2)
    
    # Save all results
    save_benchmark_results(results)
    
    # Print summary
    print(f"\n🎉 Benchmark completed!")
    print(f"📊 Total configurations tested: {len(MODELS) * len(TOP_K_VALUES) * len(SLIDE_LIMITS) * RUNS_PER_CONFIG}")
    print(f"📄 Results saved to benchmark_results.csv")
    
    return results

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Multimodal Model Benchmark")
    print("=" * 60)
    print(f"Models: {len(MODELS)}")
    print(f"Top-k values: {TOP_K_VALUES}")
    print(f"Slide limits: {SLIDE_LIMITS}")
    print(f"Runs per config: {RUNS_PER_CONFIG}")
    print(f"Total runs: {len(MODELS) * len(TOP_K_VALUES) * len(SLIDE_LIMITS) * RUNS_PER_CONFIG}")
    print("=" * 60)
    
    results = run_benchmark() 