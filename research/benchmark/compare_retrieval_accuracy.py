#!/usr/bin/env python3
"""
Compare retrieval accuracy between:
1. Original approach: 20-24 slides (top_k=6 + neighbors)
2. Reranker approach: 5-10 slides (after reranking the 20-24 slides)
"""

import pandas as pd
import sys
import os
from datetime import datetime

# Add the parent directory to the path to import ChemistryRAG
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

from rag_vf import ChemistryRAG
from reranker import Reranker, NoReranker

def compare_retrieval_approaches():
    """Compare retrieval accuracy between original and reranked approaches."""
    
    # Initialize the ChemistryRAG system
    rag = ChemistryRAG(
        quadruplet_json_path="triplets_with_youtube.json",
        save_dir="./saved_indexes",
        load_saved=True
    )
    
    # Load benchmark data
    df = pd.read_csv('benchmark/data/mcq_labelled.csv')
    
    # Test configurations
    configurations = [
        {
            "name": "Original (~90 slides)", 
            "use_reranker": False, 
            "strategy": "none",
            "max_slides": None  # Use all retrieved slides
        },
        {
            "name": "Reranked (5 slides)", 
            "use_reranker": True, 
            "strategy": "cross_encoder",
            "max_slides": 5
        },
        {
            "name": "Reranked (10 slides)", 
            "use_reranker": True, 
            "strategy": "cross_encoder",
            "max_slides": 10
        },
        {
            "name": "Reranked (20 slides)", 
            "use_reranker": True, 
            "strategy": "cross_encoder",
            "max_slides": 20
        },
    ]
    
    results = {}
    
    for config in configurations:
        # print(f"\n{'='*60}")
        # print(f"Testing: {config['name']}")
        # print(f"{'='*60}")
        
        # Initialize reranker
        if not config['use_reranker'] or config['strategy'] == "none":
            reranker = NoReranker()
        else:
            reranker = Reranker(strategy=config['strategy'])
        
        correct_retrievals = 0
        total_questions = len(df)
        total_slides_retrieved = 0
        
        for index, row in df.iterrows():
            question = row['Question']
            true_slide_path = row['slide_path'] if 'slide_path' in row else None
            
            print(f"\nQuestion {index + 1}: {question[:80]}...")
            
            # Retrieve slides using original approach (top_k=20 + neighbors)
            retrievals, total_slides = rag.retrieve(
                query=question,
                top_k=20,  # Original approach
                hybrid_weight=0.5,
                min_score_threshold=0.3,
                neighbor_window=3
            )
            
            print(f"  Retrieved {len(retrievals)} slides (original approach)")
            total_slides_retrieved += len(retrievals)
            
            # Apply reranker if enabled
            if config['use_reranker']:
                final_retrievals = reranker.rerank(
                    query=question,
                    retrieved_slides=retrievals,
                    max_slides=config['max_slides']
                )
                print(f"  After reranking: selected top {len(final_retrievals)} slides")
            else:
                # Use all retrieved slides (original approach)
                final_retrievals = retrievals
                print(f"  Using all {len(final_retrievals)} slides (no reranking)")
            
            # Check if true slide is in final selection
            if true_slide_path is not None:
                def get_speaker_and_filename(path, pos1, pos2):
                    import os
                    parts = os.path.normpath(path).split(os.sep)
                    if len(parts) < 3:
                        return path
                    return os.path.join(parts[-pos1], parts[-pos2])
                
                true_key = get_speaker_and_filename(true_slide_path, 2, 1)
                retrieved_keys = [get_speaker_and_filename(item['quadruplet']['img_path'], 3, 1) for item in final_retrievals]
                
                is_correct = true_key in retrieved_keys
                if is_correct:
                    correct_retrievals += 1
                    # print(f"  ✅ Correct retrieval")
                #else:
                    # print(f"  ❌ Incorrect retrieval")
                    # print(f"    True: {true_key}")
                    # print(f"    Retrieved: {retrieved_keys[:3]}...")
        
        accuracy = correct_retrievals / total_questions if total_questions > 0 else 0.0
        avg_slides_per_question = total_slides_retrieved / total_questions if total_questions > 0 else 0
        
        results[config['name']] = {
            'accuracy': accuracy,
            'correct': correct_retrievals,
            'total': total_questions,
            'avg_slides': avg_slides_per_question,
            'use_reranker': config['use_reranker'],
            'strategy': config['strategy'],
            'max_slides': config['max_slides']
        }
        
        print(f"\n{config['name']} Results:")
        print(f"Accuracy: {accuracy:.2%} ({correct_retrievals}/{total_questions})")
        print(f"Average slides per question: {avg_slides_per_question:.1f}")
    
    # Print summary
    # print(f"\n{'='*80}")
    # print("RETRIEVAL ACCURACY COMPARISON")
    # print(f"{'='*80}")
    
    for config_name, result in results.items():
        reranker_status = "✅ Enabled" if result['use_reranker'] else "❌ Disabled"
        max_slides_info = f"Top {result['max_slides']}" if result['max_slides'] else "All slides"
        print(f"{config_name:25} | {result['accuracy']:6.2%} | {result['correct']:2d}/{result['total']:2d} | {result['avg_slides']:5.1f} slides | {max_slides_info}")
    
    # Find best configuration
    best_config = max(results.keys(), key=lambda k: results[k]['accuracy'])
    print(f"\n🏆 Best configuration: {best_config} ({results[best_config]['accuracy']:.2%})")
    
    return results

def save_comparison_results(results, filename="retrieval_comparison_results.csv"):
    """Save comparison results to CSV."""
    with open(filename, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['Configuration', 'Accuracy', 'Correct', 'Total', 'Avg_Slides', 'Use_Reranker', 'Strategy', 'Max_Slides', 'Timestamp'])
        for config_name, result in results.items():
            writer.writerow([
                config_name,
                f"{result['accuracy']:.4f}",
                result['correct'],
                result['total'],
                f"{result['avg_slides']:.1f}",
                result['use_reranker'],
                result['strategy'],
                result['max_slides'] or "All",
                datetime.now().isoformat()
            ])
    print(f"✅ Results saved to {filename}")

if __name__ == "__main__":
    print("🔍 Comparing Retrieval Accuracy: Original vs Reranked")
    print("This compares retrieval accuracy between:")
    print("1. Original approach: ~90 slides (top_k=20 + neighbors)")
    print("2. Reranker approach: 5-20 slides (after reranking)")
    
    results = compare_retrieval_approaches()
    save_comparison_results(results)
    
    print("\n💡 Key Questions:")
    print("1. Does reranking maintain the same accuracy with fewer slides?")
    print("2. Is 5 slides enough, or do we need 10-20?")
    print("3. Can we achieve the same accuracy with much fewer slides?")
    print("4. How does batch processing perform with 100 slides?") 