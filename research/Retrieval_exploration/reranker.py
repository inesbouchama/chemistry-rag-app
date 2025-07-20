import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder
import torch
import re

class Reranker:
    """
    A modular reranker that can reorder retrieved slides to improve accuracy.
    Supports multiple reranking strategies and can be easily switched on/off.
    """
    
    def __init__(self, 
                 strategy: str = "cross_encoder",
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 use_gpu: bool = True):
        """
        Initialize the reranker.
        
        Args:
            strategy: Reranking strategy ("cross_encoder", "semantic_similarity", "hybrid", "chemistry_specific", "none")
            model_name: Model name for cross-encoder
            use_gpu: Whether to use GPU for reranking
        """
        self.strategy = strategy
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        if strategy == "cross_encoder":
            # Try to use a more general-purpose model for chemistry content
            try:
                self.cross_encoder = CrossEncoder(model_name)
                if self.use_gpu:
                    self.cross_encoder.model = self.cross_encoder.model.cuda()
                print(f"✅ Loaded cross-encoder: {model_name}")
            except Exception as e:
                print(f"⚠️ Failed to load {model_name}, falling back to default")
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                if self.use_gpu:
                    self.cross_encoder.model = self.cross_encoder.model.cuda()
        elif strategy == "semantic_similarity":
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            print("✅ Loaded semantic similarity model: BAAI/bge-large-en-v1.5")
        elif strategy == "hybrid":
            # Initialize both cross-encoder and semantic model
            try:
                self.cross_encoder = CrossEncoder(model_name)
                if self.use_gpu:
                    self.cross_encoder.model = self.cross_encoder.model.cuda()
            except Exception as e:
                print(f"⚠️ Failed to load {model_name}, falling back to default")
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                if self.use_gpu:
                    self.cross_encoder.model = self.cross_encoder.model.cuda()
            
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            print("✅ Loaded hybrid reranker with cross-encoder and semantic similarity")
        elif strategy == "chemistry_specific":
            # Chemistry-specific reranker using keyword matching and semantic similarity
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            print("✅ Loaded chemistry-specific reranker")
    
    def rerank(self, 
               query: str, 
               retrieved_slides: List[Dict], 
               target_accuracy: float = 0.95,
               max_slides: int = 5) -> List[Dict]:
        """
        Rerank retrieved slides to improve accuracy.
        
        Args:
            query: The original query
            retrieved_slides: List of retrieved slide dictionaries
            target_accuracy: Target accuracy to achieve (0.95 = 95%)
            max_slides: Maximum number of slides to return
            
        Returns:
            Reranked list of slide dictionaries
        """
        if not retrieved_slides:
            return []
        
        # Extract captions and scores
        captions = [slide['quadruplet']['caption'] for slide in retrieved_slides]
        original_scores = [slide.get('score', 0) for slide in retrieved_slides]
        
        print(f"🔍 Reranking {len(retrieved_slides)} slides with strategy: {self.strategy}")
        print(f"Query: {query[:100]}...")
        
        # For large slide sets (>50), use batch processing for efficiency
        if len(retrieved_slides) > 50:
            print(f"📦 Large slide set detected ({len(retrieved_slides)} slides), using batch processing")
            return self._batch_rerank(query, retrieved_slides, captions, original_scores, max_slides)
        
        if self.strategy == "cross_encoder":
            return self._cross_encoder_rerank(query, retrieved_slides, captions, max_slides)
        elif self.strategy == "semantic_similarity":
            return self._semantic_similarity_rerank(query, retrieved_slides, captions, max_slides)
        elif self.strategy == "hybrid":
            return self._hybrid_rerank(query, retrieved_slides, captions, original_scores, max_slides)
        elif self.strategy == "chemistry_specific":
            return self._chemistry_specific_rerank(query, retrieved_slides, captions, original_scores, max_slides)
        else:
            # No reranking, return original top slides
            return retrieved_slides[:max_slides]
    
    def _chemistry_specific_rerank(self, query: str, slides: List[Dict], captions: List[str], 
                                 original_scores: List[float], max_slides: int) -> List[Dict]:
        """Chemistry-specific reranking using keyword matching and semantic similarity."""
        # Chemistry keywords that should boost relevance
        chemistry_keywords = [
            'reaction', 'mechanism', 'synthesis', 'catalyst', 'equilibrium', 'acid', 'base',
            'molecule', 'compound', 'organic', 'inorganic', 'bond', 'electron', 'proton',
            'nucleus', 'valence', 'oxidation', 'reduction', 'hydrolysis', 'condensation',
            'polymerization', 'isomer', 'stereochemistry', 'chirality', 'enantiomer',
            'diastereomer', 'conformation', 'resonance', 'aromatic', 'aliphatic',
            'functional group', 'alkyl', 'aryl', 'hydroxyl', 'carbonyl', 'carboxyl',
            'amine', 'amide', 'ester', 'ether', 'halogen', 'nitro', 'sulfonyl'
        ]
        
        # Extract chemistry terms from query
        query_lower = query.lower()
        query_chemistry_terms = [term for term in chemistry_keywords if term in query_lower]
        
        # Calculate keyword overlap scores
        keyword_scores = []
        for caption in captions:
            caption_lower = caption.lower()
            # Count chemistry keywords in caption
            keyword_count = sum(1 for term in chemistry_keywords if term in caption_lower)
            # Bonus for query-specific chemistry terms
            query_term_bonus = sum(2 for term in query_chemistry_terms if term in caption_lower)
            keyword_scores.append(keyword_count + query_term_bonus)
        
        # Get semantic similarity scores
        query_embedding = self.semantic_model.encode([query])
        caption_embeddings = self.semantic_model.encode(captions)
        
        semantic_scores = []
        for caption_emb in caption_embeddings:
            similarity = np.dot(query_embedding[0], caption_emb) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(caption_emb)
            )
            semantic_scores.append(similarity)
        
        # Normalize scores
        keyword_scores = np.array(keyword_scores)
        semantic_scores = np.array(semantic_scores)
        original_scores = np.array(original_scores)
        
        # Normalize keyword scores to [0, 1]
        if len(keyword_scores) > 0:
            keyword_scores = (keyword_scores - np.min(keyword_scores)) / (np.max(keyword_scores) - np.min(keyword_scores) + 1e-9)
        
        # Normalize original scores to [0, 1] if they're not already
        if len(original_scores) > 0:
            # Filter out None values for normalization
            valid_scores = [s for s in original_scores if s is not None]
            if valid_scores:
                min_score = min(valid_scores)
                max_score = max(valid_scores)
                original_scores = [(s - min_score) / (max_score - min_score + 1e-9) if s is not None else 0.0 for s in original_scores]
            else:
                original_scores = [0.0] * len(original_scores)
        original_scores = np.array(original_scores)
        
        # Combine scores: prioritize semantic similarity, then keyword matching, then original scores
        chemistry_scores = 0.6 * semantic_scores + 0.3 * keyword_scores + 0.1 * original_scores
        
        # Sort slides by chemistry-specific scores
        scored_slides = list(zip(slides, chemistry_scores))
        scored_slides.sort(key=lambda x: x[1], reverse=True)
        
        # Return top slides with updated scores
        reranked_slides = []
        for slide, score in scored_slides[:max_slides]:
            slide['rerank_score'] = score
            reranked_slides.append(slide)
        
        print(f"📊 Chemistry-specific scores - Semantic: {min(semantic_scores):.3f}-{max(semantic_scores):.3f}, "
              f"Keywords: {min(keyword_scores):.3f}-{max(keyword_scores):.3f}, "
              f"Final: {min(chemistry_scores):.3f}-{max(chemistry_scores):.3f}")
        return reranked_slides
    
    def _batch_rerank(self, query: str, slides: List[Dict], captions: List[str], 
                     original_scores: List[float], max_slides: int) -> List[Dict]:
        """Efficient batch reranking for large slide sets (>50 slides)."""
        print(f"⚡ Using batch processing for {len(slides)} slides")
        
        if self.strategy == "cross_encoder":
            return self._batch_cross_encoder_rerank(query, slides, captions, max_slides)
        elif self.strategy == "semantic_similarity":
            return self._batch_semantic_rerank(query, slides, captions, max_slides)
        elif self.strategy == "hybrid":
            return self._batch_hybrid_rerank(query, slides, captions, original_scores, max_slides)
        elif self.strategy == "chemistry_specific":
            return self._batch_chemistry_rerank(query, slides, captions, original_scores, max_slides)
        else:
            # For large sets, just take top by original score
            sorted_slides = sorted(slides, key=lambda x: x.get('score', 0) if x.get('score') is not None else 0, reverse=True)
            return sorted_slides[:max_slides]
    
    def _batch_cross_encoder_rerank(self, query: str, slides: List[Dict], captions: List[str], max_slides: int) -> List[Dict]:
        """Batch cross-encoder reranking for large slide sets."""
        # Process in batches of 20 for memory efficiency
        batch_size = 20
        all_scores = []
        
        for i in range(0, len(captions), batch_size):
            batch_captions = captions[i:i+batch_size]
            batch_slides = slides[i:i+batch_size]
            
            # Create query-document pairs for this batch
            pairs = [[query, caption] for caption in batch_captions]
            
            # Get cross-encoder scores for this batch
            batch_scores = self.cross_encoder.predict(pairs)
            all_scores.extend(batch_scores)
        
        # Sort all slides by their cross-encoder scores
        scored_slides = list(zip(slides, all_scores))
        scored_slides.sort(key=lambda x: x[1], reverse=True)
        
        # Return top slides with updated scores
        reranked_slides = []
        for slide, score in scored_slides[:max_slides]:
            slide['rerank_score'] = score
            reranked_slides.append(slide)
        
        print(f"📊 Batch cross-encoder scores range: {min(all_scores):.3f} - {max(all_scores):.3f}")
        return reranked_slides
    
    def _batch_semantic_rerank(self, query: str, slides: List[Dict], captions: List[str], max_slides: int) -> List[Dict]:
        """Batch semantic similarity reranking for large slide sets."""
        # Encode query once
        query_embedding = self.semantic_model.encode([query])
        
        # Process captions in batches
        batch_size = 50
        all_similarities = []
        
        for i in range(0, len(captions), batch_size):
            batch_captions = captions[i:i+batch_size]
            batch_slides = slides[i:i+batch_size]
            
            # Encode batch captions
            caption_embeddings = self.semantic_model.encode(batch_captions)
            
            # Calculate similarities for this batch
            batch_similarities = []
            for caption_emb in caption_embeddings:
                similarity = np.dot(query_embedding[0], caption_emb) / (
                    np.linalg.norm(query_embedding[0]) * np.linalg.norm(caption_emb)
                )
                batch_similarities.append(similarity)
            
            all_similarities.extend(batch_similarities)
        
        # Sort slides by similarity scores
        scored_slides = list(zip(slides, all_similarities))
        scored_slides.sort(key=lambda x: x[1], reverse=True)
        
        # Return top slides with updated scores
        reranked_slides = []
        for slide, score in scored_slides[:max_slides]:
            slide['rerank_score'] = score
            reranked_slides.append(slide)
        
        print(f"📊 Batch semantic similarity scores range: {min(all_similarities):.3f} - {max(all_similarities):.3f}")
        return reranked_slides
    
    def _batch_hybrid_rerank(self, query: str, slides: List[Dict], captions: List[str], 
                           original_scores: List[float], max_slides: int) -> List[Dict]:
        """Batch hybrid reranking for large slide sets."""
        # Process cross-encoder in batches
        batch_size = 20
        all_cross_scores = []
        
        for i in range(0, len(captions), batch_size):
            batch_captions = captions[i:i+batch_size]
            pairs = [[query, caption] for caption in batch_captions]
            batch_cross_scores = self.cross_encoder.predict(pairs)
            all_cross_scores.extend(batch_cross_scores)
        
        # Process semantic similarity in batches
        query_embedding = self.semantic_model.encode([query])
        all_semantic_scores = []
        
        for i in range(0, len(captions), batch_size):
            batch_captions = captions[i:i+batch_size]
            caption_embeddings = self.semantic_model.encode(batch_captions)
            
            batch_semantic_scores = []
            for caption_emb in caption_embeddings:
                similarity = np.dot(query_embedding[0], caption_emb) / (
                    np.linalg.norm(query_embedding[0]) * np.linalg.norm(caption_emb)
                )
                batch_semantic_scores.append(similarity)
            
            all_semantic_scores.extend(batch_semantic_scores)
        
        # Normalize scores
        all_cross_scores = 1 / (1 + np.exp(-np.array(all_cross_scores)))  # Sigmoid normalization
        all_semantic_scores = np.array(all_semantic_scores)
        original_scores = np.array(original_scores)
        
        # Normalize original scores
        if len(original_scores) > 0:
            valid_scores = [s for s in original_scores if s is not None]
            if valid_scores:
                min_score = min(valid_scores)
                max_score = max(valid_scores)
                original_scores = [(s - min_score) / (max_score - min_score + 1e-9) if s is not None else 0.0 for s in original_scores]
            else:
                original_scores = [0.0] * len(original_scores)
        original_scores = np.array(original_scores)
        
        # Combine scores
        hybrid_scores = 0.3 * all_cross_scores + 0.5 * all_semantic_scores + 0.2 * original_scores
        
        # Sort slides by hybrid scores
        scored_slides = list(zip(slides, hybrid_scores))
        scored_slides.sort(key=lambda x: x[1], reverse=True)
        
        # Return top slides with updated scores
        reranked_slides = []
        for slide, score in scored_slides[:max_slides]:
            slide['rerank_score'] = score
            reranked_slides.append(slide)
        
        print(f"📊 Batch hybrid scores - Cross: {min(all_cross_scores):.3f}-{max(all_cross_scores):.3f}, "
              f"Semantic: {min(all_semantic_scores):.3f}-{max(all_semantic_scores):.3f}, "
              f"Final: {min(hybrid_scores):.3f}-{max(hybrid_scores):.3f}")
        return reranked_slides
    
    def _batch_chemistry_rerank(self, query: str, slides: List[Dict], captions: List[str], 
                              original_scores: List[float], max_slides: int) -> List[Dict]:
        """Batch chemistry-specific reranking for large slide sets."""
        # Chemistry keywords
        chemistry_keywords = [
            'reaction', 'mechanism', 'synthesis', 'catalyst', 'equilibrium', 'acid', 'base',
            'molecule', 'compound', 'organic', 'inorganic', 'bond', 'electron', 'proton',
            'nucleus', 'valence', 'oxidation', 'reduction', 'hydrolysis', 'condensation',
            'polymerization', 'isomer', 'stereochemistry', 'chirality', 'enantiomer',
            'diastereomer', 'conformation', 'resonance', 'aromatic', 'aliphatic',
            'functional group', 'alkyl', 'aryl', 'hydroxyl', 'carbonyl', 'carboxyl',
            'amine', 'amide', 'ester', 'ether', 'halogen', 'nitro', 'sulfonyl'
        ]
        
        # Extract chemistry terms from query
        query_lower = query.lower()
        query_chemistry_terms = [term for term in chemistry_keywords if term in query_lower]
        
        # Calculate keyword scores for all captions
        keyword_scores = []
        for caption in captions:
            caption_lower = caption.lower()
            keyword_count = sum(1 for term in chemistry_keywords if term in caption_lower)
            query_term_bonus = sum(2 for term in query_chemistry_terms if term in caption_lower)
            keyword_scores.append(keyword_count + query_term_bonus)
        
        # Process semantic similarity in batches
        query_embedding = self.semantic_model.encode([query])
        all_semantic_scores = []
        
        batch_size = 50
        for i in range(0, len(captions), batch_size):
            batch_captions = captions[i:i+batch_size]
            caption_embeddings = self.semantic_model.encode(batch_captions)
            
            batch_semantic_scores = []
            for caption_emb in caption_embeddings:
                similarity = np.dot(query_embedding[0], caption_emb) / (
                    np.linalg.norm(query_embedding[0]) * np.linalg.norm(caption_emb)
                )
                batch_semantic_scores.append(similarity)
            
            all_semantic_scores.extend(batch_semantic_scores)
        
        # Normalize scores
        keyword_scores = np.array(keyword_scores)
        all_semantic_scores = np.array(all_semantic_scores)
        original_scores = np.array(original_scores)
        
        # Normalize keyword scores
        if len(keyword_scores) > 0:
            keyword_scores = (keyword_scores - np.min(keyword_scores)) / (np.max(keyword_scores) - np.min(keyword_scores) + 1e-9)
        
        # Normalize original scores
        if len(original_scores) > 0:
            valid_scores = [s for s in original_scores if s is not None]
            if valid_scores:
                min_score = min(valid_scores)
                max_score = max(valid_scores)
                original_scores = [(s - min_score) / (max_score - min_score + 1e-9) if s is not None else 0.0 for s in original_scores]
            else:
                original_scores = [0.0] * len(original_scores)
        original_scores = np.array(original_scores)
        
        # Combine scores
        chemistry_scores = 0.6 * all_semantic_scores + 0.3 * keyword_scores + 0.1 * original_scores
        
        # Sort slides by chemistry-specific scores
        scored_slides = list(zip(slides, chemistry_scores))
        scored_slides.sort(key=lambda x: x[1], reverse=True)
        
        # Return top slides with updated scores
        reranked_slides = []
        for slide, score in scored_slides[:max_slides]:
            slide['rerank_score'] = score
            reranked_slides.append(slide)
        
        print(f"📊 Batch chemistry-specific scores - Semantic: {min(all_semantic_scores):.3f}-{max(all_semantic_scores):.3f}, "
              f"Keywords: {min(keyword_scores):.3f}-{max(keyword_scores):.3f}, "
              f"Final: {min(chemistry_scores):.3f}-{max(chemistry_scores):.3f}")
        return reranked_slides
    
    def _cross_encoder_rerank(self, query: str, slides: List[Dict], captions: List[str], max_slides: int) -> List[Dict]:
        """Rerank using cross-encoder model."""
        # Create query-document pairs
        pairs = [[query, caption] for caption in captions]
        
        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)
        
        # Sort slides by cross-encoder scores
        scored_slides = list(zip(slides, scores))
        scored_slides.sort(key=lambda x: x[1], reverse=True)
        
        # Return top slides with updated scores
        reranked_slides = []
        for slide, score in scored_slides[:max_slides]:
            slide['rerank_score'] = score
            reranked_slides.append(slide)
        
        print(f"📊 Cross-encoder scores range: {min(scores):.3f} - {max(scores):.3f}")
        return reranked_slides
    
    def _semantic_similarity_rerank(self, query: str, slides: List[Dict], captions: List[str], max_slides: int) -> List[Dict]:
        """Rerank using semantic similarity."""
        # Encode query and captions
        query_embedding = self.semantic_model.encode([query])
        caption_embeddings = self.semantic_model.encode(captions)
        
        # Calculate cosine similarities
        similarities = []
        for caption_emb in caption_embeddings:
            similarity = np.dot(query_embedding[0], caption_emb) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(caption_emb)
            )
            similarities.append(similarity)
        
        # Sort slides by similarity scores
        scored_slides = list(zip(slides, similarities))
        scored_slides.sort(key=lambda x: x[1], reverse=True)
        
        # Return top slides with updated scores
        reranked_slides = []
        for slide, score in scored_slides[:max_slides]:
            slide['rerank_score'] = score
            reranked_slides.append(slide)
        
        print(f"📊 Semantic similarity scores range: {min(similarities):.3f} - {max(similarities):.3f}")
        return reranked_slides
    
    def _hybrid_rerank(self, query: str, slides: List[Dict], captions: List[str], 
                      original_scores: List[float], max_slides: int) -> List[Dict]:
        """Rerank using hybrid approach combining cross-encoder and semantic similarity."""
        # Get cross-encoder scores
        pairs = [[query, caption] for caption in captions]
        cross_scores = self.cross_encoder.predict(pairs)
        
        # Get semantic similarity scores
        query_embedding = self.semantic_model.encode([query])
        caption_embeddings = self.semantic_model.encode(captions)
        
        semantic_scores = []
        for caption_emb in caption_embeddings:
            similarity = np.dot(query_embedding[0], caption_emb) / (
                np.linalg.norm(query_embedding[0]) * np.linalg.norm(caption_emb)
            )
            semantic_scores.append(similarity)
        
        # Better normalization: use sigmoid for cross-encoder scores
        cross_scores = 1 / (1 + np.exp(-cross_scores))  # Sigmoid normalization
        semantic_scores = np.array(semantic_scores)
        original_scores = np.array(original_scores)
        
        # Normalize original scores to [0, 1] if they're not already
        if len(original_scores) > 0:
            # Filter out None values for normalization
            valid_scores = [s for s in original_scores if s is not None]
            if valid_scores:
                min_score = min(valid_scores)
                max_score = max(valid_scores)
                original_scores = [(s - min_score) / (max_score - min_score + 1e-9) if s is not None else 0.0 for s in original_scores]
            else:
                original_scores = [0.0] * len(original_scores)
        original_scores = np.array(original_scores)
        
        # Combine scores (weighted average) - give more weight to semantic similarity for chemistry
        hybrid_scores = 0.3 * cross_scores + 0.5 * semantic_scores + 0.2 * original_scores
        
        # Sort slides by hybrid scores
        scored_slides = list(zip(slides, hybrid_scores))
        scored_slides.sort(key=lambda x: x[1], reverse=True)
        
        # Return top slides with updated scores
        reranked_slides = []
        for slide, score in scored_slides[:max_slides]:
            slide['rerank_score'] = score
            reranked_slides.append(slide)
        
        print(f"📊 Hybrid scores - Cross: {min(cross_scores):.3f}-{max(cross_scores):.3f}, "
              f"Semantic: {min(semantic_scores):.3f}-{max(semantic_scores):.3f}, "
              f"Final: {min(hybrid_scores):.3f}-{max(hybrid_scores):.3f}")
        return reranked_slides

class NoReranker:
    """Dummy reranker that returns slides without reranking."""
    
    def __init__(self, **kwargs):
        pass
    
    def rerank(self, query: str, retrieved_slides: List[Dict], **kwargs) -> List[Dict]:
        """Return slides without reranking."""
        return retrieved_slides[:kwargs.get('max_slides', 5)] 