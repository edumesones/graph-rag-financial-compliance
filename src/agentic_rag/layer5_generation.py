"""
Layer 5: Generación con Citación (The Writer)

Sintetiza la respuesta final:
- Integra resultados de Layer 4
- Genera respuesta coherente con LLM
- Añade citaciones de fuentes
- Maneja casos sin información
- Verifica consistencia

Author: Glemes
"""

import time
import json
from typing import Dict, Any, Optional, List

from ..monitoring.prompt_logger import get_prompt_logger
from ..monitoring.error_tracker import get_error_tracker
from .layer4_retrieval import RetrievalResponse, RetrievalResult


class GenerationLayer:
    """
    Capa 5: Generación de respuesta final.
    
    Sintetiza información recuperada en respuesta coherente con citaciones.
    """
    
    def __init__(self, llm: Any):
        """
        Initialize generation layer.
        
        Args:
            llm: Language model for generation
        """
        self.logger = get_prompt_logger()
        self.error_tracker = get_error_tracker()
        self.llm = llm
        
        print("\n🔧 [Layer 5: Generation] Initializing...")
        print(f"   ✅ LLM configured for generation")
        print("   ✅ Layer 5 initialized\n")
    
    def generate_response(
        self,
        query: str,
        retrieval_response: RetrievalResponse,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate final answer from retrieved documents.
        
        Args:
            query: Original user query
            retrieval_response: Results from Layer 4
            context: Additional context (company, etc.)
            
        Returns:
            Dictionary with answer, citations, metadata
        """
        print(f"\n✍️  [Layer 5] Generating response...")
        print(f"   Query: {query[:80]}...")
        print(f"   Retrieved docs: {len(retrieval_response.results)}")
        
        start_time = time.time()
        
        try:
            # Check if we have results
            if not retrieval_response.results:
                return self._generate_no_results_response(query)
            
            # Build generation prompt
            prompt = self._build_generation_prompt(
                query,
                retrieval_response.results,
                context
            )
            
            print(f"   📝 Prompt built ({len(prompt)} chars)")
            
            # Generate with LLM
            print(f"   🤖 Invoking LLM...")
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            print(f"   ✅ Response generated ({len(answer)} chars)")
            
            # Extract citations
            citations = self._extract_citations(answer, retrieval_response.results)
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                answer,
                retrieval_response,
                citations
            )
            
            elapsed = (time.time() - start_time) * 1000
            print(f"   ⏱️  Generation completed ({elapsed:.0f}ms)")
            print(f"   📊 Confidence: {confidence:.2f}")
            print(f"   📚 Citations: {len(citations)}")
            
            result = {
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "sources": [r.source for r in retrieval_response.results[:5]],
                "metadata": {
                    "generation_time_ms": elapsed,
                    "retrieval_time_ms": retrieval_response.retrieval_time_ms,
                    "total_time_ms": elapsed + retrieval_response.retrieval_time_ms,
                    "documents_used": len(retrieval_response.results),
                    "context": context,
                }
            }
            
            # Log generation
            self.logger.log_prompt(
                layer="layer5_generation",
                prompt=prompt,
                response=answer,
                metadata={
                    "confidence": confidence,
                    "citations": len(citations),
                    "docs_used": len(retrieval_response.results),
                },
                latency_ms=elapsed,
            )
            
            return result
            
        except Exception as e:
            self.error_tracker.track_error(
                "layer5_generation",
                e,
                context={"query": query[:200], "docs": len(retrieval_response.results)}
            )
            
            return {
                "answer": f"Error generating response: {str(e)}",
                "citations": [],
                "confidence": 0.0,
                "sources": [],
                "metadata": {"error": str(e)}
            }
    
    def _build_generation_prompt(
        self,
        query: str,
        results: List[RetrievalResult],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build prompt for LLM generation with retrieved context.
        """
        company = context.get("company", "") if context else ""
        company_context = f"\nCompany Context: {company}" if company else ""
        
        # Format retrieved documents
        docs_text = ""
        for i, result in enumerate(results[:5], 1):  # Use top 5
            source_info = f"[Source {i}: {result.source}]"
            content = result.content[:2000]  # Aumentado de 500 a 2000 chars para más contexto
            docs_text += f"\n\n{source_info}\n{content}..."
        
        prompt = f"""You are a financial analyst expert. Answer the user's question using ONLY the information provided in the documents below.

{company_context}

User Question: {query}

Retrieved Documents:
{docs_text}

INSTRUCTIONS:
1. Answer the question directly and concisely
2. Use ONLY information from the provided documents
3. Cite sources using [Source X] notation
4. If information is not in the documents, say "Based on the available documents, I cannot find information about..."
5. Be factual and precise
6. Include specific numbers, dates, and details when available
7. Structure your answer clearly with bullet points if appropriate

Your Answer:"""
        
        return prompt
    
    def _generate_no_results_response(self, query: str) -> Dict[str, Any]:
        """Generate response when no documents were found."""
        answer = f"""I couldn't find relevant information to answer your question: "{query}"

This could be because:
- The information is not in the indexed documents
- The query terms don't match the available content
- The documents need to be ingested first

Suggestions:
1. Try rephrasing your question with different terms
2. Check if the relevant documents have been ingested
3. Verify the company name and details are correct"""
        
        self.logger.log_prompt(
            layer="layer5_generation",
            prompt=f"No results for query: {query}",
            response=answer,
            metadata={"no_results": True},
        )
        
        return {
            "answer": answer,
            "citations": [],
            "confidence": 0.0,
            "sources": [],
            "metadata": {"no_results": True}
        }
    
    def _extract_citations(
        self,
        answer: str,
        results: List[RetrievalResult]
    ) -> List[Dict[str, Any]]:
        """
        Extract citation information from answer.
        
        Looks for [Source X] patterns in the answer.
        """
        citations = []
        
        for i, result in enumerate(results, 1):
            citation_marker = f"[Source {i}]"
            
            if citation_marker in answer:
                citations.append({
                    "source_id": i,
                    "source": result.source,
                    "content_snippet": result.content[:200],
                    "metadata": result.metadata,
                })
        
        return citations
    
    def _calculate_confidence(
        self,
        answer: str,
        retrieval_response: RetrievalResponse,
        citations: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate confidence score for the answer.
        
        Factors:
        - Number of citations
        - Retrieval scores
        - Answer length and completeness
        """
        confidence = 0.5  # Base confidence
        
        # Factor 1: Citations
        if len(citations) > 0:
            confidence += 0.2
        if len(citations) >= 2:
            confidence += 0.1
        
        # Factor 2: Number of documents
        if len(retrieval_response.results) >= 3:
            confidence += 0.1
        
        # Factor 3: Answer quality indicators
        if len(answer) > 100:
            confidence += 0.05
        if any(word in answer.lower() for word in ["because", "according to", "specifically"]):
            confidence += 0.05
        
        # Factor 4: Check for uncertainty markers
        uncertainty_markers = ["cannot find", "not sure", "unclear", "may be", "possibly"]
        if any(marker in answer.lower() for marker in uncertainty_markers):
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

