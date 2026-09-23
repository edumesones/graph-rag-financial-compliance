"""
Layer 2: Enrutamiento Inteligente (The Brain) - Docker Edition

Analiza la pregunta del usuario y decide:
- ¿Búsqueda semántica (vectors)?
- ¿Consulta exacta (SQL/Cypher)?
- ¿Búsqueda de relaciones (Graph)?
- ¿Combinación (hybrid)?

Features:
- LLM-based routing (if LLM provided)
- Rule-based fallback
- Redis caching for routing decisions

Author: Glemes
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from ..monitoring.prompt_logger import get_prompt_logger
from ..monitoring.error_tracker import get_error_tracker
from ..db.redis_cache import get_redis


class QueryType(Enum):
    """Tipos de búsqueda disponibles."""
    SEMANTIC = "semantic"  # Vector similarity search
    GRAPH = "graph"  # Entity relationship queries
    HYBRID = "hybrid"  # Combination of both
    KEYWORD = "keyword"  # Exact keyword matching


@dataclass
class RoutingDecision:
    """Decisión de enrutamiento."""
    query_type: QueryType
    reasoning: str
    confidence: float  # 0.0 to 1.0
    recommended_tools: List[str]
    metadata: Dict[str, Any]


class RoutingLayer:
    """
    Capa 2: Análisis y enrutamiento de consultas.

    Decide qué herramientas usar basándose en el tipo de pregunta.
    """

    def __init__(self, llm: Optional[Any] = None, use_cache: bool = True):
        """
        Initialize routing layer.

        Args:
            llm: Optional LLM for advanced routing (if None, uses rule-based)
            use_cache: Whether to use Redis caching (default: True)
        """
        self.logger = get_prompt_logger()
        self.error_tracker = get_error_tracker()
        self.redis_cache = get_redis() if use_cache else None
        self.llm = llm
        self.use_cache = use_cache

        print("\n[Layer 2: Routing] Initializing...")

        if self.llm:
            print(f"   LLM-based routing enabled")
        else:
            print(f"   Using rule-based routing (LLM not provided)")

        if self.redis_cache:
            print(f"   Redis cache available")
        else:
            print(f"   Redis cache not available")

        print("   Layer 2 initialized\n")
    
    def route_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Analyze query and decide routing strategy with Redis caching.

        Args:
            query: User's query
            context: Additional context (company, domain, etc.)

        Returns:
            RoutingDecision with strategy
        """
        print(f"\n[Layer 2] Routing query...")
        print(f"   Query: {query[:100]}...")

        # Generate cache key
        cache_key = self._generate_cache_key(query, context)

        # Check cache first
        if self.redis_cache:
            try:
                import asyncio
                cached = asyncio.run(self.redis_cache.get(cache_key))
                if cached:
                    print(f"   Cache HIT! Returning cached routing decision")
                    # Reconstruct RoutingDecision from cached dict
                    return RoutingDecision(
                        query_type=QueryType(cached["query_type"]),
                        reasoning=cached["reasoning"],
                        confidence=cached["confidence"],
                        recommended_tools=cached["recommended_tools"],
                        metadata={**cached.get("metadata", {}), "from_cache": True},
                    )
            except Exception as e:
                print(f"   Cache read error: {e}")

        print(f"   Cache MISS - executing routing")
        start_time = time.time()

        try:
            if self.llm:
                decision = self._route_with_llm(query, context)
            else:
                decision = self._route_with_rules(query, context)

            elapsed = (time.time() - start_time) * 1000

            print(f"   Route decided: {decision.query_type.value}")
            print(f"   Confidence: {decision.confidence:.2f}")
            print(f"   Tools: {', '.join(decision.recommended_tools)}")
            print(f"   Reasoning: {decision.reasoning}")
            print(f"   ⏱Routing completed ({elapsed:.0f}ms)")

            # Cache the result
            if self.redis_cache:
                try:
                    cache_data = {
                        "query_type": decision.query_type.value,
                        "reasoning": decision.reasoning,
                        "confidence": decision.confidence,
                        "recommended_tools": decision.recommended_tools,
                        "metadata": decision.metadata,
                    }
                    # Cache for 1 hour (3600 seconds)
                    import asyncio
                    asyncio.run(self.redis_cache.set(cache_key, cache_data, ttl=3600))
                    print(f"   Routing decision cached")
                except Exception as e:
                    print(f"   Cache write error: {e}")

            # Log decision
            self.logger.log_prompt(
                layer="layer2_routing",
                prompt=self._format_routing_prompt(query, context),
                response=json.dumps({
                    "query_type": decision.query_type.value,
                    "confidence": decision.confidence,
                    "tools": decision.recommended_tools,
                    "reasoning": decision.reasoning,
                }),
                metadata={"method": "llm" if self.llm else "rules", "from_cache": False},
                latency_ms=elapsed,
            )

            return decision

        except Exception as e:
            self.error_tracker.track_error(
                "layer2_routing",
                e,
                context={"query": query[:200], "context": context}
            )
            # Fallback to semantic search
            return RoutingDecision(
                query_type=QueryType.SEMANTIC,
                reasoning="Error occurred, falling back to semantic search",
                confidence=0.5,
                recommended_tools=["VectorSearch"],
                metadata={"error": str(e)}
            )
    
    def _route_with_llm(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> RoutingDecision:
        """
        Use LLM to make intelligent routing decision.
        """
        prompt = self._build_llm_routing_prompt(query, context)
        
        try:
            # Invoke LLM
            response = self.llm.invoke(prompt)
            
            # Parse response
            # Expected format: JSON with query_type, reasoning, confidence
            decision = self._parse_llm_response(response.content if hasattr(response, 'content') else str(response))
            
            return decision
            
        except Exception as e:
            print(f"   LLM routing failed, falling back to rules: {e}")
            return self._route_with_rules(query, context)
    
    def _build_llm_routing_prompt(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for LLM routing decision."""
        context_str = json.dumps(context) if context else "None"
        
        return f"""You are a query routing expert. Analyze the user's query and decide the best search strategy.

Available query types:
1. SEMANTIC - For conceptual questions, summaries, general information (uses vector embeddings)
2. GRAPH - For relationship questions, connections between entities (uses knowledge graph)
3. HYBRID - For complex questions requiring both approaches
4. KEYWORD - For exact term matching, specific IDs or numbers

Query: {query}
Context: {context_str}

Analyze the query and respond with JSON:
{{
  "query_type": "semantic|graph|hybrid|keyword",
  "reasoning": "Explanation of why this type is best",
  "confidence": 0.0-1.0,
  "recommended_tools": ["VectorSearch", "GraphQuery"]
}}

Think step by step:
1. What is the user asking for?
2. Does it require finding connections (graph) or similar content (semantic)?
3. Is it a complex multi-step question (hybrid)?
4. Does it mention specific entities or relationships?

Respond ONLY with the JSON object, no other text."""
    
    def _parse_llm_response(self, response: str) -> RoutingDecision:
        """Parse LLM response into RoutingDecision."""
        try:
            # Extract JSON from response
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end]
            
            data = json.loads(json_str)
            
            return RoutingDecision(
                query_type=QueryType(data.get('query_type', 'semantic')),
                reasoning=data.get('reasoning', ''),
                confidence=float(data.get('confidence', 0.7)),
                recommended_tools=data.get('recommended_tools', ['VectorSearch']),
                metadata=data.get('metadata', {})
            )
        except Exception as e:
            print(f"   Failed to parse LLM response: {e}")
            # Fallback
            return RoutingDecision(
                query_type=QueryType.SEMANTIC,
                reasoning="Failed to parse LLM response, defaulting to semantic",
                confidence=0.5,
                recommended_tools=["VectorSearch"],
                metadata={"parse_error": str(e)}
            )
    
    def _route_with_rules(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> RoutingDecision:
        """
        Rule-based routing (fallback when no LLM).
        
        Rules:
        - Keywords like "relationship", "connected", "link" → GRAPH
        - Keywords like "similar", "about", "explain" → SEMANTIC
        - Complex multi-part questions → HYBRID
        - Simple lookups → KEYWORD
        """
        query_lower = query.lower()
        
        # Graph indicators
        graph_keywords = [
            "relationship", "connected", "link", "relate",
            "connection", "between", "involve", "associate"
        ]
        
        # Semantic indicators
        semantic_keywords = [
            "what", "explain", "describe", "summarize",
            "tell me about", "information about", "similar to"
        ]
        
        # Keyword indicators
        keyword_indicators = [
            "id:", "number", "code:", "exact",
            "specific", "particular"
        ]
        
        # Check for indicators
        has_graph = any(kw in query_lower for kw in graph_keywords)
        has_semantic = any(kw in query_lower for kw in semantic_keywords)
        has_keyword = any(kw in query_lower for kw in keyword_indicators)
        
        # Decide
        if has_graph and has_semantic:
            return RoutingDecision(
                query_type=QueryType.HYBRID,
                reasoning="Query mentions both relationships and semantic concepts",
                confidence=0.8,
                recommended_tools=["VectorSearch", "GraphQuery"],
                metadata={"method": "rule-based"}
            )
        
        elif has_graph:
            return RoutingDecision(
                query_type=QueryType.GRAPH,
                reasoning="Query focuses on entity relationships",
                confidence=0.7,
                recommended_tools=["GraphQuery"],
                metadata={"method": "rule-based"}
            )
        
        elif has_keyword:
            return RoutingDecision(
                query_type=QueryType.KEYWORD,
                reasoning="Query requests specific exact matches",
                confidence=0.9,
                recommended_tools=["VectorSearch"],  # With exact filtering
                metadata={"method": "rule-based"}
            )
        
        else:
            # Default to semantic
            return RoutingDecision(
                query_type=QueryType.SEMANTIC,
                reasoning="General information query, best served by semantic search",
                confidence=0.6,
                recommended_tools=["VectorSearch"],
                metadata={"method": "rule-based"}
            )
    
    def _generate_cache_key(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate Redis cache key from query and context.

        Args:
            query: User's query
            context: Optional context dictionary

        Returns:
            Cache key string
        """
        import hashlib

        # Create unique hash from query + context
        key_components = [
            query,
            json.dumps(context, sort_keys=True) if context else "",
        ]
        key_string = "|".join(key_components)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"routing:{key_hash}"

    def _format_routing_prompt(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Format prompt for logging."""
        return f"ROUTING QUERY\nQuery: {query}\nContext: {json.dumps(context) if context else 'None'}"

