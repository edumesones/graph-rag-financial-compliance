"""
Layer 3: Construcción de Consultas (The Translator)

Optimiza y transforma la consulta del usuario:
- Descomposición en sub-preguntas
- Expansión con sinónimos y términos relacionados
- Traducción a lenguajes específicos (Cypher, SQL, etc.)
- Refinamiento iterativo

Author: Glemes
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..monitoring.prompt_logger import get_prompt_logger
from ..monitoring.error_tracker import get_error_tracker
from .layer2_routing import QueryType


@dataclass
class QueryPlan:
    """Plan de ejecución de consulta."""
    original_query: str
    sub_queries: List[str]
    expanded_terms: List[str]
    translated_queries: Dict[str, str]  # {type: query}
    execution_order: List[int]
    metadata: Dict[str, Any]


class QueryBuilderLayer:
    """
    Capa 3: Construcción y optimización de consultas.
    
    Transforma consultas del usuario en queries optimizadas para cada tool.
    """
    
    def __init__(self, llm: Optional[Any] = None):
        """
        Initialize query builder layer.
        
        Args:
            llm: Optional LLM for query decomposition and expansion
        """
        self.logger = get_prompt_logger()
        self.error_tracker = get_error_tracker()
        self.llm = llm
        
        print("\n🔧 [Layer 3: Query Builder] Initializing...")
        
        if self.llm:
            print(f"   ✅ LLM-based query building enabled")
        else:
            print(f"   ⚠️  Using rule-based query building")
        
        print("   ✅ Layer 3 initialized\n")
    
    def build_query_plan(
        self,
        query: str,
        query_type: QueryType,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """
        Build execution plan for query.
        
        Args:
            query: User's original query
            query_type: Type from routing layer
            context: Additional context
            
        Returns:
            QueryPlan with sub-queries and translations
        """
        print(f"\n🔨 [Layer 3] Building query plan...")
        print(f"   Query: {query[:100]}...")
        print(f"   Type: {query_type.value}")
        
        start_time = time.time()
        
        try:
            # Step 1: Decompose into sub-queries (if complex)
            sub_queries = self._decompose_query(query, context)
            print(f"   ✅ Decomposed into {len(sub_queries)} sub-queries")
            
            # Step 2: Expand with related terms
            expanded_terms = self._expand_terms(query)
            print(f"   ✅ Expanded with {len(expanded_terms)} related terms")
            
            # Step 3: Translate to specific query languages
            translated = self._translate_queries(sub_queries, query_type, context)
            print(f"   ✅ Translated to {len(translated)} query formats")
            
            # Step 4: Determine execution order
            execution_order = list(range(len(sub_queries)))
            
            plan = QueryPlan(
                original_query=query,
                sub_queries=sub_queries,
                expanded_terms=expanded_terms,
                translated_queries=translated,
                execution_order=execution_order,
                metadata={
                    "query_type": query_type.value,
                    "complexity": len(sub_queries),
                }
            )
            
            elapsed = (time.time() - start_time) * 1000
            print(f"   ⏱️  Query planning completed ({elapsed:.0f}ms)")
            
            # Log plan
            self.logger.log_prompt(
                layer="layer3_query_builder",
                prompt=self._format_planning_prompt(query, query_type, context),
                response=json.dumps({
                    "sub_queries": sub_queries,
                    "expanded_terms": expanded_terms,
                    "translations": translated,
                }, indent=2),
                metadata={"query_type": query_type.value},
                latency_ms=elapsed,
            )
            
            return plan
            
        except Exception as e:
            self.error_tracker.track_error(
                "layer3_query_builder",
                e,
                context={"query": query[:200], "type": query_type.value}
            )
            # Fallback: simple plan
            return QueryPlan(
                original_query=query,
                sub_queries=[query],
                expanded_terms=[],
                translated_queries={"semantic": query},
                execution_order=[0],
                metadata={"error": str(e), "fallback": True}
            )
    
    def _decompose_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """
        Decompose complex query into sub-queries.
        
        Examples:
        - "What violations did Tesla have and how do they compare to Ford?"
          → ["What violations did Tesla have?", "What violations did Ford have?", "Compare violations"]
        """
        if self.llm:
            return self._decompose_with_llm(query, context)
        else:
            return self._decompose_with_rules(query)
    
    def _decompose_with_llm(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Use LLM to decompose query."""
        prompt = f"""Break down this complex query into simpler sub-queries that can be answered independently.

Query: {query}
Context: {json.dumps(context) if context else 'None'}

Rules:
1. Keep sub-queries focused and atomic
2. Maintain logical order
3. Each sub-query should be self-contained
4. If query is already simple, return it as-is

Respond with JSON array:
["sub-query 1", "sub-query 2", ...]

If the query is simple enough, return:
["{query}"]
"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON array
            start = content.find('[')
            end = content.rfind(']') + 1
            json_str = content[start:end]
            
            sub_queries = json.loads(json_str)
            return sub_queries if sub_queries else [query]
            
        except Exception as e:
            print(f"   ⚠️  LLM decomposition failed: {e}")
            return [query]
    
    def _decompose_with_rules(self, query: str) -> List[str]:
        """Rule-based decomposition."""
        # Simple heuristic: split on "and", "also", "additionally"
        separators = [" and ", " also ", " additionally ", ", "]
        
        for sep in separators:
            if sep in query.lower():
                parts = query.split(sep)
                # Filter out very short parts
                parts = [p.strip() for p in parts if len(p.strip()) > 10]
                if len(parts) > 1:
                    return parts
        
        # No decomposition needed
        return [query]
    
    def _expand_terms(self, query: str) -> List[str]:
        """
        Expand query with synonyms and related terms.
        
        For now, uses simple keyword extraction.
        In production, would use word embeddings or LLM.
        """
        # Simple expansion: extract key terms
        words = query.lower().split()
        
        # Financial domain synonyms
        synonyms = {
            "violation": ["breach", "infringement", "non-compliance"],
            "regulation": ["rule", "compliance", "requirement"],
            "company": ["corporation", "firm", "business"],
            "financial": ["fiscal", "monetary", "economic"],
        }
        
        expanded = []
        for word in words:
            if word in synonyms:
                expanded.extend(synonyms[word])
        
        return list(set(expanded))[:10]  # Aumentado de 5 a 10 términos expandidos
    
    def _translate_queries(
        self,
        sub_queries: List[str],
        query_type: QueryType,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Translate natural language to specific query languages.
        
        Returns:
            Dictionary with translations: {"semantic": "...", "cypher": "...", etc.}
        """
        translations = {}
        
        # Semantic query (for vector search)
        translations["semantic"] = " ".join(sub_queries)
        
        # Cypher query (for graph search)
        if query_type in [QueryType.GRAPH, QueryType.HYBRID]:
            cypher = self._generate_cypher(sub_queries[0], context)
            if cypher:
                translations["cypher"] = cypher
        
        return translations
    
    def _generate_cypher(
        self,
        query: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Generate Cypher query from natural language.
        
        Simple rule-based for now. In production, would use LLM.
        """
        query_lower = query.lower()
        company = context.get("company", "") if context else ""
        
        # Template-based Cypher generation
        if "violation" in query_lower or "breach" in query_lower:
            if company:
                return f"MATCH (c:Company {{name: '{company}'}})-[:VIOLATED]->(r:Regulation) RETURN r.name, r.date"
            else:
                return "MATCH (c:Company)-[:VIOLATED]->(r:Regulation) RETURN c.name, r.name, r.date"
        
        elif "relationship" in query_lower or "connected" in query_lower:
            if company:
                return f"MATCH (c:Company {{name: '{company}'}})-[r]->(n) RETURN type(r), n LIMIT 20"
            else:
                return "MATCH (c:Company)-[r]->(n) RETURN c.name, type(r), n LIMIT 20"
        
        return None
    
    def _format_planning_prompt(
        self,
        query: str,
        query_type: QueryType,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Format prompt for logging."""
        return f"""QUERY PLANNING
Original Query: {query}
Query Type: {query_type.value}
Context: {json.dumps(context) if context else 'None'}"""

