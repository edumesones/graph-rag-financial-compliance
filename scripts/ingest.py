#!/usr/bin/env python3
"""
Data Ingestion Pipeline for Fintech Agentic RAG - Docker Edition

Downloads SEC filings, extracts entities, and populates:
1. Neo4j knowledge graph (entity relationships)
2. ChromaDB vector store (document embeddings)

Author: Glemes

LLM: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B:novita (via HuggingFace)
Embeddings: BAAI/bge-large-en-v1.5 (Local)

Usage:
    python scripts/ingest.py --company TSLA --limit 5
    python scripts/ingest.py --company AAPL --limit 10
"""

import os
import json
import glob
import argparse
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

# LangChain imports
from langchain.schema import Document
from sec_edgar_downloader import Downloader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI


# ============================================================================
# Helper Classes
# ============================================================================

@dataclass
class SemanticChunk:
    """Represents a semantic chunk with metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    start_pos: int = 0
    end_pos: int = 0


class AgenticChunker:
    """Intelligent chunking system that creates semantically meaningful chunks."""

    def __init__(self, llm: Any, max_chunk_size: int = 3000, min_chunk_size: int = 500):
        self.llm = llm
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[SemanticChunk]:
        """Create semantic chunks from document text."""
        if not text or len(text.strip()) < self.min_chunk_size:
            return []

        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_id = 0
        start_pos = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) > self.max_chunk_size and len(current_chunk) >= self.min_chunk_size:
                chunks.append(SemanticChunk(
                    content=current_chunk,
                    metadata=metadata.copy(),
                    chunk_id=f"{metadata.get('source', 'unknown')}_{chunk_id}",
                    start_pos=start_pos,
                    end_pos=start_pos + len(current_chunk)
                ))
                chunk_id += 1
                start_pos += len(current_chunk)
                current_chunk = para
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para

        if len(current_chunk) >= self.min_chunk_size:
            chunks.append(SemanticChunk(
                content=current_chunk,
                metadata=metadata.copy(),
                chunk_id=f"{metadata.get('source', 'unknown')}_{chunk_id}",
                start_pos=start_pos,
                end_pos=start_pos + len(current_chunk)
            ))

        return chunks

    def chunks_to_langchain_documents(self, chunks: List[SemanticChunk]) -> List[Document]:
        """Convert semantic chunks to LangChain Documents."""
        docs = []
        for chunk in chunks:
            metadata = chunk.metadata.copy()
            metadata['chunk_id'] = chunk.chunk_id
            metadata['start_pos'] = chunk.start_pos
            metadata['end_pos'] = chunk.end_pos
            docs.append(Document(page_content=chunk.content, metadata=metadata))
        return docs


class SECTextExtractor:
    """Intelligent SEC filing text extractor."""

    def __init__(self):
        pass

    def extract_from_filing_directory(self, filing_dir: str) -> Optional[str]:
        """Extract text from a filing directory."""
        if not os.path.exists(filing_dir):
            print(f"   Directory not found: {filing_dir}")
            return None

        file_priorities = [
            ("full-submission.txt", self._extract_txt),
            ("*.txt", self._extract_txt),
            ("*.html", self._extract_html),
            ("*.htm", self._extract_html),
        ]

        for pattern, extractor in file_priorities:
            files = glob.glob(os.path.join(filing_dir, pattern))
            if files:
                file_path = files[0]
                print(f"   Extracting from: {os.path.basename(file_path)}")
                text = extractor(file_path)
                if text and len(text.strip()) > 100:
                    return text

        print(f"   No readable files found in {filing_dir}")
        return None

    def _extract_txt(self, file_path: str) -> Optional[str]:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"   Error reading TXT: {e}")
            return None

    def _extract_html(self, file_path: str) -> Optional[str]:
        """Extract text from HTML file."""
        try:
            try:
                from bs4 import BeautifulSoup
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text()
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = '\n'.join(chunk for chunk in chunks if chunk)
                    return text
            except ImportError:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception as e:
            print(f"   Error reading HTML: {e}")
            return None


# ============================================================================
# Main Ingestion Function (Refactored from Modal)
# ============================================================================

def ingest_sec_filings(
    company_ticker: str,
    limit: int = 5,
    vectors_dir: str = "./vectors",
    data_dir: str = "./data"
) -> Dict[str, Any]:
    """
    Download and process SEC filings for a company.

    Args:
        company_ticker: Stock ticker (e.g., 'TSLA', 'AAPL')
        limit: Number of filings to download
        vectors_dir: Directory for vector store (default: ./vectors)
        data_dir: Directory for downloaded data (default: ./data)

    Returns:
        Dict with ingestion statistics
    """
    import sys

    print(f"Starting ingestion for {company_ticker}...")

    # Create directories if they don't exist
    os.makedirs(vectors_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # 1. Download SEC filings
    print(f"Downloading {limit} 10-K filings...")
    # SEC EDGAR fair-access policy requires a real contact in the User-Agent.
    # Requests with a placeholder are throttled or blocked, so fail loudly instead.
    user_agent = os.environ.get("SEC_USER_AGENT")
    if not user_agent:
        raise SystemExit(
            "SEC_USER_AGENT is not set. SEC EDGAR requires a User-Agent that "
            "identifies the requester, e.g.: "
            'SEC_USER_AGENT="Your Name your.email@example.com"'
        )

    download_dir = os.path.join(data_dir, "sec-data")

    os.makedirs(download_dir, exist_ok=True)
    print(f"📁 Download directory: {download_dir}")

    # Initialize downloader
    print(f"🔧 Initializing Downloader...")
    dl = Downloader(download_dir, user_agent)
    print(f"✅ Downloader initialized")

    try:
        print(f"⬇️  Starting download for {company_ticker}...")
        result = dl.get("10-K", company_ticker, limit=limit)
        print(f"✅ Download call completed")
    except Exception as e:
        print(f"❌ Download error: {e}")
        import traceback
        traceback.print_exc()

    # 2. Extract text from filings
    print("\n📄 Extracting clean text from filings...")
    documents = []

    extractor = SECTextExtractor()

    # Search for downloaded filings
    search_paths = [
        os.path.join(download_dir, "sec-edgar-filings"),
        download_dir,
        os.path.expanduser("~/sec-edgar-filings"),
    ]

    filing_dirs = []
    for search_path in search_paths:
        if os.path.exists(search_path):
            company_dirs = glob.glob(f"{search_path}/{company_ticker}/**/", recursive=True)
            filing_dirs.extend([d for d in company_dirs if os.path.isdir(d)])

    print(f"\n📊 Found {len(filing_dirs)} filing directories")

    if len(filing_dirs) == 0:
        print("⚠️  WARNING: No filing directories found! Check download location.")
        return {
            "status": "error",
            "error": "No filing directories found after download",
            "company_ticker": company_ticker,
        }

    # Extract text from each filing
    for filing_dir in filing_dirs[:limit]:
        print(f"\n📂 Processing: {filing_dir}")

        text = extractor.extract_from_filing_directory(filing_dir)

        if text and len(text.strip()) > 100:
            print(f"   ✅ Using full document ({len(text)} chars)")

            documents.append({
                "text": text,
                "source": filing_dir,
                "company": company_ticker
            })
            print(f"   ✅ Extracted {len(text)} characters")
        else:
            print(f"   ⚠️  No readable text found in this filing")

    print(f"\n✅ Extracted {len(documents)} documents with clean text")

    # 3. Extract entities using LLM
    print(f"Extracting entities with deepseek-ai/DeepSeek-R1-0528-Qwen3-8B:novita...")
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", "")

    if not hf_token:
        print("⚠️  WARNING: HUGGINGFACE_TOKEN not set in environment")

    llm = ChatOpenAI(
        model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B:novita",
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
        temperature=0.1,
        max_tokens=4096,
    )

    entities_list = []
    for doc in documents:
        extraction_prompt = f"""Extract financial compliance entities from this SEC filing excerpt.

Text:
{doc['text'][:20000]}

You MUST respond with ONLY a valid JSON object. No explanations, no markdown, no code blocks.
Format:
{{"company": "Full company name", "violations": ["violation 1", "violation 2"], "regulations": ["regulation 1", "regulation 2"], "dates": ["2023-01-15", "2023-06-30"], "risks": ["risk 1", "risk 2"]}}

JSON:"""

        try:
            response = llm.invoke(extraction_prompt)
            content = response.content.strip() if hasattr(response, 'content') else str(response).strip()

            # Try to find JSON in response
            if '{' in content:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                content = content[json_start:json_end]

            entities = json.loads(content)
            entities_list.append(entities)
            print(f"Extracted entities: {len(entities.get('violations', []))} violations, {len(entities.get('regulations', []))} regulations")
        except Exception as e:
            print(f"Entity extraction error: {e}")
            entities_list.append({
                "company": company_ticker,
                "violations": [],
                "regulations": [],
                "dates": [],
                "risks": []
            })

    # 4. Create semantic chunks
    print("Creating semantic chunks with Agentic Chunking...")

    agentic_chunker = AgenticChunker(
        llm=llm,
        max_chunk_size=3000,
        min_chunk_size=500
    )

    all_chunks = []
    for doc in documents:
        metadata = {"source": doc["source"], "company": doc["company"]}
        semantic_chunks = agentic_chunker.chunk_document(doc["text"], metadata)

        langchain_docs = agentic_chunker.chunks_to_langchain_documents(semantic_chunks)
        all_chunks.extend(langchain_docs)

    splits = all_chunks
    print(f"Created {len(splits)} semantic chunks")

    # 5. Initialize embeddings
    print(f"Loading embedding model: BAAI/bge-large-en-v1.5")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        cache_folder=os.path.join(vectors_dir, ".cache"),
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 6. Embed and store in ChromaDB
    print("\n" + "="*60)
    print("📦 Step 6: Embedding and storing documents in ChromaDB...")
    print("="*60)

    # Check if collection exists
    try:
        existing_vectorstore = Chroma(
            persist_directory=vectors_dir,
            embedding_function=embeddings,
            collection_name="fintech-rag-demo"
        )
        existing_count = existing_vectorstore._collection.count()
        print(f"\n📊 Existing collection found: {existing_count} documents")

        if existing_count > 0:
            print(f"🔍 Checking for duplicates...")
            try:
                existing_docs = existing_vectorstore.get(include=["metadatas"])
                existing_sources = set()
                if existing_docs and "metadatas" in existing_docs:
                    for metadata in existing_docs["metadatas"]:
                        if metadata and "source" in metadata:
                            existing_sources.add(metadata["source"])

                print(f"   Found {len(existing_sources)} unique sources")

                new_splits = []
                skipped_count = 0
                for split in splits:
                    split_source = split.metadata.get("source", "")
                    if split_source not in existing_sources:
                        new_splits.append(split)
                    else:
                        skipped_count += 1

                print(f"   New documents to add: {len(new_splits)}")
                print(f"   Skipped (already exist): {skipped_count}")

                if len(new_splits) > 0:
                    print(f"✅ Adding {len(new_splits)} new documents...")
                    existing_vectorstore.add_documents(new_splits)
                    vectorstore = existing_vectorstore
                    final_count = existing_vectorstore._collection.count()
                    print(f"✅ Added {len(new_splits)} documents. Total: {final_count}")
                else:
                    print(f"⚠️  All documents already exist. Nothing to add.")
                    vectorstore = existing_vectorstore
                    print(f"   Total documents: {existing_count}")

            except Exception as e:
                print(f"⚠️  Could not check for duplicates: {e}")
                print(f"   Adding all documents (may create duplicates)...")
                existing_vectorstore.add_documents(splits)
                vectorstore = existing_vectorstore
                print(f"✅ Added {len(splits)} documents. Total: {existing_vectorstore._collection.count()}")
        else:
            print(f"⚠️  Collection exists but is empty. Creating new collection...")
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=vectors_dir,
                collection_name="fintech-rag-demo"
            )
            print(f"✅ Created collection with {len(splits)} documents")
    except Exception as e:
        print(f"\n📝 Collection doesn't exist. Creating new collection...")
        print(f"   Embedding {len(splits)} documents (this may take a few minutes)...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=vectors_dir,
            collection_name="fintech-rag-demo"
        )
        print(f"✅ Created collection with {len(splits)} documents")

    print(f"\n✅ Vector store persisted to {vectors_dir}")
    print(f"   Total documents in collection: {vectorstore._collection.count()}")

    # 7. Build knowledge graph in Neo4j
    print("\n" + "="*60)
    print("🔗 Step 7: Building knowledge graph in Neo4j...")
    print("="*60)

    # Get Neo4j credentials from environment
    neo4j_uri = os.environ.get("NEO4J_URI", "")
    neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
    neo4j_password = os.environ.get("NEO4J_PASSWORD", "")

    print(f"\n📋 Neo4j Configuration:")
    print(f"   URI: {neo4j_uri if neo4j_uri else 'NOT SET'}")
    print(f"   User: {neo4j_user}")
    print(f"   Password: {'*' * len(neo4j_password) if neo4j_password else 'NOT SET'}")

    if not neo4j_uri or not neo4j_password:
        print("\n❌ ERROR: Neo4j credentials not configured!")
        print("\n💡 Solution:")
        print("   Set environment variables:")
        print("   export NEO4J_URI=bolt://neo4j:7687")
        print("   export NEO4J_USER=neo4j")
        print("   export NEO4J_PASSWORD=your-password")
        print("\n⚠️  Vector store is SAFE - it's persisted")
        print("   Only the graph database connection failed")
        return {
            "status": "partial_success",
            "vector_store": "persisted",
            "graph_database": "failed_connection",
            "message": "Vector store saved successfully. Graph database connection failed - check Neo4j credentials."
        }

    # Test connection
    print(f"\n🔌 Testing Neo4j connection...")
    driver = None
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        with driver.session() as test_session:
            result = test_session.run("RETURN 1 as test")
            test_value = result.single()["test"]
            if test_value == 1:
                print(f"   ✅ Connection successful!")
            else:
                raise Exception("Connection test returned unexpected value")

    except Exception as e:
        print(f"\n❌ ERROR: Neo4j connection failed: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Verify Neo4j is running: docker-compose ps")
        print(f"   2. Check URI: {neo4j_uri}")
        print(f"   3. Verify credentials")
        print("\n⚠️  Vector store is SAFE")
        if driver:
            try:
                driver.close()
            except:
                pass
        return {
            "status": "partial_success",
            "vector_store": "persisted",
            "graph_database": "failed_connection",
            "error": str(e),
            "message": "Vector store saved successfully. Graph database connection failed."
        }

    # Create graph
    print(f"\n📊 Creating graph indexes and nodes...")
    try:
        with driver.session() as session:
            # Create indexes
            print(f"   Creating indexes...")
            session.run("CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.name)")
            session.run("CREATE INDEX regulation_name IF NOT EXISTS FOR (r:Regulation) ON (r.name)")
            print(f"   ✅ Indexes created")

            # Insert entities
            print(f"   Inserting {len(entities_list)} entity sets...")
            total_nodes = 0
            for idx, entities in enumerate(entities_list):
                company_name = entities.get("company", company_ticker)

                # Create company node
                session.run("""
                    MERGE (c:Company {name: $company_name})
                    SET c.ticker = $ticker
                """, company_name=company_name, ticker=company_ticker)
                total_nodes += 1

                # Create regulation nodes
                regulations = entities.get("regulations", [])
                for regulation in regulations:
                    session.run("""
                        MATCH (c:Company {name: $company_name})
                        MERGE (r:Regulation {name: $regulation})
                        MERGE (c)-[:SUBJECT_TO]->(r)
                    """, company_name=company_name, regulation=regulation)
                    total_nodes += 1

                # Create violation nodes
                violations = entities.get("violations", [])
                for i, violation in enumerate(violations):
                    date = entities.get("dates", [None])[i] if i < len(entities.get("dates", [])) else None
                    session.run("""
                        MATCH (c:Company {name: $company_name})
                        CREATE (v:Violation {description: $violation, date: $date})
                        CREATE (c)-[:HAD_VIOLATION]->(v)
                    """, company_name=company_name, violation=violation, date=date)
                    total_nodes += 1

                # Create risk nodes
                risks = entities.get("risks", [])
                for risk in risks:
                    session.run("""
                        MATCH (c:Company {name: $company_name})
                        MERGE (r:Risk {description: $risk})
                        MERGE (c)-[:HAS_RISK]->(r)
                    """, company_name=company_name, risk=risk)
                    total_nodes += 1

                if (idx + 1) % 10 == 0:
                    print(f"   Processed {idx + 1}/{len(entities_list)} entity sets...")

            print(f"✅ Graph database populated!")
            print(f"   Total nodes created: ~{total_nodes}")

            driver.close()
    except Exception as e:
        print(f"❌ ERROR building knowledge graph: {e}")
        print(f"\n⚠️  Vector store is SAFE")
        try:
            driver.close()
        except:
            pass
        return {
            "status": "partial_success",
            "vector_store": "persisted",
            "graph_database": "failed_insertion",
            "error": str(e),
            "message": "Vector store saved successfully. Graph database insertion failed."
        }

    print("✅ Knowledge graph built successfully")

    # 8. Return statistics
    return {
        "status": "success",
        "company_ticker": company_ticker,
        "filings_downloaded": len(filing_dirs),
        "documents_processed": len(documents),
        "text_chunks_created": len(splits),
        "entities_extracted": len(entities_list),
        "vector_embeddings": len(splits),
        "graph_nodes_created": f"~{total_nodes}",
        "llm_model": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B:novita",
        "embedding_model": "BAAI/bge-large-en-v1.5",
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """
    CLI entry point for ingestion pipeline.

    Usage:
        python scripts/ingest.py --company TSLA --limit 5
        python scripts/ingest.py --company AAPL --limit 10 --vectors-dir ./custom_vectors
    """
    parser = argparse.ArgumentParser(
        description="Ingest SEC filings into RAG system (ChromaDB + Neo4j)"
    )
    parser.add_argument(
        "--company",
        type=str,
        required=True,
        help="Company ticker symbol (e.g., TSLA, AAPL, MSFT)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of 10-K filings to download (default: 5)"
    )
    parser.add_argument(
        "--vectors-dir",
        type=str,
        default="./vectors",
        help="Directory for vector store (default: ./vectors)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory for downloaded SEC data (default: ./data)"
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"INGESTION PIPELINE: {args.company}")
    print(f"LLM: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B:novita")
    print(f"Embeddings: BAAI/bge-large-en-v1.5")
    print(f"{'='*60}\n")

    result = ingest_sec_filings(
        company_ticker=args.company,
        limit=args.limit,
        vectors_dir=args.vectors_dir,
        data_dir=args.data_dir
    )

    print(f"\n{'='*60}")
    print("INGESTION COMPLETE")
    print(f"{'='*60}")
    print(json.dumps(result, indent=2))
    print(f"\n✅ Data ready for querying!")
    print(f"   - Vector store: {args.vectors_dir}")
    print(f"   - Graph database: {os.getenv('NEO4J_URI', 'bolt://neo4j:7687')}")


if __name__ == "__main__":
    main()
