"""
Script para eliminar documentos de ChromaDB

Uso:
    python src/delete_documents.py --all                    # Eliminar todos los documentos
    python src/delete_documents.py --source "path/to/file"  # Eliminar por source
    python src/delete_documents.py --company TSLA           # Eliminar por company
    python src/delete_documents.py --ids id1 id2 id3        # Eliminar por IDs específicos
"""

import os
import sys
import argparse
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def delete_all_documents(vectorstore: Chroma) -> int:
    """Eliminar todos los documentos de la colección"""
    count = vectorstore._collection.count()
    vectorstore.delete_collection()
    # Recreate empty collection
    vectorstore = Chroma(
        persist_directory=vectorstore._persist_directory,
        embedding_function=vectorstore._embedding_function,
        collection_name=vectorstore._collection_name
    )
    return count


def delete_by_source(vectorstore: Chroma, source_pattern: str) -> int:
    """Eliminar documentos por source (path del archivo)"""
    # Get all documents with their IDs and metadata
    all_docs = vectorstore.get(include=["metadatas"])
    
    if not all_docs or "metadatas" not in all_docs:
        return 0
    
    ids_to_delete = []
    for i, metadata in enumerate(all_docs["metadatas"]):
        if metadata and "source" in metadata:
            source = metadata["source"]
            if source_pattern in source or source == source_pattern:
                ids_to_delete.append(all_docs["ids"][i])
    
    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)
    
    return len(ids_to_delete)


def delete_by_company(vectorstore: Chroma, company: str) -> int:
    """Eliminar documentos por company"""
    all_docs = vectorstore.get(include=["metadatas"])
    
    if not all_docs or "metadatas" not in all_docs:
        return 0
    
    ids_to_delete = []
    for i, metadata in enumerate(all_docs["metadatas"]):
        if metadata and metadata.get("company") == company:
            ids_to_delete.append(all_docs["ids"][i])
    
    if ids_to_delete:
        vectorstore.delete(ids=ids_to_delete)
    
    return len(ids_to_delete)


def delete_by_ids(vectorstore: Chroma, ids: List[str]) -> int:
    """Eliminar documentos por IDs específicos"""
    # Verify IDs exist
    all_docs = vectorstore.get(include=["metadatas"])
    existing_ids = set(all_docs["ids"]) if all_docs and "ids" in all_docs else set()
    
    valid_ids = [id for id in ids if id in existing_ids]
    
    if valid_ids:
        vectorstore.delete(ids=valid_ids)
    
    return len(valid_ids)


def list_documents(vectorstore: Chroma, limit: int = 10):
    """Listar documentos en la colección"""
    all_docs = vectorstore.get(include=["metadatas", "documents"])
    
    if not all_docs:
        print("No documents found")
        return
    
    total = len(all_docs["ids"])
    print(f"\nDocuments in collection: {total}")
    print(f"   Showing first {min(limit, total)}:\n")
    
    for i in range(min(limit, total)):
        doc_id = all_docs["ids"][i]
        metadata = all_docs["metadatas"][i] if "metadatas" in all_docs else {}
        doc_text = all_docs["documents"][i][:100] + "..." if "documents" in all_docs else ""
        
        print(f"   [{i+1}] ID: {doc_id[:50]}...")
        print(f"       Source: {metadata.get('source', 'N/A')}")
        print(f"       Company: {metadata.get('company', 'N/A')}")
        print(f"       Text preview: {doc_text}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Delete documents from ChromaDB")
    parser.add_argument("--all", action="store_true", help="Delete all documents")
    parser.add_argument("--source", type=str, help="Delete by source (file path)")
    parser.add_argument("--company", type=str, help="Delete by company ticker")
    parser.add_argument("--ids", nargs="+", help="Delete by document IDs")
    parser.add_argument("--list", action="store_true", help="List documents")
    parser.add_argument("--count", action="store_true", help="Show document count")
    parser.add_argument("--persist-dir", type=str, default="/vectors", help="ChromaDB persist directory")
    
    args = parser.parse_args()
    
    print("="*60)
    print("CHROMADB DOCUMENT DELETION")
    print("="*60)
    
    # Initialize embeddings and vectorstore
    print("\nLoading vector store...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        persist_dir = args.persist_dir if os.path.exists(args.persist_dir) else "./vectors"
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="fintech-rag-demo"
        )
        
        initial_count = vectorstore._collection.count()
        print(f"Vector store loaded")
        print(f"   Persist directory: {persist_dir}")
        print(f"   Current documents: {initial_count}")
        
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return
    
    # List documents
    if args.list:
        list_documents(vectorstore)
        return
    
    # Show count only
    if args.count:
        print(f"\nTotal documents: {initial_count}")
        return
    
    # Delete operations
    deleted_count = 0
    
    if args.all:
        print(f"\nWARNING: This will delete ALL {initial_count} documents!")
        response = input("Are you sure? (yes/no): ")
        if response.lower() == "yes":
            deleted_count = delete_all_documents(vectorstore)
            print(f"Deleted {deleted_count} documents")
        else:
            print("Cancelled")
    
    elif args.source:
        print(f"\nSearching for documents with source: {args.source}")
        deleted_count = delete_by_source(vectorstore, args.source)
        print(f"Deleted {deleted_count} documents")
    
    elif args.company:
        print(f"\nSearching for documents with company: {args.company}")
        deleted_count = delete_by_company(vectorstore, args.company)
        print(f"Deleted {deleted_count} documents")
    
    elif args.ids:
        print(f"\nDeleting {len(args.ids)} document IDs...")
        deleted_count = delete_by_ids(vectorstore, args.ids)
        print(f"Deleted {deleted_count} documents")
        if deleted_count < len(args.ids):
            print(f"{len(args.ids) - deleted_count} IDs were not found")
    
    else:
        print("\nUsage examples:")
        print("   python src/delete_documents.py --list                    # List documents")
        print("   python src/delete_documents.py --count                   # Show count")
        print("   python src/delete_documents.py --all                     # Delete all")
        print("   python src/delete_documents.py --company TSLA             # Delete by company")
        print("   python src/delete_documents.py --source '/path/to/file'  # Delete by source")
        print("   python src/delete_documents.py --ids id1 id2 id3         # Delete by IDs")
        return
    
    # Show final count
    if deleted_count > 0:
        final_count = vectorstore._collection.count()
        print(f"\nFinal document count: {final_count}")
        print(f"   Deleted: {deleted_count}")
        print(f"   Remaining: {final_count}")


if __name__ == "__main__":
    main()

