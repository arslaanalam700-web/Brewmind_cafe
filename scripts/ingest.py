"""Master ingestion script for BaristaAI: converts coffee menu into vector & BM25 indices."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.menu_loader import get_all_menu_chunks
from src.ingestion.indexer import build_indices

def run_ingestion():
    print("=" * 70)
    print(" BARISTAAI: COFFEE SHOP MENU INGESTION PIPELINE")
    print("=" * 70)
    
    chunks = get_all_menu_chunks()
    print(f"Loaded {len(chunks)} menu items from data/coffee_menu.json.")
    
    index_results = build_indices(chunks)
    
    print("\n" + "=" * 70)
    print(" MENU INGESTION COMPLETE")
    print(f" Total Menu Items Indexed: {index_results['total_chunks']}")
    print("=" * 70)

if __name__ == "__main__":
    run_ingestion()
