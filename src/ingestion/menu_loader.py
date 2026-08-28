"""Menu Loader: Converts coffee menu items into rich text passages for RAG indexing."""
import json
from pathlib import Path
from typing import List, Dict, Any

from config import MENU_JSON_PATH

def load_menu_items() -> List[Dict[str, Any]]:
    """Load raw menu items from coffee_menu.json."""
    if not MENU_JSON_PATH.exists():
        raise FileNotFoundError(f"Menu dataset not found at {MENU_JSON_PATH}")
        
    with open(MENU_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def convert_item_to_chunk(item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a menu item dictionary into a text chunk with rich metadata."""
    item_id = item["id"]
    name = item["name"]
    category = item.get("category", "General")
    price = item.get("price", 0.0)
    caffeine = item.get("caffeine_level", "medium")
    temp = item.get("temp", "hot/iced")
    flavor = ", ".join(item.get("flavor_notes", []))
    dietary = ", ".join(item.get("dietary_tags", []))
    ingredients = ", ".join(item.get("ingredients", []))
    desc = item.get("description", "")
    pairing = item.get("pairing_suggestion", "")

    # Construct rich text passage for semantic vector & keyword BM25 search
    passage = f"Menu Item: {name} (ID: {item_id})\n"
    passage += f"Category: {category} | Price: ${price:.2f} | Temperature: {temp}\n"
    passage += f"Caffeine Level: {caffeine} | Dietary Suitable Tags: {dietary}\n"
    passage += f"Flavor Notes: {flavor}\n"
    passage += f"Description: {desc}\n"
    passage += f"Ingredients: {ingredients}\n"
    if pairing:
        passage += f"Recommended Pairing: {pairing}\n"

    return {
        "chunk_id": item_id,
        "paper_id": item_id,
        "paper_title": name,
        "page": 1,
        "section": category,
        "text": passage,
        "item_data": item,
        "word_count": len(passage.split())
    }

def get_all_menu_chunks() -> List[Dict[str, Any]]:
    """Load and convert all menu items into indexable chunks."""
    items = load_menu_items()
    return [convert_item_to_chunk(item) for item in items]
