"""
Smart categorization utilities for expenses.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from difflib import SequenceMatcher

from config.settings import BASE_DIR


class ExpenseCategorizer:
    """Handles automatic categorization of expenses based on merchant names and rules."""
    
    def __init__(self, categories_file: Path = None):
        if categories_file is None:
            categories_file = BASE_DIR / "config" / "categories.json"
        
        self.categories_file = categories_file
        self.categories_data = self._load_categories()
        self.merchant_cache = {}  # Cache for previously categorized merchants
        
        # Add specific rules for Japanese stores
        self.store_categories = {
            "maruetsu": "Groceries",
            "マルエツ": "Groceries",
            "イトーヨーカドー": "Groceries",
            "ライフ": "Groceries",
            "サミット": "Groceries",
            "オーケー": "Groceries",
            "イオン": "Groceries",
            "成城石井": "Groceries",
            "コープ": "Groceries",
        }
        # Convert store names to lowercase for case-insensitive matching
        self.store_categories_lower = {k.lower(): v for k, v in self.store_categories.items()}
    
    def _load_categories(self) -> Dict:
        """Load categories configuration from JSON file."""
        try:
            with open(self.categories_file, 'r') as f:
                data = json.load(f)
                # If the file has the old structure, convert it to the new structure
                if "_metadata" not in data:
                    # Create metadata structure
                    metadata = {
                        "categories": {},
                        "auto_categorization_rules": {
                            "min_confidence": 0.6,
                            "fallback_category": "Other",
                            "case_sensitive": False
                        }
                    }
                    # Convert each category to the new format
                    for category, color in data.items():
                        metadata["categories"][category] = {
                            "color": color,
                            "keywords": []
                        }
                    return metadata
                return data["_metadata"]
        except (FileNotFoundError, json.JSONDecodeError):
            # Return default structure if file doesn't exist or is invalid
            return {
                "categories": {
                    "Other": {
                        "color": "#747D8C",
                        "keywords": []
                    }
                },
                "auto_categorization_rules": {
                    "min_confidence": 0.6,
                    "fallback_category": "Other",
                    "case_sensitive": False
                }
            }
    
    def categorize_expense(self, merchant: str, amount: float = None, description: str = "") -> Tuple[str, float]:
        """
        Categorize an expense based on merchant name, amount, and description.
        
        Returns:
            Tuple of (category_name, confidence_score)
        """
        if not merchant:
            return self.categories_data["auto_categorization_rules"]["fallback_category"], 0.5
            
        # Check cache first
        merchant_key = merchant.lower().strip()
        if merchant_key in self.merchant_cache:
            return self.merchant_cache[merchant_key]
        
        # Check specific store rules first - case insensitive
        for store_name in self.store_categories_lower:
            if store_name in merchant_key:
                category = self.store_categories_lower[store_name]
                self.merchant_cache[merchant_key] = (category, 1.0)
                return category, 1.0
        
        # Clean merchant name for better matching
        clean_merchant = self._clean_merchant_name(merchant)
        
        # Try exact keyword matching first
        category, confidence = self._match_by_keywords(clean_merchant, description)
        
        # If no good match, try fuzzy matching
        if confidence < self.categories_data["auto_categorization_rules"]["min_confidence"]:
            fuzzy_category, fuzzy_confidence = self._fuzzy_match_merchant(clean_merchant)
            if fuzzy_confidence > confidence:
                category, confidence = fuzzy_category, fuzzy_confidence
        
        # Use fallback category if confidence is still too low
        min_confidence = self.categories_data["auto_categorization_rules"]["min_confidence"]
        if confidence < min_confidence:
            category = self.categories_data["auto_categorization_rules"]["fallback_category"]
            confidence = 0.5
        
        # Cache the result
        self.merchant_cache[merchant_key] = (category, confidence)
        
        return category, confidence
    
    def _clean_merchant_name(self, merchant: str) -> str:
        """Clean merchant name for better matching."""
        # Remove common suffixes and prefixes
        merchant = re.sub(r'\b(inc|llc|corp|ltd|co)\b', '', merchant, flags=re.IGNORECASE)
        merchant = re.sub(r'\b(the|a|an)\b', '', merchant, flags=re.IGNORECASE)
        
        # Remove special characters and extra spaces
        merchant = re.sub(r'[^\w\s]', ' ', merchant)
        merchant = re.sub(r'\s+', ' ', merchant).strip()
        
        return merchant.lower()
    
    def _match_by_keywords(self, merchant: str, description: str = "") -> Tuple[str, float]:
        """Match expense to category based on keywords."""
        best_category = None
        best_score = 0.0

        # Combine merchant and description for matching
        text_to_match = f"{merchant} {description}".lower()

        for category_name, category_data in self.categories_data["categories"].items():
            for keyword in category_data["keywords"]:
                if keyword.lower() in text_to_match:
                    score = 1.0  # Exact keyword hit
                    if score > best_score:
                        best_category = category_name
                        best_score = score

        # Always return a result to prevent unpacking errors
        return (best_category or self.categories_data["auto_categorization_rules"]["fallback_category"], 
                best_score)
    
    def _fuzzy_match_merchant(self, merchant: str) -> Tuple[str, float]:
        """Fuzzy match merchant name to keywords if exact match fails."""
        best_category = None
        best_score = 0.0

        for category_name, category_data in self.categories_data["categories"].items():
            for keyword in category_data["keywords"]:
                ratio = SequenceMatcher(None, merchant, keyword.lower()).ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_category = category_name
        
        return (best_category or self.categories_data["auto_categorization_rules"]["fallback_category"], 
                best_score)
        