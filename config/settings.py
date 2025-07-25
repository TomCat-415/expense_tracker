"""
Configuration settings for the expense tracker application.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Safe base directory (works in Streamlit Cloud too)
APP_ROOT = Path.cwd()

# Define paths
DATA_DIR = APP_ROOT / "data"
RECEIPTS_DIR = APP_ROOT / "receipts"
PROCESSED_RECEIPTS_DIR = RECEIPTS_DIR / "processed"
FAILED_RECEIPTS_DIR = RECEIPTS_DIR / "failed"
CACHE_DIR = DATA_DIR / "cache"

# Ensure directories exist
for d in [DATA_DIR, RECEIPTS_DIR, PROCESSED_RECEIPTS_DIR, FAILED_RECEIPTS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Database settings
DATABASE_FILE = DATA_DIR / "expenses.db"

# API Keys (from environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_CLOUD_VISION_KEY = os.getenv("GOOGLE_CLOUD_VISION_KEY")

# OCR Settings
OCR_CONFIDENCE_THRESHOLD = 0.7
MAX_IMAGE_SIZE_MB = 10

# Receipt processing
ALLOWED_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
MAX_RECEIPT_AGE_DAYS = 90  # How long to keep processed receipts

# App settings
APP_TITLE = "Family Expense Tracker"
APP_ICON = "💰"
DEFAULT_CURRENCY = "JPY"
CURRENCY_SYMBOL = "¥"

# File paths
CONFIG_DIR = Path(__file__).parent
CATEGORIES_FILE = CONFIG_DIR / "categories.json"

# Load categories from JSON
with open(CATEGORIES_FILE) as f:
    DEFAULT_CATEGORIES = json.load(f)

# Budget limits per category (monthly)
BUDGET_LIMITS = {
    "Food & Dining": 80000,
    "Groceries": 50000,
    "Transportation": 30000,
    "Shopping": 40000,
    "Entertainment": 30000,
    "Health & Medical": 20000,
    "Utilities": 35000,
    "Housing": 150000,
    "Education": 25000,
    "Travel": 50000,
    "Mila": 30000,
    "Other": 30000
}

# Payment methods
PAYMENT_METHODS = [
    "Credit Card",
    "Debit Card", 
    "Cash",
    "Check",
    "Digital Wallet",
    "Bank Transfer"
]

# Streamlit page config
STREAMLIT_CONFIG = {
    "page_title": APP_TITLE,
    "page_icon": APP_ICON,
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Cache settings
CACHE_TIMEOUT = 300  # 5 minutes

# Chart settings
CHART_COLORS = {
    "primary": "#4ECDC4",
    "secondary": "#556270",
    "accent": "#C7F464",
    "warning": "#FF6B6B"
}

# Alert thresholds (percentage of budget)
BUDGET_ALERT_THRESHOLDS = {
    "warning": 80,  # Show warning when 80% of budget is used
    "danger": 95    # Show danger alert when 95% of budget is used
}
