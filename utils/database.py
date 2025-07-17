"""
Database utilities for expense management (Supabase version).
"""
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date
import hashlib
import logging
from functools import lru_cache
from utils.supabase_client import supabase

logger = logging.getLogger(__name__)

# Load default settings
with open(Path(__file__).parent.parent / 'config' / 'categories.json', 'r') as f:
    DEFAULT_CATEGORIES = json.load(f)

# Default budget limits per category (monthly) - Full Lifestyle Mode
BUDGET_LIMITS = {
    "🏠 Rent & Housing": 150000,
    "🍽 Eats & Treats": 80000,
    "☕ Coffee": 15000,
    "🍺 Alcohol": 20000,
    "🛍 Retail Therapy": 40000,
    "💡 Bills & Utilities": 35000,
    "💻 Tech & Gear": 25000,
    "🧠 Education": 25000,
    "💅 Self-Care": 20000,
    "👶 Kids & Baby": 30000,
    "🚗 Transport": 30000,
    "🌱 Wellness": 20000,
    "🧾 Medical & Care": 20000,
    "🎭 Fun & Entertainment": 30000,
    "🎁 Gifts": 15000,
    "🌀 Misc & One-Offs": 30000
}

class DatabaseError(Exception):
    pass

class ValidationError(Exception):
    pass

def calculate_expense_hash(expense_data: Dict) -> str:
    date_str = expense_data['date'].strftime('%Y-%m-%d') if isinstance(expense_data['date'], (datetime, date)) else expense_data['date']
    hash_string = f"{date_str}{expense_data['merchant']}{expense_data['amount']}"
    return hashlib.sha256(hash_string.encode()).hexdigest()

def is_duplicate_expense(user_id: str, expense_data: Dict) -> bool:
    expense_hash = calculate_expense_hash(expense_data)
    try:
        result = supabase.table("expenses").select("id").eq("hash", expense_hash).eq("user_id", user_id).execute()
        return len(result.data) > 0
    except Exception as e:
        raise DatabaseError(f"Error checking for duplicates: {str(e)}")

@lru_cache(maxsize=None)
def get_expenses_df(user_id: str) -> pd.DataFrame:
    try:
        logger.info(f"Fetching expenses for user_id: {user_id}")
        result = supabase.table("expenses").select("*").eq("user_id", user_id).order("date", desc=True).execute()
        logger.info(f"Got response from Supabase: {result.data if hasattr(result, 'data') else 'No data'}")
        
        if not result or not hasattr(result, 'data'):
            logger.warning("No result or data attribute from Supabase")
            return pd.DataFrame()
            
        df = pd.DataFrame(result.data)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        logger.error(f"Error fetching expenses: {str(e)}")
        if hasattr(e, 'message'):
            logger.error(f"Error message: {e.message}")
        raise DatabaseError(f"Error fetching expenses: {str(e)}")

def invalidate_expenses_cache():
    get_expenses_df.cache_clear()

def add_expense(user_id: str, expense_data: Dict) -> str:
    try:
        logger.info(f"Starting add_expense with user_id: {user_id}")
        logger.info(f"Original expense_data: {expense_data}")
        
        # Create a new dict with only the fields we want to insert
        insert_data = {
            "date": expense_data['date'].isoformat() if isinstance(expense_data['date'], (datetime, date)) else expense_data['date'],
            "merchant": expense_data['merchant'].strip(),
            "amount": expense_data['amount'],
            "category": expense_data['category'],
            "description": expense_data.get('description', '').strip() if expense_data.get('description') else None,
            "payment_method": expense_data['payment_method'],
            "receipt_path": expense_data.get('receipt_path'),
            "user_id": user_id
        }
        
        # Calculate hash after normalizing the data
        insert_data['hash'] = calculate_expense_hash(insert_data)
        
        logger.info(f"Prepared expense data for Supabase: {insert_data}")
        
        try:
            result = supabase.table("expenses").insert(insert_data).execute()
            logger.info(f"Supabase insert response: {result.data if hasattr(result, 'data') else result}")
            
            if not result or not hasattr(result, 'data') or not result.data:
                raise DatabaseError("No data returned from Supabase insert")
                
            invalidate_expenses_cache()
            return result.data[0]['id'] if result.data else None
            
        except Exception as e:
            logger.error(f"Supabase insert error: {str(e)}")
            if hasattr(e, 'message'):
                logger.error(f"Error message: {e.message}")
            raise DatabaseError(f"Failed to insert expense: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in add_expense: {str(e)}")
        logger.error(f"Full expense data was: {expense_data}")
        raise DatabaseError(f"Error adding expense: {str(e)}")

def update_expense(user_id: str, expense_id: str, expense_data: Dict) -> None:
    try:
        # Create a new dict with only the fields we want to update
        update_data = {
            "date": expense_data['date'].isoformat() if isinstance(expense_data['date'], (datetime, date)) else expense_data['date'],
            "merchant": expense_data['merchant'].strip(),
            "amount": expense_data['amount'],
            "category": expense_data['category'],
            "description": expense_data.get('description', '').strip() if expense_data.get('description') else None,
            "payment_method": expense_data['payment_method'],
            "receipt_path": expense_data.get('receipt_path'),
            "user_id": user_id
        }
        
        # Calculate hash after normalizing the data
        update_data['hash'] = calculate_expense_hash(update_data)
        
        try:
            result = supabase.table("expenses").update(update_data).eq("id", expense_id).eq("user_id", user_id).execute()
            if not result.data:
                raise DatabaseError(f"No expense found with ID {expense_id}")
            invalidate_expenses_cache()
        except Exception as e:
            logger.error(f"Supabase update error: {str(e)}")
            raise DatabaseError(f"Failed to update expense: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in update_expense: {str(e)}")
        raise DatabaseError(f"Error updating expense: {str(e)}")

def delete_expense(user_id: str, expense_id: str) -> None:
    try:
        # Get expense details including recurring_id before deleting
        expense_result = supabase.table("expenses").select("receipt_path", "recurring_id").eq("id", expense_id).eq("user_id", user_id).single().execute()
        
        if not expense_result.data:
            raise DatabaseError(f"Expense with ID {expense_id} not found")
        
        expense_data = expense_result.data
        receipt_path = expense_data.get("receipt_path")
        recurring_id = expense_data.get("recurring_id")

        # Delete the expense
        supabase.table("expenses").delete().eq("id", expense_id).eq("user_id", user_id).execute()

        # If this expense was generated from a recurring expense, ask user if they want to delete the recurring expense too
        if recurring_id and recurring_id not in [None, '', 'null', 'None']:
            # Check if this was the only expense generated from this recurring expense
            remaining_expenses = supabase.table("expenses").select("id").eq("recurring_id", recurring_id).eq("user_id", user_id).execute()
            
            if not remaining_expenses.data:
                # No other expenses from this recurring expense, delete the recurring expense
                supabase.table("recurring_expenses").delete().eq("id", recurring_id).eq("user_id", user_id).execute()
                logger.info(f"Deleted recurring expense {recurring_id} as it had no remaining generated expenses")

        # Clean up receipt file if it exists
        if receipt_path:
            receipt_file = Path(receipt_path)
            if receipt_file.exists():
                receipt_file.unlink()

        invalidate_expenses_cache()
    except Exception as e:
        raise DatabaseError(f"Error deleting expense: {str(e)}")

def import_from_csv(user_id: str, file_path: Path) -> Tuple[int, int, List[str]]:
    try:
        df = pd.read_csv(file_path)
        required_columns = ['date', 'merchant', 'amount', 'category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        df['date'] = pd.to_datetime(df['date']).dt.date
        total_records = len(df)
        imported_count = 0
        errors = []

        for _, row in df.iterrows():
            try:
                expense_data = row.to_dict()
                if not is_duplicate_expense(user_id, expense_data):
                    add_expense(user_id, expense_data)
                    imported_count += 1
                else:
                    errors.append(f"Duplicate expense: {row['date']} - {row['merchant']} - {row['amount']}")
            except Exception as e:
                errors.append(f"Error importing row: {str(e)}")

        return total_records, imported_count, errors

    except Exception as e:
        raise DatabaseError(f"Error importing CSV: {str(e)}")

def export_to_csv(user_id: str, file_path: Path, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> int:
    df = get_expenses_df(user_id)
    if start_date:
        df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] <= end_date]

    df = df.drop(columns=['id', 'user_id'], errors='ignore')
    df.to_csv(file_path, index=False)
    return len(df)

def create_backup(user_id: str, backup_dir: Path) -> str:
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = backup_dir / f"expenses_backup_{timestamp}.json"

    try:
        df = get_expenses_df(user_id)
        data = {
            'expenses': df.to_dict(orient='records'),
            'metadata': {
                'created_at': timestamp,
                'record_count': len(df),
                'version': '2.0-supabase',
                'user_id': user_id
            }
        }
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return str(backup_file)
    except Exception as e:
        if backup_file.exists():
            backup_file.unlink()
        raise DatabaseError(f"Error creating backup: {str(e)}")

def restore_from_backup(user_id: str, backup_file: Path) -> Tuple[int, int]:
    try:
        with open(backup_file) as f:
            data = json.load(f)

        if 'expenses' not in data:
            raise ValueError("Invalid backup file format")

        # Delete only the user's expenses
        supabase.table("expenses").delete().eq("user_id", user_id).execute()

        total_records = len(data['expenses'])
        restored_count = 0

        for expense in data['expenses']:
            try:
                expense['hash'] = calculate_expense_hash(expense)
                expense['user_id'] = user_id  # Ensure user_id is set
                supabase.table("expenses").insert(expense).execute()
                restored_count += 1
            except Exception as e:
                logger.error(f"Error restoring expense: {str(e)}")

        return total_records, restored_count
    except Exception as e:
        raise DatabaseError(f"Error restoring from backup: {str(e)}")

# User Settings Functions
def get_user_settings(user_id: str) -> Dict:
    """Get user's custom settings including budgets."""
    try:
        result = supabase.table("user_settings").select("*").eq("user_id", user_id).execute()
        if result.data:
            return result.data[0]
        # Return default settings if none exist
        return {
            "monthly_budget": sum(BUDGET_LIMITS.values()),
            "budget_by_category": BUDGET_LIMITS,
            "custom_categories": DEFAULT_CATEGORIES
        }
    except Exception as e:
        logger.error(f"Error getting user settings: {e}")
        raise DatabaseError(f"Failed to get user settings: {e}")

def update_user_settings(user_id: str, settings: Dict) -> None:
    """Update user's custom settings."""
    try:
        # Check if settings exist
        result = supabase.table("user_settings").select("*").eq("user_id", user_id).execute()
        
        if result.data:
            # Update existing settings
            supabase.table("user_settings").update(settings).eq("user_id", user_id).execute()
        else:
            # Create new settings
            settings["user_id"] = user_id
            supabase.table("user_settings").insert(settings).execute()
    except Exception as e:
        logger.error(f"Error updating user settings: {e}")
        raise DatabaseError(f"Failed to update user settings: {e}")

def update_category_budget(user_id: str, category: str, budget: float) -> None:
    """Update budget for a specific category."""
    try:
        settings = get_user_settings(user_id)
        budgets = settings.get("budget_by_category", {})
        budgets[category] = budget
        settings["budget_by_category"] = budgets
        update_user_settings(user_id, settings)
    except Exception as e:
        logger.error(f"Error updating category budget: {e}")
        raise DatabaseError(f"Failed to update category budget: {e}")

def add_custom_category(user_id: str, category_name: str, color: str) -> None:
    """Add a new custom category."""
    try:
        settings = get_user_settings(user_id)
        categories = settings.get("custom_categories", {})
        if category_name in categories:
            raise ValidationError(f"Category '{category_name}' already exists")
        categories[category_name] = color
        settings["custom_categories"] = categories
        update_user_settings(user_id, settings)
    except ValidationError as e:
        raise e
    except Exception as e:
        logger.error(f"Error adding custom category: {e}")
        raise DatabaseError(f"Failed to add custom category: {e}")

def delete_custom_category(user_id: str, category_name: str) -> None:
    """Delete a custom category."""
    try:
        settings = get_user_settings(user_id)
        categories = settings.get("custom_categories", {})
        if category_name not in categories:
            raise ValidationError(f"Category '{category_name}' does not exist")
        del categories[category_name]
        settings["custom_categories"] = categories
        update_user_settings(user_id, settings)
    except ValidationError as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting custom category: {e}")
        raise DatabaseError(f"Failed to delete custom category: {e}")

def sync_default_categories_with_user_settings(user_id: str) -> None:
    """Sync default categories from categories.json with user settings."""
    try:
        settings = get_user_settings(user_id)
        
        # Get current custom categories or empty dict
        current_custom = settings.get("custom_categories", {})
        
        # If user has no custom categories, or if they want to sync with defaults
        if not current_custom or current_custom == {}:
            # Copy default categories to user settings
            settings["custom_categories"] = DEFAULT_CATEGORIES.copy()
            update_user_settings(user_id, settings)
            logger.info(f"Synced default categories for user {user_id}")
            
    except Exception as e:
        logger.error(f"Error syncing default categories: {e}")

def get_available_categories(user_id: str = None) -> List[str]:
    """Get list of available expense categories including custom ones."""
    if user_id is None:
        return [cat for cat in DEFAULT_CATEGORIES.keys() if cat != "_metadata"]
    
    try:
        settings = get_user_settings(user_id)
        categories = settings.get("custom_categories", {})
        
        # If user has no custom categories, sync with defaults
        if not categories:
            sync_default_categories_with_user_settings(user_id)
            settings = get_user_settings(user_id)  # Refresh settings
            categories = settings.get("custom_categories", DEFAULT_CATEGORIES)
            
        return [cat for cat in categories.keys() if cat != "_metadata"]
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        return [cat for cat in DEFAULT_CATEGORIES.keys() if cat != "_metadata"]


# =============================================
# RECURRING EXPENSES FUNCTIONS
# =============================================

def add_recurring_expense(user_id: str, recurring_data: Dict) -> str:
    """Add a new recurring expense."""
    try:
        from datetime import datetime, timedelta
        
        # Calculate next due date based on frequency
        start_date = recurring_data['start_date']
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        
        next_due = calculate_next_due_date(start_date, recurring_data['frequency'])
        
        # Default to averaging for quarterly and yearly expenses
        frequency = recurring_data['frequency']
        if frequency in ['quarterly', 'yearly']:
            default_averaging = 'monthly'
        else:
            default_averaging = 'none'
        
        insert_data = {
            "user_id": user_id,
            "name": recurring_data['name'].strip(),
            "merchant": recurring_data['merchant'].strip(),
            "amount": recurring_data['amount'],
            "category": recurring_data['category'],
            "description": recurring_data.get('description', '').strip() if recurring_data.get('description') else None,
            "payment_method": recurring_data['payment_method'],
            "frequency": frequency,
            "start_date": start_date.isoformat(),
            "end_date": recurring_data.get('end_date').isoformat() if recurring_data.get('end_date') else None,
            "next_due_date": next_due.isoformat(),
            "last_generated_date": None,
            "is_active": True,
            "averaging_type": recurring_data.get('averaging_type', default_averaging),
            "created_at": datetime.now().isoformat()
        }
        
        result = supabase.table("recurring_expenses").insert(insert_data).execute()
        
        if not result or not hasattr(result, 'data') or not result.data:
            raise DatabaseError("No data returned from Supabase insert")
            
        return result.data[0]['id'] if result.data else None
        
    except Exception as e:
        logger.error(f"Error adding recurring expense: {str(e)}")
        raise DatabaseError(f"Error adding recurring expense: {str(e)}")


def get_recurring_expenses(user_id: str) -> List[Dict]:
    """Get all recurring expenses for a user."""
    try:
        result = supabase.table("recurring_expenses").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        
        if not result or not hasattr(result, 'data'):
            return []
            
        return result.data
        
    except Exception as e:
        logger.error(f"Error fetching recurring expenses: {str(e)}")
        raise DatabaseError(f"Error fetching recurring expenses: {str(e)}")


def update_recurring_expense(user_id: str, recurring_id: str, recurring_data: Dict) -> None:
    """Update an existing recurring expense."""
    try:
        from datetime import datetime
        
        # Recalculate next due date if frequency or start date changed
        if 'frequency' in recurring_data or 'start_date' in recurring_data:
            start_date = recurring_data.get('start_date')
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            
            if start_date:
                next_due = calculate_next_due_date(start_date, recurring_data.get('frequency', 'monthly'))
                recurring_data['next_due_date'] = next_due.isoformat()
        
        # Prepare update data
        update_data = {
            "name": recurring_data['name'].strip(),
            "merchant": recurring_data['merchant'].strip(),
            "amount": recurring_data['amount'],
            "category": recurring_data['category'],
            "description": recurring_data.get('description', '').strip() if recurring_data.get('description') else None,
            "payment_method": recurring_data['payment_method'],
            "frequency": recurring_data['frequency'],
            "start_date": recurring_data['start_date'].isoformat() if hasattr(recurring_data['start_date'], 'isoformat') else recurring_data['start_date'],
            "end_date": recurring_data.get('end_date').isoformat() if recurring_data.get('end_date') and hasattr(recurring_data['end_date'], 'isoformat') else recurring_data.get('end_date'),
            "is_active": recurring_data.get('is_active', True),
            "averaging_type": recurring_data.get('averaging_type', 'none'),
            "updated_at": datetime.now().isoformat()
        }
        
        if 'next_due_date' in recurring_data:
            update_data['next_due_date'] = recurring_data['next_due_date']
        
        result = supabase.table("recurring_expenses").update(update_data).eq("id", recurring_id).eq("user_id", user_id).execute()
        
        if not result.data:
            raise DatabaseError(f"No recurring expense found with ID {recurring_id}")
            
    except Exception as e:
        logger.error(f"Error updating recurring expense: {str(e)}")
        raise DatabaseError(f"Error updating recurring expense: {str(e)}")


def delete_recurring_expense(user_id: str, recurring_id: str) -> None:
    """Delete a recurring expense."""
    try:
        result = supabase.table("recurring_expenses").delete().eq("id", recurring_id).eq("user_id", user_id).execute()
        
        if not result.data:
            raise DatabaseError(f"No recurring expense found with ID {recurring_id}")
            
    except Exception as e:
        logger.error(f"Error deleting recurring expense: {str(e)}")
        raise DatabaseError(f"Error deleting recurring expense: {str(e)}")


def calculate_next_due_date(start_date, frequency: str):
    """Calculate the next due date based on frequency."""
    from datetime import datetime, timedelta
    import calendar
    
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    
    today = datetime.now().date()
    
    if frequency == 'weekly':
        # Find next week from today
        days_ahead = 7 - (today.weekday() - start_date.weekday()) % 7
        if days_ahead == 0:  # If today is the day
            days_ahead = 7
        return today + timedelta(days=days_ahead)
    
    elif frequency == 'monthly':
        # Find next month with same day
        next_month = today.replace(day=1)
        if next_month.month == 12:
            next_month = next_month.replace(year=next_month.year + 1, month=1)
        else:
            next_month = next_month.replace(month=next_month.month + 1)
        
        # Handle end of month dates
        try:
            return next_month.replace(day=start_date.day)
        except ValueError:
            # If day doesn't exist in next month, use last day of month
            last_day = calendar.monthrange(next_month.year, next_month.month)[1]
            return next_month.replace(day=last_day)
    
    elif frequency == 'quarterly':
        # Find next quarter (3 months ahead)
        months_ahead = 3
        new_month = today.month + months_ahead
        new_year = today.year
        
        while new_month > 12:
            new_month -= 12
            new_year += 1
            
        next_quarter = today.replace(year=new_year, month=new_month, day=1)
        
        try:
            return next_quarter.replace(day=start_date.day)
        except ValueError:
            last_day = calendar.monthrange(next_quarter.year, next_quarter.month)[1]
            return next_quarter.replace(day=last_day)
    
    elif frequency == 'yearly':
        # Find next year with same month and day
        next_year = today.replace(year=today.year + 1)
        try:
            return next_year.replace(month=start_date.month, day=start_date.day)
        except ValueError:
            # Handle leap year edge case
            return next_year.replace(month=start_date.month, day=28)
    
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")


def get_due_recurring_expenses(user_id: str) -> List[Dict]:
    """Get recurring expenses that are due for generation."""
    try:
        from datetime import datetime
        
        today = datetime.now().date().isoformat()
        
        result = supabase.table("recurring_expenses").select("*").eq("user_id", user_id).eq("is_active", True).lte("next_due_date", today).execute()
        
        if not result or not hasattr(result, 'data'):
            return []
            
        return result.data
        
    except Exception as e:
        logger.error(f"Error fetching due recurring expenses: {str(e)}")
        raise DatabaseError(f"Error fetching due recurring expenses: {str(e)}")


def generate_expense_from_recurring(user_id: str, recurring_id: str, override_data: Dict = None) -> str:
    """Generate an actual expense from a recurring expense."""
    try:
        from datetime import datetime
        
        # Get the recurring expense
        recurring_result = supabase.table("recurring_expenses").select("*").eq("id", recurring_id).eq("user_id", user_id).single().execute()
        
        if not recurring_result.data:
            raise DatabaseError(f"No recurring expense found with ID {recurring_id}")
            
        recurring = recurring_result.data
        
        # Create expense data
        expense_data = {
            "date": datetime.now().date(),
            "merchant": override_data.get('merchant', recurring['merchant']) if override_data else recurring['merchant'],
            "amount": override_data.get('amount', recurring['amount']) if override_data else recurring['amount'],
            "category": override_data.get('category', recurring['category']) if override_data else recurring['category'],
            "description": override_data.get('description', recurring['description']) if override_data else recurring['description'],
            "payment_method": override_data.get('payment_method', recurring['payment_method']) if override_data else recurring['payment_method'],
            "recurring_id": recurring_id
        }
        
        # Add the expense
        expense_id = add_expense(user_id, expense_data)
        
        # Update recurring expense's next due date and last generated date
        next_due = calculate_next_due_date(recurring['start_date'], recurring['frequency'])
        
        update_data = {
            "next_due_date": next_due.isoformat(),
            "last_generated_date": datetime.now().date().isoformat()
        }
        
        supabase.table("recurring_expenses").update(update_data).eq("id", recurring_id).eq("user_id", user_id).execute()
        
        return expense_id
        
    except Exception as e:
        logger.error(f"Error generating expense from recurring: {str(e)}")
        raise DatabaseError(f"Error generating expense from recurring: {str(e)}")


def suggest_recurring_expenses(user_id: str) -> List[Dict]:
    """Suggest recurring expenses based on expense patterns."""
    try:
        from datetime import datetime, timedelta
        
        # Get expenses from last 6 months
        six_months_ago = datetime.now() - timedelta(days=180)
        df = get_expenses_df(user_id)
        
        if df.empty:
            return []
            
        # Filter to last 6 months
        recent_df = df[df['date'] >= six_months_ago]
        
        # Group by merchant and amount to find patterns
        patterns = recent_df.groupby(['merchant', 'amount']).agg({
            'date': ['count', 'min', 'max'],
            'category': 'first',
            'payment_method': 'first'
        }).reset_index()
        
        # Flatten column names
        patterns.columns = ['merchant', 'amount', 'count', 'first_date', 'last_date', 'category', 'payment_method']
        
        # Find potential recurring expenses (appeared 3+ times)
        suggestions = []
        for _, row in patterns.iterrows():
            if row['count'] >= 3:
                # Calculate days between first and last occurrence
                days_diff = (row['last_date'] - row['first_date']).days
                avg_days = days_diff / (row['count'] - 1) if row['count'] > 1 else 0
                
                # Suggest frequency based on average days
                if 25 <= avg_days <= 35:  # Monthly
                    frequency = 'monthly'
                elif 85 <= avg_days <= 95:  # Quarterly
                    frequency = 'quarterly'
                elif 360 <= avg_days <= 370:  # Yearly
                    frequency = 'yearly'
                elif 6 <= avg_days <= 8:  # Weekly
                    frequency = 'weekly'
                else:
                    continue  # Skip if doesn't match common patterns
                
                suggestions.append({
                    'merchant': row['merchant'],
                    'amount': row['amount'],
                    'category': row['category'],
                    'payment_method': row['payment_method'],
                    'frequency': frequency,
                    'occurrences': row['count'],
                    'confidence': min(row['count'] / 6, 1.0)  # Max confidence at 6 occurrences
                })
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return suggestions[:10]  # Return top 10 suggestions
        
    except Exception as e:
        logger.error(f"Error suggesting recurring expenses: {str(e)}")
        return []


# =============================================
# EXPENSE AVERAGING FUNCTIONS
# =============================================

def calculate_monthly_equivalent(amount: float, frequency: str) -> float:
    """Calculate the monthly equivalent of an expense based on frequency."""
    frequency_multipliers = {
        'monthly': 1.0,
        'quarterly': 1.0 / 3.0,  # Quarterly = every 3 months
        'yearly': 1.0 / 12.0     # Yearly = every 12 months
    }
    return amount * frequency_multipliers.get(frequency, 1.0)

def get_averaging_expenses(user_id: str) -> List[Dict]:
    """Get all expenses that should be averaged (from recurring expenses with averaging enabled)."""
    try:
        # Get recurring expenses with averaging enabled
        result = supabase.table("recurring_expenses").select("*").eq("user_id", user_id).eq("averaging_type", "monthly").execute()
        
        recurring_with_averaging = []
        for recurring in result.data:
            monthly_amount = calculate_monthly_equivalent(recurring['amount'], recurring['frequency'])
            recurring_with_averaging.append({
                'id': recurring['id'],
                'name': recurring['name'],
                'merchant': recurring['merchant'],
                'category': recurring['category'],
                'original_amount': recurring['amount'],
                'monthly_amount': monthly_amount,
                'frequency': recurring['frequency']
            })
        
        return recurring_with_averaging
        
    except Exception as e:
        logger.error(f"Error getting averaging expenses: {str(e)}")
        return []

def apply_expense_averaging(df: pd.DataFrame, user_id: str, include_averaging: bool = False) -> pd.DataFrame:
    """Apply averaging to expenses dataframe if requested."""
    if not include_averaging or df.empty:
        return df
    
    try:
        # Get expenses that should be averaged
        averaging_expenses = get_averaging_expenses(user_id)
        if not averaging_expenses:
            return df
        
        # Create a copy to avoid modifying the original
        averaged_df = df.copy()
        
        # Get date range for averaging
        min_date = pd.to_datetime(df['date'].min())
        max_date = pd.to_datetime(df['date'].max())
        
        # For each averaging expense, find actual payments and replace with monthly amounts
        for avg_expense in averaging_expenses:
            # Find expenses that match this recurring expense
            matching_expenses = averaged_df[
                (averaged_df['merchant'] == avg_expense['merchant']) &
                (averaged_df['category'] == avg_expense['category']) &
                (averaged_df['amount'] == avg_expense['original_amount'])
            ]
            
            if not matching_expenses.empty:
                # Remove the original large payments
                averaged_df = averaged_df.drop(matching_expenses.index)
                
                # Add monthly averaged payments for each month in the range
                current_date = min_date.replace(day=1)  # Start from first day of month
                
                while current_date <= max_date:
                    # Add averaged payment for this month
                    averaged_payment = {
                        'date': current_date,
                        'merchant': avg_expense['merchant'],
                        'category': avg_expense['category'],
                        'amount': avg_expense['monthly_amount'],
                        'description': f"[Averaged] {avg_expense['name']}",
                        'payment_method': matching_expenses.iloc[0]['payment_method'] if not matching_expenses.empty else 'Other',
                        'user_id': user_id,
                        'recurring_id': avg_expense['id']
                    }
                    
                    # Add to dataframe
                    new_row_df = pd.DataFrame([averaged_payment])
                    averaged_df = pd.concat([averaged_df, new_row_df], ignore_index=True)
                    
                    # Move to next month
                    if current_date.month == 12:
                        current_date = current_date.replace(year=current_date.year + 1, month=1)
                    else:
                        current_date = current_date.replace(month=current_date.month + 1)
        
        # Sort by date
        averaged_df = averaged_df.sort_values('date').reset_index(drop=True)
        
        return averaged_df
        
    except Exception as e:
        logger.error(f"Error applying expense averaging: {str(e)}")
        return df
        