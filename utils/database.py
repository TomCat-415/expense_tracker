"""
Database utilities for expense management.
"""
import sqlite3
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, date
import hashlib
from functools import lru_cache
import logging
from config.settings import DATABASE_FILE, CACHE_TIMEOUT, DATA_DIR
import shutil

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass

class ValidationError(Exception):
    """Custom exception for data validation."""
    pass

def get_db_connection():
    """Get a database connection with row factory."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise DatabaseError("Could not connect to database")

def init_db():
    """Initialize the database with required tables."""
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)
    
    conn = get_db_connection()
    try:
        c = conn.cursor()
        
        # Check if expenses table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='expenses'")
        table_exists = c.fetchone() is not None
        
        if table_exists:
            # Check if hash column exists
            c.execute("PRAGMA table_info(expenses)")
            columns = [col[1] for col in c.fetchall()]
            
            if 'hash' not in columns:
                # Create a backup before modifying
                backup_path = DATABASE_FILE.parent / f"expenses_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                shutil.copy2(DATABASE_FILE, backup_path)
                
                # Add hash column
                c.execute("ALTER TABLE expenses ADD COLUMN hash TEXT")
                
                # Update existing rows with hash values
                c.execute("SELECT id, date, merchant, amount FROM expenses")
                for row in c.fetchall():
                    expense_data = {
                        'date': row[1],
                        'merchant': row[2],
                        'amount': row[3]
                    }
                    expense_hash = calculate_expense_hash(expense_data)
                    c.execute("UPDATE expenses SET hash = ? WHERE id = ?", (expense_hash, row[0]))
                
                # Now add the UNIQUE constraint
                c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_expenses_hash ON expenses(hash)")
        else:
            # Create expenses table
            c.execute('''
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                merchant TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT NOT NULL,
                description TEXT,
                payment_method TEXT,
                receipt_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hash TEXT UNIQUE
            )
            ''')
        
        # Create backup_history table
        c.execute('''
        CREATE TABLE IF NOT EXISTS backup_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            record_count INTEGER NOT NULL,
            status TEXT NOT NULL
        )
        ''')
        
        conn.commit()
    finally:
        conn.close()

def calculate_expense_hash(expense_data: Dict) -> str:
    """Calculate a unique hash for an expense to detect duplicates."""
    # Create a string combining key fields
    # Convert date to string if it's a date object
    date_str = expense_data['date'].strftime('%Y-%m-%d') if isinstance(expense_data['date'], (datetime, date)) else expense_data['date']
    hash_string = f"{date_str}{expense_data['merchant']}{expense_data['amount']}"
    # Create SHA-256 hash
    return hashlib.sha256(hash_string.encode()).hexdigest()

def is_duplicate_expense(expense_data: Dict) -> bool:
    """Check if an expense is a potential duplicate."""
    conn = get_db_connection()
    try:
        c = conn.cursor()
        expense_hash = calculate_expense_hash(expense_data)
        
        # Check for exact hash match
        c.execute('SELECT id FROM expenses WHERE hash = ?', (expense_hash,))
        if c.fetchone():
            return True
        
        # Convert date to string if it's a date object
        if isinstance(expense_data['date'], (datetime, date)):
            date_str = expense_data['date'].strftime('%Y-%m-%d')
        else:
            # If it's already a string, try to parse it to validate format
            try:
                date_obj = datetime.strptime(expense_data['date'], '%Y-%m-%d')
                date_str = expense_data['date']
            except ValueError:
                raise ValidationError("Invalid date format. Expected YYYY-MM-DD")
        
        # Check for similar expenses within 24 hours
        c.execute('''
        SELECT id FROM expenses 
        WHERE date = ? 
        AND merchant = ? 
        AND ABS(amount - ?) < 0.01
        ''', (
            date_str,
            expense_data['merchant'],
            expense_data['amount']
        ))
        
        return bool(c.fetchone())
    finally:
        conn.close()

@lru_cache(maxsize=None)
def get_expenses_df(cache_key: Optional[str] = None) -> pd.DataFrame:
    """Get all expenses as a pandas DataFrame with caching."""
    conn = get_db_connection()
    try:
        query = '''
        SELECT 
            id,
            date,
            merchant,
            amount,
            category,
            description,
            payment_method,
            receipt_path,
            created_at,
            updated_at
        FROM expenses
        ORDER BY date DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        df['date'] = pd.to_datetime(df['date'])
        return df
    finally:
        conn.close()

def invalidate_expenses_cache():
    """Clear the expenses DataFrame cache."""
    get_expenses_df.cache_clear()

def add_expense(expense_data: Dict) -> int:
    """Add a new expense to the database with duplicate detection."""
    if is_duplicate_expense(expense_data):
        raise DatabaseError("This appears to be a duplicate expense")
    
    conn = get_db_connection()
    try:
        c = conn.cursor()
        expense_hash = calculate_expense_hash(expense_data)
        
        c.execute('''
        INSERT INTO expenses (
            date, merchant, amount, category, description,
            payment_method, receipt_path, hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            expense_data['date'],
            expense_data['merchant'],
            expense_data['amount'],
            expense_data['category'],
            expense_data.get('description', ''),
            expense_data.get('payment_method', ''),
            expense_data.get('receipt_path', ''),
            expense_hash
        ))
        
        conn.commit()
        # Invalidate cache after successful insert
        invalidate_expenses_cache()
        return c.lastrowid
    except sqlite3.Error as e:
        raise DatabaseError(f"Error adding expense: {str(e)}")
    finally:
        conn.close()

def update_expense(expense_id: int, expense_data: Dict) -> None:
    """Update an existing expense."""
    conn = get_db_connection()
    try:
        c = conn.cursor()
        expense_hash = calculate_expense_hash(expense_data)
        
        c.execute('''
        UPDATE expenses SET
            date = ?,
            merchant = ?,
            amount = ?,
            category = ?,
            description = ?,
            payment_method = ?,
            receipt_path = ?,
            hash = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        ''', (
            expense_data['date'],
            expense_data['merchant'],
            expense_data['amount'],
            expense_data['category'],
            expense_data.get('description', ''),
            expense_data.get('payment_method', ''),
            expense_data.get('receipt_path', ''),
            expense_hash,
            expense_id
        ))
        
        if c.rowcount == 0:
            raise DatabaseError(f"No expense found with ID {expense_id}")
        
        conn.commit()
        # Invalidate cache after successful update
        invalidate_expenses_cache()
    except sqlite3.Error as e:
        raise DatabaseError(f"Error updating expense: {str(e)}")
    finally:
        conn.close()

def delete_expense(expense_id: int) -> None:
    """Delete an expense from the database."""
    conn = get_db_connection()
    try:
        c = conn.cursor()
        
        # Get receipt path before deletion
        c.execute('SELECT receipt_path FROM expenses WHERE id = ?', (expense_id,))
        result = c.fetchone()
        if not result:
            raise DatabaseError(f"No expense found with ID {expense_id}")
        
        receipt_path = result[0]
        
        # Delete the expense
        c.execute('DELETE FROM expenses WHERE id = ?', (expense_id,))
        conn.commit()
        
        # Delete associated receipt file if it exists
        if receipt_path:
            receipt_file = Path(receipt_path)
            if receipt_file.exists():
                receipt_file.unlink()
        
        # Invalidate cache after successful deletion
        invalidate_expenses_cache()
                
    except sqlite3.Error as e:
        raise DatabaseError(f"Error deleting expense: {str(e)}")
    finally:
        conn.close()

def import_from_csv(file_path: Path) -> Tuple[int, int, List[str]]:
    """Import expenses from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        required_columns = ['date', 'merchant', 'amount', 'category']
        
        # Validate columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Convert date strings to datetime
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Track import results
        total_records = len(df)
        imported_count = 0
        errors = []
        
        # Import each record
        for _, row in df.iterrows():
            try:
                expense_data = row.to_dict()
                if not is_duplicate_expense(expense_data):
                    add_expense(expense_data)
                    imported_count += 1
                else:
                    errors.append(f"Duplicate expense: {row['date']} - {row['merchant']} - {row['amount']}")
            except Exception as e:
                errors.append(f"Error importing row: {str(e)}")
        
        return total_records, imported_count, errors
    
    except Exception as e:
        raise DatabaseError(f"Error importing CSV: {str(e)}")

def export_to_csv(file_path: Path, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> int:
    """Export expenses to a CSV file with optional date range."""
    df = get_expenses_df()
    
    if start_date:
        df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] <= end_date]
    
    # Drop internal columns
    df = df.drop(['id', 'created_at', 'updated_at'], axis=1)
    
    # Export to CSV
    df.to_csv(file_path, index=False)
    return len(df)

def create_backup(backup_dir: Path) -> str:
    """Create a backup of the entire database."""
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = backup_dir / f"expenses_backup_{timestamp}.json"
    
    conn = get_db_connection()
    try:
        # Get all data
        df = pd.read_sql_query('SELECT * FROM expenses', conn)
        
        # Convert to JSON
        data = {
            'expenses': df.to_dict(orient='records'),
            'metadata': {
                'created_at': timestamp,
                'record_count': len(df),
                'version': '1.0'
            }
        }
        
        # Save backup
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Record backup in history
        c = conn.cursor()
        c.execute('''
        INSERT INTO backup_history (filename, record_count, status)
        VALUES (?, ?, ?)
        ''', (str(backup_file), len(df), 'success'))
        conn.commit()
        
        return str(backup_file)
    
    except Exception as e:
        if backup_file.exists():
            backup_file.unlink()
        raise DatabaseError(f"Error creating backup: {str(e)}")
    finally:
        conn.close()

def restore_from_backup(backup_file: Path) -> Tuple[int, int]:
    """Restore database from a backup file."""
    try:
        # Read backup file
        with open(backup_file) as f:
            data = json.load(f)
        
        if 'expenses' not in data:
            raise ValueError("Invalid backup file format")
        
        # Clear existing data
        conn = get_db_connection()
        try:
            c = conn.cursor()
            c.execute('DELETE FROM expenses')
            
            # Insert backup data
            total_records = len(data['expenses'])
            restored_count = 0
            
            for expense in data['expenses']:
                try:
                    c.execute('''
                    INSERT INTO expenses (
                        date, merchant, amount, category, description,
                        payment_method, receipt_path, hash, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        expense['date'],
                        expense['merchant'],
                        expense['amount'],
                        expense['category'],
                        expense.get('description', ''),
                        expense.get('payment_method', ''),
                        expense.get('receipt_path', ''),
                        expense.get('hash', calculate_expense_hash(expense)),
                        expense.get('created_at', datetime.now().isoformat()),
                        expense.get('updated_at', datetime.now().isoformat())
                    ))
                    restored_count += 1
                except sqlite3.Error as e:
                    logger.error(f"Error restoring expense: {str(e)}")
            
            conn.commit()
            return total_records, restored_count
            
        finally:
            conn.close()
            
    except Exception as e:
        raise DatabaseError(f"Error restoring from backup: {str(e)}")

# Initialize database on module import
init_db()
    