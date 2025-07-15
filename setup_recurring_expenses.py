#!/usr/bin/env python3
"""
Setup script for recurring expenses table in Supabase.
Run this once to create the recurring_expenses table and related components.
"""

import os
from pathlib import Path
from utils.supabase_client import supabase

def setup_recurring_expenses_table():
    """Create the recurring_expenses table and related components."""
    
    # Read the SQL script
    sql_file = Path(__file__).parent / "create_recurring_expenses_table.sql"
    
    if not sql_file.exists():
        print("❌ SQL file not found. Please create create_recurring_expenses_table.sql first.")
        return False
    
    with open(sql_file, 'r') as f:
        sql_content = f.read()
    
    # Split the SQL content into individual statements
    statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
    
    print("🚀 Setting up recurring_expenses table...")
    
    success_count = 0
    total_count = len(statements)
    
    for i, statement in enumerate(statements, 1):
        if not statement:
            continue
            
        try:
            print(f"📝 Executing statement {i}/{total_count}...")
            
            # Execute the statement
            result = supabase.rpc('exec_sql', {'sql': statement}).execute()
            
            if result.data:
                print(f"✅ Statement {i} executed successfully")
                success_count += 1
            else:
                print(f"⚠️ Statement {i} completed with warnings")
                success_count += 1
                
        except Exception as e:
            print(f"❌ Error executing statement {i}: {str(e)}")
            print(f"Statement was: {statement[:100]}...")
            
            # For some statements, we can continue even if they fail
            if any(phrase in statement.lower() for phrase in ['if not exists', 'if exists', 'create policy']):
                print("ℹ️ This error might be expected (table/policy already exists)")
                success_count += 1
                continue
            else:
                print("💥 Critical error, stopping setup")
                return False
    
    print(f"\n🎉 Setup completed! {success_count}/{total_count} statements executed successfully")
    
    # Test the table by trying to select from it
    try:
        test_result = supabase.table("recurring_expenses").select("count", count="exact").execute()
        print(f"✅ Table is accessible and has {test_result.count} rows")
        return True
    except Exception as e:
        print(f"❌ Error testing table access: {str(e)}")
        return False

if __name__ == "__main__":
    print("🛠️ Recurring Expenses Table Setup")
    print("=" * 40)
    
    success = setup_recurring_expenses_table()
    
    if success:
        print("\n🎉 Setup completed successfully!")
        print("You can now use the recurring expenses feature in your app.")
    else:
        print("\n❌ Setup failed!")
        print("Please check the errors above and try again.")
        print("You may need to run the SQL statements manually in your Supabase dashboard.") 