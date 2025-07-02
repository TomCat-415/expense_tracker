"""
Expense Tracker Application
"""
import streamlit as st
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import time

from utils.database import (
    DatabaseError,
    ValidationError,
    add_expense,
    update_expense,
    delete_expense,
    get_expenses_df,
    import_from_csv,
    export_to_csv,
    create_backup,
    restore_from_backup
)
from utils.ocr import extract_text_from_image
from utils.charts import (
    create_daily_spending_chart,
    create_top_merchants_chart,
    create_monthly_average_chart,
    create_budget_tracking_chart,
    create_category_budget_allocation,
    create_weekly_comparison_chart,
    create_monthly_comparison_chart,
    create_spending_forecast
)
from config.settings import (
    DEFAULT_CATEGORIES,
    CURRENCY_SYMBOL,
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_IMAGE_SIZE_MB,
    RECEIPTS_DIR,
    PROCESSED_RECEIPTS_DIR,
    FAILED_RECEIPTS_DIR,
    BUDGET_LIMITS,
    BUDGET_ALERT_THRESHOLDS,
    STREAMLIT_CONFIG
)

# Configure Streamlit page
st.set_page_config(**STREAMLIT_CONFIG)

# Initialize session state variables
if "show_success" not in st.session_state:
    st.session_state.show_success = False
    st.session_state.success_message = ""

if "show_error" not in st.session_state:
    st.session_state.show_error = False
    st.session_state.error_message = ""

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Dashboard"

# Initialize pagination and search state
if "page_number" not in st.session_state:
    st.session_state.page_number = 1

if "expenses_per_page" not in st.session_state:
    st.session_state.expenses_per_page = 20

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# Helper functions for notifications
def show_success(message: str) -> None:
    """Show a success message."""
    st.session_state.show_success = True
    st.session_state.success_message = message

def show_error(message: str) -> None:
    """Show an error message."""
    st.session_state.show_error = True
    st.session_state.error_message = message

def change_page(page: int) -> None:
    """Change the current page number."""
    st.session_state.page_number = page

# Define tab layout and active mapping
tabs = st.tabs(["📊 Dashboard", "➕ Add Expense", "📸 Scan Receipt", "📋 All Expenses", "📊 Enhanced Analytics", "🗄️ Data Management"])
tab_mapping = {
    "📊 Dashboard": 0,
    "➕ Add Expense": 1,
    "📸 Scan Receipt": 2,
    "📋 All Expenses": 3,
    "📊 Enhanced Analytics": 4,
    "🗄️ Data Management": 5
}
active_index = tab_mapping.get(st.session_state.active_tab, 0)
tab1, tab2, tab3, tab4, tab5, tab6 = tabs

# Get all expenses for initial display
try:
    df = get_expenses_df()
except DatabaseError as e:
    show_error(f"Database error: {str(e)}")
    df = pd.DataFrame()

# ---------- Dashboard ----------
with tab1:
    st.header("📊 Dashboard")
    
    if df.empty:
        st.info("No expenses recorded yet. Add some expenses to see your dashboard!")
    else:
        # Calculate summary statistics
        total_expenses = df['amount'].sum()
        avg_daily = df.groupby(df['date'])['amount'].sum().mean()
        num_transactions = len(df)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Expenses", f"¥{total_expenses:,.0f}")
        with col2:
            st.metric("Average Daily", f"¥{avg_daily:,.0f}")
        with col3:
            st.metric("Transactions", num_transactions)
        
        # Display charts
        st.subheader("Daily Spending")
        daily_chart = create_daily_spending_chart(df)
        st.plotly_chart(daily_chart, use_container_width=True)
        
        st.subheader("Top Merchants")
        merchants_chart = create_top_merchants_chart(df)
        st.plotly_chart(merchants_chart, use_container_width=True)
        
        st.subheader("Monthly Average")
        monthly_chart = create_monthly_average_chart(df)
        st.plotly_chart(monthly_chart, use_container_width=True)

# ---------- Add Expense ----------
with tab2:
    st.header("➕ Add Expense")
    
    with st.form("add_expense_form", clear_on_submit=True):
        # Date input
        expense_date = st.date_input(
            "Date",
            value=date.today(),
            help="Select the date of the expense"
        )
        
        # Amount and merchant in same row
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input(
                "Amount (¥)",
                min_value=0.0,
                step=100.0,
                help="Enter the expense amount"
            )
        with col2:
            merchant = st.text_input(
                "Merchant",
                help="Enter the name of the merchant or store"
            )
        
        # Category and payment method in same row
        col3, col4 = st.columns(2)
        with col3:
            category = st.selectbox(
                "Category",
                options=list(DEFAULT_CATEGORIES.keys()),
                help="Select the expense category"
            )
        with col4:
            payment_method = st.selectbox(
                "Payment Method",
                options=["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Other"],
                help="Select the payment method used"
            )
        
        # Optional description
        description = st.text_area(
            "Description (optional)",
            help="Add any additional notes about the expense"
        )
        
        # Submit button
        submitted = st.form_submit_button(
            "Add Expense",
            use_container_width=True,
            help="Save this expense to the database"
        )
        
        if submitted:
            try:
                # Validate inputs
                if not merchant.strip():
                    raise ValidationError("Merchant name is required")
                if amount <= 0:
                    raise ValidationError("Amount must be greater than 0")
                
                # Create expense data
                expense_data = {
                    "date": expense_date,
                    "merchant": merchant.strip(),
                    "amount": amount,
                    "category": category,
                    "description": description.strip(),
                    "payment_method": payment_method
                }
                
                # Add to database
                add_expense(expense_data)
                show_success("✅ Expense added successfully!")
                time.sleep(1)
                st.rerun()
                
            except (ValidationError, DatabaseError) as e:
                show_error(str(e))

# ---------- Receipt Scanner ----------
with tab3:
    st.header("📸 Scan Receipt")
    
    # File upload with validation
    uploaded_files = st.file_uploader(
        "Upload Receipt(s)",
        type=list(ALLOWED_IMAGE_EXTENSIONS),
        accept_multiple_files=True,
        help="Upload one or more receipt images for automatic processing"
    )

    lang_option = st.radio(
        "Language for OCR",
        options=["English", "Japanese", "Both"],
        index=2,
        horizontal=True,
        help="Select the language(s) to use for text recognition"
    )
    lang_map = {
        "English": "eng",
        "Japanese": "jpn",
        "Both": "eng+jpn"
    }
    selected_lang = lang_map[lang_option]

    if uploaded_files:
        for uploaded in uploaded_files:
            try:
                # Validate file size
                file_size = len(uploaded.getvalue()) / (1024 * 1024)  # Convert to MB
                if file_size > 10:
                    show_error(f"File {uploaded.name} is too large (max 10MB)")
                    continue

                st.markdown(f"### Processing: {uploaded.name}")
                
                temp_path = RECEIPTS_DIR / uploaded.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded.read())

                # Show original image with cropping
                col1, col2 = st.columns(2)
                with col1:
                    st.image(temp_path, caption="Original Receipt", use_container_width=True)

                with col2:
                    with st.spinner("Processing receipt..."):
                        # Extract data with confidence scores
                        extracted_data = extract_text_from_image(temp_path, lang=selected_lang)
                        
                        # Create form with extracted data
                        with st.form(f"receipt_form_{uploaded.name}", clear_on_submit=True):
                            st.markdown("### Extracted Data")
                            
                            # Date field with confidence indicator
                            date_col, date_conf_col = st.columns([3, 1])
                            with date_col:
                                expense_date = st.date_input(
                                    "Date",
                                    value=extracted_data.date.date() if extracted_data.date else date.today(),
                                    help="Select the date of the expense"
                                )
                            with date_conf_col:
                                st.metric("Date Confidence", f"{extracted_data.date_confidence:.0%}")
                            
                            # Merchant field with confidence indicator
                            merchant_col, merchant_conf_col = st.columns([3, 1])
                            with merchant_col:
                                merchant = st.text_input(
                                    "Merchant",
                                    value=extracted_data.merchant or "",
                                    help="Enter the name of the merchant or store"
                                )
                            with merchant_conf_col:
                                st.metric("Merchant Confidence", f"{extracted_data.merchant_confidence:.0%}")
                            
                            # Amount field with confidence indicator
                            amount_col, amount_conf_col = st.columns([3, 1])
                            with amount_col:
                                amount = st.number_input(
                                    "Amount (¥)",
                                    value=float(extracted_data.amount) if extracted_data.amount else 0.0,
                                    min_value=0.0,
                                    step=100.0,
                                    help="Enter the expense amount"
                                )
                            with amount_conf_col:
                                st.metric("Amount Confidence", f"{extracted_data.amount_confidence:.0%}")
                            
                            # Category and payment method
                            cat_col, pay_col = st.columns(2)
                            with cat_col:
                                category = st.selectbox(
                                    "Category",
                                    list(DEFAULT_CATEGORIES.keys()),
                                    help="Select the expense category"
                                )
                            with pay_col:
                                payment_method = st.selectbox(
                                    "Payment Method",
                                    ["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Other"],
                                    help="Select the payment method used"
                                )
                            
                            # Description and raw text
                            description = st.text_area(
                                "Description (optional)",
                                help="Add any additional notes about the expense"
                            )
                            
                            with st.expander("Show Raw OCR Text"):
                                st.text_area(
                                    "Raw Text",
                                    value=extracted_data.raw_text,
                                    height=150,
                                    disabled=True
                                )
                            
                            # Save button
                            save = st.form_submit_button(
                                "💾 Save Expense",
                                use_container_width=True,
                                help="Save this expense to the database"
                            )
                            
                            if save:
                                try:
                                    if not merchant.strip():
                                        raise ValidationError("Merchant name is required")
                                    if amount <= 0:
                                        raise ValidationError("Amount must be greater than 0")

                                    expense_data = {
                                        "date": expense_date,
                                        "merchant": merchant,
                                        "amount": amount,
                                        "category": category,
                                        "description": description,
                                        "payment_method": payment_method,
                                        "receipt_path": str(temp_path)
                                    }

                                    with st.spinner("Saving expense..."):
                                        # Move receipt to processed folder
                                        processed_path = RECEIPTS_DIR / temp_path.name
                                        temp_path.rename(processed_path)
                                        expense_data["receipt_path"] = str(processed_path)
                                        
                                        # Save to database
                                        add_expense(expense_data)
                                        show_success(f"✅ Expense from {uploaded.name} added successfully!")
                                        time.sleep(1)
                                        st.rerun()

                                except (ValidationError, DatabaseError) as e:
                                    show_error(str(e))

            except Exception as e:
                show_error(f"Error processing {uploaded.name}: {str(e)}")
                if temp_path.exists():
                    temp_path.unlink()  # Clean up temp file

# ---------- All Expenses ----------
with tab4:
    st.header("📋 All Expenses")

    if df.empty:
        st.info("No expenses to display.")
    else:
        # Search and filter
        search = st.text_input(
            "🔍 Search expenses",
            value=st.session_state.search_query,
            help="Search by merchant, category, or description"
        )
        
        if search != st.session_state.search_query:
            st.session_state.search_query = search
            st.session_state.page_number = 1
            st.rerun()

        # Filter expenses based on search
        if search:
            search_lower = search.lower()
            df = df[
                df['merchant'].str.lower().str.contains(search_lower) |
                df['category'].str.lower().str.contains(search_lower) |
                df['description'].str.lower().str.contains(search_lower, na=False)
            ]

        # Display expenses with pagination
        st.markdown("### 🧾 Your Expenses")
        
        total_pages = max(1, -(-len(df) // st.session_state.expenses_per_page))  # Ceiling division, minimum 1 page
        
        # Only show pagination if there's more than one page
        if total_pages > 1:
            col1, col2, col3 = st.columns([2, 3, 2])
            with col2:
                page_nums = st.select_slider(
                    "Page",
                    options=range(1, total_pages + 1),
                    value=min(st.session_state.page_number, total_pages),
                    key="page_slider"
                )
                if page_nums != st.session_state.page_number:
                    st.session_state.page_number = page_nums
                    st.rerun()
        else:
            st.session_state.page_number = 1

        # Display expenses for current page
        start_idx = (st.session_state.page_number - 1) * st.session_state.expenses_per_page
        end_idx = min(start_idx + st.session_state.expenses_per_page, len(df))
        page_expenses = df.iloc[start_idx:end_idx]

        # Show total count
        total_count = len(df)
        if total_count > 0:
            st.markdown(f"Showing {start_idx + 1}-{end_idx} of {total_count} expense{'s' if total_count != 1 else ''}")

        for _, row in page_expenses.iterrows():
            # Check if this row is marked for confirmation
            confirm_key = f"confirm_{row['id']}"
            delete_key = f"delete_{row['id']}"
            confirm_delete = st.session_state.get(confirm_key, False)

            cols = st.columns([4, 1])

            with cols[0]:
                st.markdown(
                    f"📅 **{row['date'].strftime('%Y-%m-%d')}** &nbsp;&nbsp; "
                    f"🏪 **{row['merchant']}** &nbsp;&nbsp; "
                    f"💴 **¥{row['amount']:,.0f}** &nbsp;&nbsp; "
                    f"🗂️ {row['category']} &nbsp;&nbsp; "
                    f"💳 {row['payment_method'] or '—'} &nbsp;&nbsp; "
                    f"📝 {row['description'] or ''}",
                    unsafe_allow_html=True
                )

            with cols[1]:
                if not confirm_delete:
                    if st.button("🗑️ Delete", key=delete_key):
                        st.session_state[confirm_key] = True
                        st.rerun()
                else:
                    st.warning("Are you sure you want to delete this expense?")

                    col_confirm, col_cancel = st.columns([1, 1])
                    with col_confirm:
                        if st.button("✅ Yes", key=f"yes_{row['id']}"):
                            try:
                                with st.spinner("Deleting expense..."):
                                    delete_expense(row['id'])
                                    del st.session_state[confirm_key]
                                    show_success("Expense deleted successfully!")
                                    st.rerun()
                            except DatabaseError as e:
                                show_error(f"Error deleting expense: {str(e)}")
                    with col_cancel:
                        if st.button("❌ No", key=f"cancel_{row['id']}"):
                            del st.session_state[confirm_key]
                            st.rerun()

        st.markdown("---")

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="expenses.csv",
            mime="text/csv",
            help="Download all filtered expenses as CSV"
        )

# ---------- Analytics ----------
with tab5:
    st.header("📊 Enhanced Analytics")
    
    if df.empty:
        st.warning("No expenses found. Add some expenses to see analytics.")
    else:
        # Time period selector
        time_period = st.radio(
            "Select Time Period",
            ["Last 30 Days", "Last 3 Months", "Last 6 Months", "Last Year", "All Time"],
            horizontal=True
        )
        
        # Filter data based on selected time period
        today = pd.Timestamp.now()
        if time_period == "Last 30 Days":
            df = df[df['date'] >= today - pd.Timedelta(days=30)]
        elif time_period == "Last 3 Months":
            df = df[df['date'] >= today - pd.Timedelta(days=90)]
        elif time_period == "Last 6 Months":
            df = df[df['date'] >= today - pd.Timedelta(days=180)]
        elif time_period == "Last Year":
            df = df[df['date'] >= today - pd.Timedelta(days=365)]
        
        # Create tabs for different analytics views
        analytics_tabs = st.tabs([
            "Budget Tracking",
            "Spending Trends",
            "Time Comparisons",
            "Forecasting"
        ])
        
        # Budget Tracking Tab
        with analytics_tabs[0]:
            st.subheader("Budget Tracking")
            
            # Monthly budget tracking
            budget_fig = create_budget_tracking_chart(df)
            st.plotly_chart(budget_fig, use_container_width=True)
            
            # Category budget allocation
            allocation_fig = create_category_budget_allocation(df)
            st.plotly_chart(allocation_fig, use_container_width=True)
            
            # Budget alerts
            current_month = datetime.now().replace(day=1)
            month_mask = (df['date'] >= current_month)
            monthly_spending = df[month_mask].groupby('category')['amount'].sum()
            
            for category, budget in BUDGET_LIMITS.items():
                spent = monthly_spending.get(category, 0)
                percentage = (spent / budget) * 100
                
                if percentage >= BUDGET_ALERT_THRESHOLDS["danger"]:
                    st.error(f"⚠️ {category}: {percentage:.1f}% of budget used (¥{spent:,.0f} / ¥{budget:,.0f})")
                elif percentage >= BUDGET_ALERT_THRESHOLDS["warning"]:
                    st.warning(f"⚠️ {category}: {percentage:.1f}% of budget used (¥{spent:,.0f} / ¥{budget:,.0f})")
        
        # Spending Trends Tab
        with analytics_tabs[1]:
            st.subheader("Spending Trends")
            
            # Daily spending pattern
            daily_spending = df.groupby('date')['amount'].sum()
            
            # Calculate key metrics
            total_spent = daily_spending.sum()
            avg_daily = daily_spending.mean()
            max_daily = daily_spending.max()
            
            # Display metrics
            metrics_cols = st.columns(4)
            metrics_cols[0].metric("Total Spent", f"¥{total_spent:,.0f}")
            metrics_cols[1].metric("Avg. Daily", f"¥{avg_daily:,.0f}")
            metrics_cols[2].metric("Max. Daily", f"¥{max_daily:,.0f}")
            metrics_cols[3].metric("Days Tracked", len(daily_spending))
            
            # Weekly comparison chart
            weekly_fig = create_weekly_comparison_chart(df)
            st.plotly_chart(weekly_fig, use_container_width=True)
        
        # Time Comparisons Tab
        with analytics_tabs[2]:
            st.subheader("Time Comparisons")
            
            # Monthly comparison across years
            monthly_fig = create_monthly_comparison_chart(df)
            st.plotly_chart(monthly_fig, use_container_width=True)
            
            # Year-over-year analysis
            if len(df['date'].dt.year.unique()) > 1:
                yearly_spending = df.groupby(df['date'].dt.year)['amount'].sum()
                yoy_change = (yearly_spending.iloc[-1] / yearly_spending.iloc[-2] - 1) * 100
                st.metric(
                    "Year-over-Year Change",
                    f"{yoy_change:+.1f}%",
                    delta_color="inverse"
                )
        
        # Forecasting Tab
        with analytics_tabs[3]:
            st.subheader("Spending Forecast")
            
            # Forecast period selector
            forecast_days = st.slider(
                "Forecast Period (Days)",
                min_value=7,
                max_value=90,
                value=30,
                step=7,
                help="Select the number of days to forecast"
            )
            
            # Create forecast
            forecast_fig, predicted_total = create_spending_forecast(df, forecast_days)
            st.plotly_chart(forecast_fig, use_container_width=True)
            
            # Display forecast summary
            st.info(f"Predicted spending for the next {forecast_days} days: ¥{predicted_total:,.0f}")
            
            # Monthly average for comparison
            monthly_avg = df.groupby(
                [df['date'].dt.year, df['date'].dt.month]
            )['amount'].sum().mean()
            
            st.metric(
                "Forecast vs Monthly Average",
                f"¥{predicted_total:,.0f}",
                f"{((predicted_total - monthly_avg) / monthly_avg * 100):+.1f}%"
            )

# ---------- Data Management ----------
with tab6:
    st.header("🗄️ Data Management")
    
    # Create tabs for different data management features
    data_tabs = st.tabs([
        "Import/Export",
        "Backup/Restore",
        "Bulk Operations"
    ])
    
    # Import/Export Tab
    with data_tabs[0]:
        st.subheader("Import/Export Data")
        
        # Import section
        with st.expander("Import from CSV", expanded=True):
            uploaded_csv = st.file_uploader(
                "Upload CSV File",
                type=["csv"],
                help="Upload a CSV file with expense data"
            )
            
            if uploaded_csv:
                try:
                    # Save uploaded file temporarily
                    temp_csv = Path("data") / "temp_import.csv"
                    with open(temp_csv, "wb") as f:
                        f.write(uploaded_csv.getvalue())
                    
                    # Preview data
                    df = pd.read_csv(temp_csv)
                    st.write("Preview of data to import:")
                    st.dataframe(df.head())
                    
                    if st.button("Import Data", use_container_width=True):
                        with st.spinner("Importing data..."):
                            total, imported, errors = import_from_csv(temp_csv)
                            
                            st.success(f"✅ Imported {imported} out of {total} records")
                            
                            if errors:
                                with st.expander("Show Import Errors"):
                                    for error in errors:
                                        st.error(error)
                    
                    # Clean up
                    temp_csv.unlink()
                    
                except Exception as e:
                    show_error(f"Error importing data: {str(e)}")
        
        # Export section
        with st.expander("Export to CSV", expanded=True):
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                start_date = st.date_input(
                    "Start Date",
                    value=None,
                    help="Optional: Export expenses from this date"
                )
            
            with export_col2:
                end_date = st.date_input(
                    "End Date",
                    value=None,
                    help="Optional: Export expenses until this date"
                )
            
            if st.button("Export Data", use_container_width=True):
                try:
                    # Create export file
                    export_file = Path("data") / f"expenses_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    
                    # Export data
                    with st.spinner("Exporting data..."):
                        record_count = export_to_csv(
                            export_file,
                            start_date=start_date if start_date else None,
                            end_date=end_date if end_date else None
                        )
                    
                    # Offer download
                    with open(export_file, "rb") as f:
                        st.download_button(
                            "📥 Download Export",
                            f,
                            file_name=export_file.name,
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    st.success(f"✅ Exported {record_count} records")
                    
                    # Clean up
                    export_file.unlink()
                    
                except Exception as e:
                    show_error(f"Error exporting data: {str(e)}")
    
    # Backup/Restore Tab
    with data_tabs[1]:
        st.subheader("Backup & Restore")
        
        # Backup section
        with st.expander("Create Backup", expanded=True):
            if st.button("Create New Backup", use_container_width=True):
                try:
                    with st.spinner("Creating backup..."):
                        backup_file = create_backup(Path("data/backups"))
                        st.success(f"✅ Backup created: {backup_file}")
                        
                        # Offer download
                        with open(backup_file, "rb") as f:
                            st.download_button(
                                "📥 Download Backup",
                                f,
                                file_name=Path(backup_file).name,
                                mime="application/json",
                                use_container_width=True
                            )
                except Exception as e:
                    show_error(f"Error creating backup: {str(e)}")
        
        # Restore section
        with st.expander("Restore from Backup", expanded=True):
            uploaded_backup = st.file_uploader(
                "Upload Backup File",
                type=["json"],
                help="Upload a backup file to restore"
            )
            
            if uploaded_backup:
                try:
                    # Save uploaded file temporarily
                    temp_backup = Path("data") / "temp_restore.json"
                    with open(temp_backup, "wb") as f:
                        f.write(uploaded_backup.getvalue())
                    
                    # Show warning
                    st.warning("⚠️ Restoring from backup will replace all existing data!")
                    
                    if st.button("Restore Data", use_container_width=True):
                        with st.spinner("Restoring data..."):
                            total, restored = restore_from_backup(temp_backup)
                            st.success(f"✅ Restored {restored} out of {total} records")
                    
                    # Clean up
                    temp_backup.unlink()
                    
                except Exception as e:
                    show_error(f"Error restoring backup: {str(e)}")
    
    # Bulk Operations Tab
    with data_tabs[2]:
        st.subheader("Bulk Operations")
        
        # Get all expenses
        df = get_expenses_df()
        
        if df.empty:
            st.warning("No expenses found")
        else:
            # Add select column
            df['select'] = False
            
            # Show data with selection
            selection = st.data_editor(
                df,
                hide_index=True,
                column_config={
                    "select": st.column_config.CheckboxColumn(
                        "Select",
                        help="Select expenses for bulk operations",
                        default=False
                    )
                },
                use_container_width=True
            )
            
            # Get selected rows
            selected_rows = selection[selection['select']]
            selected_count = len(selected_rows)
            
            if selected_count > 0:
                st.info(f"Selected {selected_count} expenses")
                
                # Bulk operations
                operation = st.selectbox(
                    "Choose Operation",
                    ["Update Category", "Delete Selected"]
                )
                
                if operation == "Update Category":
                    new_category = st.selectbox(
                        "New Category",
                        list(DEFAULT_CATEGORIES.keys())
                    )
                    
                    if st.button("Update Category", use_container_width=True):
                        with st.spinner("Updating categories..."):
                            success_count = 0
                            for _, row in selected_rows.iterrows():
                                try:
                                    update_expense(row['id'], {
                                        'date': row['date'],
                                        'merchant': row['merchant'],
                                        'amount': row['amount'],
                                        'category': new_category,
                                        'description': row['description'],
                                        'payment_method': row['payment_method'],
                                        'receipt_path': row['receipt_path']
                                    })
                                    success_count += 1
                                except Exception as e:
                                    st.error(f"Error updating expense {row['id']}: {str(e)}")
                            
                            st.success(f"✅ Updated {success_count} expenses")
                            st.rerun()
                
                elif operation == "Delete Selected":
                    if st.button("Delete Selected", use_container_width=True, type="primary"):
                        with st.spinner("Deleting expenses..."):
                            success_count = 0
                            for _, row in selected_rows.iterrows():
                                try:
                                    delete_expense(row['id'])
                                    success_count += 1
                                except Exception as e:
                                    st.error(f"Error deleting expense {row['id']}: {str(e)}")
                            
                            st.success(f"✅ Deleted {success_count} expenses")
                            st.rerun()

# Show success/error messages
if st.session_state.show_success:
    st.success(st.session_state.success_message)
if st.session_state.show_error:
    st.error(st.session_state.error_message)
    # Clear error after showing
    st.session_state.show_error = False
    st.session_state.error_message = ""
