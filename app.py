"""
Expense Tracker Application
"""
import streamlit as st
from pathlib import Path
from datetime import datetime, date, timedelta
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
    restore_from_backup,
    init_db
)
from utils.ocr import extract_text_from_image, process_receipt
from utils.charts import (
    create_daily_spending_chart,
    create_top_merchants_chart,
    create_monthly_average_chart,
    create_budget_tracking_chart,
    create_category_budget_allocation,
    create_weekly_comparison_chart,
    create_monthly_comparison_chart,
    create_spending_forecast,
    create_spending_trend,
    create_spending_by_category_pie,
    get_available_categories
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

# Initialize database
init_db()

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
        st.info("No expenses to analyze. Add some expenses to see analytics!")
    else:
        # Calculate KPI metrics
        total_spending = df['amount'].sum()
        daily_avg = df.groupby('date')['amount'].sum().mean()
        total_transactions = len(df)
        
        # Display KPI metrics in columns
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric("Total Spending", f"¥{total_spending:,.0f}")
        with kpi2:
            st.metric("Daily Average", f"¥{daily_avg:,.0f}")
        with kpi3:
            st.metric("Total Transactions", f"{total_transactions:,}")
        
        st.markdown("---")
        
        # Daily spending trend
        st.plotly_chart(
            create_daily_spending_chart(df),
            use_container_width=True,
            key="daily_spending"
        )
        
        # Create two columns for the pie chart and top merchants
        col1, col2 = st.columns(2)
        
        with col1:
            # Category spending distribution
            st.plotly_chart(
                create_spending_by_category_pie(
                    df.to_dict('records')
                ),
                use_container_width=True,
                key="dashboard_category_pie"
            )
        
        with col2:
            # Top merchants chart
            st.plotly_chart(
                create_top_merchants_chart(df),
                use_container_width=True,
                key="dashboard_merchants"
            )
        
        # Monthly average chart below
        st.plotly_chart(
            create_monthly_average_chart(df),
            use_container_width=True,
            key="monthly_average"
        )

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
                                        processed_path = PROCESSED_RECEIPTS_DIR / temp_path.name
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
                if 'temp_path' in locals() and temp_path.exists():
                    temp_path.unlink()  # Clean up temp file
                # Move to failed folder
                failed_path = FAILED_RECEIPTS_DIR / uploaded.name
                with open(failed_path, "wb") as f:
                    f.write(uploaded.getvalue())

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

        total_pages = max(1, -(-len(df) // st.session_state.expenses_per_page))
        
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
            # Check if this row is marked for editing or deletion
            edit_key = f"edit_{row['id']}"
            confirm_key = f"confirm_{row['id']}"
            delete_key = f"delete_{row['id']}"
            confirm_delete = st.session_state.get(confirm_key, False)
            is_editing = st.session_state.get(edit_key, False)

            if not is_editing:
                cols = st.columns([4, 1, 1])  # Added another column for edit button

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
                    if st.button("✏️ Edit", key=f"edit_btn_{row['id']}"):
                        st.session_state[edit_key] = True
                        st.rerun()

                with cols[2]:
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
            else:
                # Edit form
                with st.form(key=f"edit_form_{row['id']}"):
                    st.markdown("### ✏️ Edit Expense")
                    
                    # Date input
                    edited_date = st.date_input(
                        "Date",
                        value=row['date'].date(),
                        help="Select the date of the expense"
                    )
                    
                    # Amount and merchant in same row
                    col1, col2 = st.columns(2)
                    with col1:
                        edited_amount = st.number_input(
                            "Amount (¥)",
                            value=float(row['amount']),
                            min_value=0.0,
                            step=100.0,
                            help="Enter the expense amount"
                        )
                    with col2:
                        edited_merchant = st.text_input(
                            "Merchant",
                            value=row['merchant'],
                            help="Enter the name of the merchant or store"
                        )
                    
                    # Category and payment method in same row
                    col3, col4 = st.columns(2)
                    with col3:
                        edited_category = st.selectbox(
                            "Category",
                            options=list(DEFAULT_CATEGORIES.keys()),
                            index=list(DEFAULT_CATEGORIES.keys()).index(row['category']) if row['category'] in DEFAULT_CATEGORIES else 0,
                            help="Select the expense category"
                        )
                    with col4:
                        edited_payment_method = st.selectbox(
                            "Payment Method",
                            options=["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Other"],
                            index=["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Other"].index(row['payment_method']) if row['payment_method'] in ["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Other"] else 0,
                            help="Select the payment method used"
                        )
                    
                    # Optional description
                    edited_description = st.text_area(
                        "Description (optional)",
                        value=row['description'] if row['description'] else "",
                        help="Add any additional notes about the expense"
                    )
                    
                    # Form buttons
                    col5, col6 = st.columns(2)
                    with col5:
                        if st.form_submit_button("💾 Save Changes"):
                            try:
                                if not edited_merchant.strip():
                                    raise ValidationError("Merchant name is required")
                                if edited_amount <= 0:
                                    raise ValidationError("Amount must be greater than 0")
                                
                                # Update expense data
                                expense_data = {
                                    "date": edited_date,
                                    "merchant": edited_merchant.strip(),
                                    "amount": edited_amount,
                                    "category": edited_category,
                                    "description": edited_description.strip(),
                                    "payment_method": edited_payment_method,
                                    "receipt_path": row['receipt_path'] if 'receipt_path' in row else None
                                }
                                
                                # Update in database
                                update_expense(row['id'], expense_data)
                                del st.session_state[edit_key]
                                show_success("✅ Expense updated successfully!")
                                st.rerun()
                                
                            except (ValidationError, DatabaseError) as e:
                                show_error(str(e))
                    
                    with col6:
                        if st.form_submit_button("❌ Cancel"):
                            del st.session_state[edit_key]
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

# ---------- Enhanced Analytics ----------
with tab5:
    st.header("📊 Enhanced Analytics")

    if df.empty:
        st.warning("No expenses found. Add some expenses to see analytics.")
    else:
        # Time period selector with +/- buttons
        st.markdown("### Select Time Period")
        col1, col2, col3 = st.columns([0.1, 2.8, 0.1])  # Adjusted ratios to bring buttons closer
        
        # Initialize time_period in session state if not exists
        if 'time_period' not in st.session_state:
            st.session_state.time_period = 30
        
        with col1:
            st.write("")  # Add some vertical spacing
            if st.button("➖", help="Decrease by 1 day", key="decrease_days"):
                st.session_state.time_period = max(7, st.session_state.time_period - 1)
        
        with col2:
            time_period = st.slider(
                "Days to Analyze",
                min_value=7,
                max_value=90,
                value=st.session_state.time_period,
                step=1,
                help="Select the number of days of data to analyze"
            )
            if time_period != st.session_state.time_period:
                st.session_state.time_period = time_period
        
        with col3:
            st.write("")  # Add some vertical spacing
            if st.button("➕", help="Increase by 1 day", key="increase_days"):
                st.session_state.time_period = min(90, st.session_state.time_period + 1)

        # Filter data based on selected time period
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=st.session_state.time_period)
        filtered_df = df[df['date'] >= cutoff_date]
        
        # Create tabs for different analytics views
        analytics_tabs = st.tabs([
            "Budget Tracking",
            "Time Comparisons",
            "Forecasting"
        ])
        
        # Budget Tracking Tab
        with analytics_tabs[0]:
            st.subheader("Budget Tracking")
            
            # Monthly budget tracking using filtered data
            budget_fig = create_budget_tracking_chart(filtered_df)
            st.plotly_chart(budget_fig, use_container_width=True, key="budget_tracking")
            
            # Category budget allocation using filtered data
            allocation_fig = create_category_budget_allocation(filtered_df)
            st.plotly_chart(allocation_fig, use_container_width=True, key="budget_allocation")
            
            # Budget alerts using filtered data
            monthly_spending = filtered_df.groupby('category')['amount'].sum()
            
            for category, budget in BUDGET_LIMITS.items():
                # Adjust budget for the time period
                days_in_period = min(st.session_state.time_period, 30)  # Cap at 30 days for budget comparison
                adjusted_budget = (budget / 30) * days_in_period
                spent = monthly_spending.get(category, 0)
                percentage = (spent / adjusted_budget * 100) if adjusted_budget > 0 else 0
                
                if percentage >= BUDGET_ALERT_THRESHOLDS["danger"]:
                    st.error(f"⚠️ {category}: {percentage:.1f}% of budget used (¥{spent:,.0f} / ¥{adjusted_budget:,.0f})")
                elif percentage >= BUDGET_ALERT_THRESHOLDS["warning"]:
                    st.warning(f"⚠️ {category}: {percentage:.1f}% of budget used (¥{spent:,.0f} / ¥{adjusted_budget:,.0f})")
        
        # Time Comparisons Tab
        with analytics_tabs[1]:
            st.subheader("Time Comparisons")
            
            # Weekly comparison chart using filtered data
            weekly_fig = create_weekly_comparison_chart(filtered_df)
            st.plotly_chart(weekly_fig, use_container_width=True, key="weekly_comparison")
            
            # Monthly comparison chart using filtered data
            monthly_fig = create_monthly_comparison_chart(filtered_df)
            st.plotly_chart(monthly_fig, use_container_width=True, key="monthly_comparison")
            
            # Year-over-year analysis if filtered data spans multiple years
            years = filtered_df['date'].dt.year.unique()
            if len(years) > 1:
                yearly_spending = filtered_df.groupby(filtered_df['date'].dt.year)['amount'].sum()
                yoy_change = (yearly_spending.iloc[-1] / yearly_spending.iloc[-2] - 1) * 100
                st.metric(
                    "Year-over-Year Change",
                    f"{yoy_change:+.1f}%",
                    delta_color="inverse"
                )
        
        # Forecasting Tab
        with analytics_tabs[2]:
            st.subheader("Spending Forecast")
            
            # Create forecast using filtered data
            forecast_fig, predicted_total = create_spending_forecast(filtered_df, st.session_state.time_period)
            st.plotly_chart(forecast_fig, use_container_width=True, key="spending_forecast")
            
            # Display forecast summary
            st.info(f"Predicted spending for the next {st.session_state.time_period} days: ¥{predicted_total:,.0f}")
            
            # Monthly average from filtered data for comparison
            monthly_avg = filtered_df.groupby(
                [filtered_df['date'].dt.year, filtered_df['date'].dt.month]
            )['amount'].sum().mean()
            
            st.metric(
                "Forecast vs Monthly Average",
                f"¥{predicted_total:,.0f}",
                f"{((predicted_total - monthly_avg) / monthly_avg * 100):+.1f}%"
            )

# ---------- Data Management ----------
with tab6:
    st.header("🗄️ Data Management")
    
    # Create two columns for export and import
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Export Data")
        if st.button("Download Expenses CSV"):
            csv = convert_df_to_csv(df)
            st.download_button(
                label="Click to Download",
                data=csv,
                file_name='expenses.csv',
                mime='text/csv',
            )
    
    with col2:
        st.subheader("Import Data")
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                import_df = pd.read_csv(uploaded_file)
                required_columns = ['date', 'amount', 'category', 'merchant', 'description']
                
                if all(col in import_df.columns for col in required_columns):
                    # Convert date strings to datetime
                    import_df['date'] = pd.to_datetime(import_df['date'])
                    
                    if st.button("Import Data"):
                        # Append new data to database
                        for _, row in import_df.iterrows():
                            add_expense(
                                date=row['date'],
                                amount=row['amount'],
                                category=row['category'],
                                merchant=row['merchant'],
                                description=row['description']
                            )
                        st.success("Data imported successfully!")
                        st.experimental_rerun()
                else:
                    st.error("Invalid CSV format. Required columns: date, amount, category, merchant, description")
            except Exception as e:
                st.error(f"Error importing data: {str(e)}")
    
    # Database maintenance section
    st.markdown("---")
    st.subheader("Database Maintenance")
    
    # Backup database
    if st.button("Create Database Backup"):
        try:
            backup_path = create_database_backup()
            st.success(f"Database backup created successfully at: {backup_path}")
        except Exception as e:
            st.error(f"Error creating backup: {str(e)}")
    
    # Restore database
    uploaded_backup = st.file_uploader("Restore from Backup", type=['db'])
    if uploaded_backup is not None:
        if st.button("Restore Database"):
            try:
                restore_database_from_backup(uploaded_backup)
                st.success("Database restored successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error restoring database: {str(e)}")
    
    # Clear database option with confirmation
    st.markdown("---")
    st.subheader("⚠️ Danger Zone")
    
    if st.button("Clear All Data"):
        st.warning("This will permanently delete all expense data. Are you sure?")
        if st.button("Yes, I'm Sure"):
            try:
                clear_database()
                st.success("Database cleared successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error clearing database: {str(e)}")

# Display notifications
if st.session_state.show_success:
    st.success(st.session_state.success_message)
    st.session_state.show_success = False
    st.session_state.success_message = ""

if st.session_state.show_error:
    st.error(st.session_state.error_message)
    st.session_state.show_error = False
    st.session_state.error_message = ""
