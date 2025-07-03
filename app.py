"""
Expense Tracker Application - Supabase Multi-User Version
"""
import streamlit as st
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
import time
import plotly.graph_objects as go

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
    get_user_settings,
    update_user_settings,
    update_category_budget,
    add_custom_category,
    delete_custom_category,
    get_available_categories,
    DEFAULT_CATEGORIES,
    BUDGET_LIMITS
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
    create_weekly_spending_heatmap,
    create_category_trend_chart,
    create_payment_method_chart
)
from config.settings import (
    CURRENCY_SYMBOL,
    ALLOWED_IMAGE_EXTENSIONS,
    MAX_IMAGE_SIZE_MB,
    RECEIPTS_DIR,
    PROCESSED_RECEIPTS_DIR,
    FAILED_RECEIPTS_DIR,
    STREAMLIT_CONFIG
)
from utils.supabase_client import supabase

# ---- Streamlit Page Config ----
st.set_page_config(**STREAMLIT_CONFIG)

# ---- Supabase Auth ----

# 1. Track auth mode and user
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"
if "user" not in st.session_state:
    # Try to get existing session
    try:
        session = supabase.auth.get_session()
        if session and session.user:
            st.session_state.user = session.user
        else:
            st.session_state.user = None
    except Exception:
        st.session_state.user = None

def switch_auth_mode():
    st.session_state.auth_mode = "signup" if st.session_state.auth_mode == "login" else "login"

def login():
    st.title("🧙‍♂️ Expensei")
    st.caption("Let Expensei guide your money journey!")
    
    st.header("🔐 Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Login", use_container_width=True):
            try:
                auth_response = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })
                if not auth_response.user:
                    st.error("Invalid email or password.")
                    return
                st.session_state.user = auth_response.user
                st.success("Logged in!")
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")
    with col2:
        st.button("Don't have an account? Sign Up", on_click=switch_auth_mode, use_container_width=True)

def signup():
    st.title("🧙‍♂️ Expensei")
    st.caption("Let Expensei guide your money journey!")
    
    st.header("📝 Sign Up")
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_password")
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Sign Up", use_container_width=True):
            try:
                auth_response = supabase.auth.sign_up({
                    "email": email,
                    "password": password
                })
                if auth_response.user:
                    st.success("Signup successful! Please check your email for a confirmation link, then log in.")
                    st.session_state.auth_mode = "login"
                else:
                    st.error("Signup failed. Try a different email.")
            except Exception as e:
                st.error(f"Signup failed: {e}")
    with col2:
        st.button("Already have an account? Login", on_click=switch_auth_mode, use_container_width=True)

def logout():
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state.user = None
    st.success("Logged out.")
    st.rerun()

# 2. Show login or signup screen until authenticated
if st.session_state.user is None:
    if st.session_state.auth_mode == "login":
        login()
    else:
        signup()
    st.stop()

# 3. User is authenticated: show main app
st.title("🧙‍♂️ Expensei")
st.caption("Let Expensei guide your money journey!")

# Sidebar
with st.sidebar:
    st.write(f"👋 Hello, {st.session_state.user.email}")
    if st.button("Logout", use_container_width=True):
        logout()

user_id = st.session_state.user.id  # This is the user's unique id for ALL queries

# Initialize session state variables
if "show_success" not in st.session_state:
    st.session_state.show_success = False
    st.session_state.success_message = ""

if "show_error" not in st.session_state:
    st.session_state.show_error = False
    st.session_state.error_message = ""

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Dashboard"

if "page_number" not in st.session_state:
    st.session_state.page_number = 1

if "expenses_per_page" not in st.session_state:
    st.session_state.expenses_per_page = 20

if "search_query" not in st.session_state:
    st.session_state.search_query = ""

# ---- Helper Functions ----
def show_success(message: str) -> None:
    st.session_state.show_success = True
    st.session_state.success_message = message

def show_error(message: str) -> None:
    st.session_state.show_error = True
    st.session_state.error_message = message

def change_page(page: int) -> None:
    st.session_state.page_number = page

# ---- Tab Layout ----
tabs = st.tabs([
    "📊 Dashboard",
    "➕ Add Expense",
    "📸 Scan Receipt",
    "📋 All Expenses",
    "📊 Enhanced Analytics",
    "🗄️ Data Management"
])
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

# ---- Load All Expenses for This User ----
try:
    df = get_expenses_df(user_id)
except DatabaseError as e:
    show_error(f"Database error: {str(e)}")
    df = pd.DataFrame()

# ---------- Dashboard ----------
with tab1:
    st.header("📊 Dashboard")
    
    if df.empty:
        st.info("No expenses to analyze. Add some expenses to see analytics!")
    else:
        # Calculate monthly and yearly spending
        today = pd.Timestamp.now()
        current_month_df = df[df['date'].dt.to_period('M') == today.to_period('M')]
        current_year_df = df[df['date'].dt.year == today.year]
        
        monthly_spending = current_month_df['amount'].sum()
        yearly_spending = current_year_df['amount'].sum()
        daily_avg = df.groupby('date')['amount'].sum().mean()
        total_transactions = len(df)
        
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric("Monthly Spending", f"¥{monthly_spending:,.0f}")
        with kpi2:
            st.metric("Yearly Spending", f"¥{yearly_spending:,.0f}")
        with kpi3:
            st.metric("Daily Average", f"¥{daily_avg:,.0f}")
        
        st.markdown("---")
        
        # Create daily spending trend with daily totals
        daily_totals = df.groupby('date')['amount'].sum().reset_index()
        daily_totals = daily_totals.sort_values('date')
        
        # Create the daily spending trend chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_totals['date'],
            y=daily_totals['amount'],
            mode='lines+markers',
            name='Daily Total',
            line=dict(color='#00A6FB', width=2),
            marker=dict(size=8, color='#0582CA')
        ))
        
        fig.update_layout(
            title='Daily Spending Trend',
            xaxis_title='Date',
            yaxis_title='Amount (¥)',
            hovermode='x unified',
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                tickformat='%Y-%m-%d'
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                tickprefix='¥',
                tickformat=',d'
            )
        )
        
        st.plotly_chart(fig, use_container_width=True, key="daily_spending")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_spending_by_category_pie(df.to_dict('records')), use_container_width=True, key="dashboard_category_pie")
        with col2:
            st.plotly_chart(create_top_merchants_chart(df), use_container_width=True, key="dashboard_merchants")
        st.plotly_chart(create_monthly_average_chart(df), use_container_width=True, key="monthly_average")

# ---------- Add Expense ----------
with tab2:
    st.header("➕ Add Expense")
    with st.form("add_expense_form", clear_on_submit=True):
        expense_date = st.date_input("Date", value=date.today(), help="Select the date of the expense")
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Amount (¥)", min_value=0.0, step=100.0, help="Enter the expense amount")
        with col2:
            merchant = st.text_input("Merchant", help="Enter the name of the merchant or store")
        col3, col4 = st.columns(2)
        with col3:
            category = st.selectbox("Category", options=list(DEFAULT_CATEGORIES.keys()), help="Select the expense category")
        with col4:
            payment_method = st.selectbox("Payment Method", options=["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Other"], help="Select the payment method used")
        description = st.text_area("Description (optional)", help="Add any additional notes about the expense")
        submitted = st.form_submit_button("Add Expense", use_container_width=True, help="Save this expense to the database")
        if submitted:
            try:
                if not merchant.strip():
                    raise ValidationError("Merchant name is required")
                if amount <= 0:
                    raise ValidationError("Amount must be greater than 0")
                expense_data = {
                    "date": expense_date,
                    "merchant": merchant.strip(),
                    "amount": amount,
                    "category": category,
                    "description": description.strip(),
                    "payment_method": payment_method,
                }
                add_expense(user_id, expense_data)
                show_success("✅ Expense added successfully!")
                time.sleep(1)
                st.rerun()
            except (ValidationError, DatabaseError) as e:
                show_error(str(e))

# ---------- Receipt Scanner ----------
with tab3:
    st.header("📸 Scan Receipt")
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
                file_size = len(uploaded.getvalue()) / (1024 * 1024)  # Convert to MB
                if file_size > MAX_IMAGE_SIZE_MB:
                    show_error(f"File {uploaded.name} is too large (max {MAX_IMAGE_SIZE_MB}MB)")
                    continue
                st.markdown(f"### Processing: {uploaded.name}")
                temp_path = RECEIPTS_DIR / uploaded.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded.read())
                col1, col2 = st.columns(2)
                with col1:
                    st.image(temp_path, caption="Original Receipt", use_container_width=True)
                with col2:
                    with st.spinner("Processing receipt..."):
                        extracted_data = extract_text_from_image(temp_path, lang=selected_lang)
                        with st.form(f"receipt_form_{uploaded.name}", clear_on_submit=True):
                            st.markdown("### Extracted Data")
                            date_col, date_conf_col = st.columns([3, 1])
                            with date_col:
                                expense_date = st.date_input(
                                    "Date",
                                    value=extracted_data.date.date() if extracted_data.date else date.today(),
                                    help="Select the date of the expense"
                                )
                            with date_conf_col:
                                st.metric("Date Confidence", f"{extracted_data.date_confidence:.0%}")
                            merchant_col, merchant_conf_col = st.columns([3, 1])
                            with merchant_col:
                                merchant = st.text_input(
                                    "Merchant",
                                    value=extracted_data.merchant or "",
                                    help="Enter the name of the merchant or store"
                                )
                            with merchant_conf_col:
                                st.metric("Merchant Confidence", f"{extracted_data.merchant_confidence:.0%}")
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
                                    
                                    st.write("Preparing expense data...")
                                    expense_data = {
                                        "date": expense_date,
                                        "merchant": merchant.strip(),
                                        "amount": amount,
                                        "category": category,
                                        "description": description.strip() if description else None,
                                        "payment_method": payment_method,
                                        "receipt_path": str(temp_path)
                                    }
                                    
                                    st.write("Moving receipt file...")
                                    processed_path = PROCESSED_RECEIPTS_DIR / temp_path.name
                                    temp_path.rename(processed_path)
                                    expense_data["receipt_path"] = str(processed_path)
                                    
                                    st.write("Saving to database...")
                                    try:
                                        expense_id = add_expense(user_id, expense_data)
                                        if not expense_id:
                                            raise DatabaseError("No expense ID returned from database")
                                        st.write(f"✅ Expense saved with ID: {expense_id}")
                                        show_success(f"✅ Expense from {uploaded.name} added successfully!")
                                        time.sleep(1)
                                        st.rerun()
                                    except DatabaseError as db_e:
                                        st.error(f"Database error: {str(db_e)}")
                                        # Move receipt back if database save failed
                                        if processed_path.exists():
                                            processed_path.rename(temp_path)
                                except ValidationError as val_e:
                                    st.error(f"Validation error: {str(val_e)}")
                                except Exception as e:
                                    st.error(f"Unexpected error: {str(e)}")
                                    if 'temp_path' in locals() and temp_path.exists():
                                        temp_path.unlink()
                                    failed_path = FAILED_RECEIPTS_DIR / uploaded.name
                                    with open(failed_path, "wb") as f:
                                        f.write(uploaded.getvalue())
            except Exception as e:
                show_error(f"Error processing {uploaded.name}: {str(e)}")
                if 'temp_path' in locals() and temp_path.exists():
                    temp_path.unlink()
                failed_path = FAILED_RECEIPTS_DIR / uploaded.name
                with open(failed_path, "wb") as f:
                    f.write(uploaded.getvalue())

# ---------- All Expenses ----------
with tab4:
    st.header("📋 All Expenses")
    if df.empty:
        st.info("No expenses to display.")
    else:
        search = st.text_input(
            "🔍 Search expenses",
            value=st.session_state.search_query,
            help="Search by merchant, category, or description"
        )
        if search != st.session_state.search_query:
            st.session_state.search_query = search
            st.session_state.page_number = 1
            st.rerun()
        if search:
            search_lower = search.lower()
            df = df[
                df['merchant'].str.lower().str.contains(search_lower) |
                df['category'].str.lower().str.contains(search_lower) |
                df['description'].str.lower().str.contains(search_lower, na=False)
            ]
        st.markdown("### 🧾 Your Expenses")
        total_pages = max(1, -(-len(df) // st.session_state.expenses_per_page))
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
        start_idx = (st.session_state.page_number - 1) * st.session_state.expenses_per_page
        end_idx = min(start_idx + st.session_state.expenses_per_page, len(df))
        page_expenses = df.iloc[start_idx:end_idx]
        total_count = len(df)
        if total_count > 0:
            st.markdown(f"Showing {start_idx + 1}-{end_idx} of {total_count} expense{'s' if total_count != 1 else ''}")
        for _, row in page_expenses.iterrows():
            edit_key = f"edit_{row['id']}"
            confirm_key = f"confirm_{row['id']}"
            delete_key = f"delete_{row['id']}"
            confirm_delete = st.session_state.get(confirm_key, False)
            is_editing = st.session_state.get(edit_key, False)
            if not is_editing:
                cols = st.columns([4, 1, 1])
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
                                        delete_expense(user_id, row['id'])
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
                with st.form(key=f"edit_form_{row['id']}"):
                    st.markdown("### ✏️ Edit Expense")
                    edited_date = st.date_input(
                        "Date",
                        value=row['date'].date(),
                        help="Select the date of the expense"
                    )
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
                    edited_description = st.text_area(
                        "Description (optional)",
                        value=row['description'] if row['description'] else "",
                        help="Add any additional notes about the expense"
                    )
                    col5, col6 = st.columns(2)
                    with col5:
                        if st.form_submit_button("💾 Save Changes"):
                            try:
                                if not edited_merchant.strip():
                                    raise ValidationError("Merchant name is required")
                                if edited_amount <= 0:
                                    raise ValidationError("Amount must be greater than 0")
                                expense_data = {
                                    "date": edited_date,
                                    "merchant": edited_merchant.strip(),
                                    "amount": edited_amount,
                                    "category": edited_category,
                                    "description": edited_description.strip(),
                                    "payment_method": edited_payment_method,
                                    "receipt_path": row['receipt_path'] if 'receipt_path' in row else None
                                }
                                update_expense(user_id, row['id'], expense_data)
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
        st.info("No expenses to analyze. Add some expenses to see enhanced analytics!")
    else:
        # Budget Tracking
        st.subheader("🎯 Budget Tracking")
        st.plotly_chart(create_budget_tracking_chart(df), use_container_width=True)
        
        # Spending Forecast
        st.subheader("📈 Spending Forecast")
        forecast_days = st.slider("Forecast Days", min_value=7, max_value=90, value=30)
        forecast_chart, predicted_total = create_spending_forecast(df, days_ahead=forecast_days)
        st.plotly_chart(forecast_chart, use_container_width=True)
        st.info(f"Predicted spending for next {forecast_days} days: ¥{predicted_total:,.0f}")
        
        # Weekly Analysis
        st.subheader("📅 Weekly Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_weekly_comparison_chart(df), use_container_width=True)
        with col2:
            st.plotly_chart(create_weekly_spending_heatmap(df), use_container_width=True)
        
        # Category Analysis
        st.subheader("📊 Category Analysis")
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(create_category_budget_allocation(df), use_container_width=True)
        with col4:
            st.plotly_chart(create_payment_method_chart(df), use_container_width=True)
        
        # Trend Analysis
        st.subheader("📈 Trend Analysis")
        categories = ["All"] + list(DEFAULT_CATEGORIES.keys())
        selected_category = st.selectbox("Select Category", categories)
        if selected_category == "All":
            st.plotly_chart(create_category_trend_chart(df), use_container_width=True)
        else:
            st.plotly_chart(create_spending_trend(df.to_dict('records'), category_filter=selected_category), use_container_width=True)

# ---------- Data Management ----------
with tab6:
    st.header("🗄️ Data Management")
    
    # Settings Section
    st.subheader("⚙️ Settings")
    settings_tab1, settings_tab2 = st.tabs(["Budget Settings", "Category Management"])
    
    with settings_tab1:
        st.markdown("### 💰 Budget Settings")
        try:
            user_settings = get_user_settings(user_id)
            current_budget = user_settings.get("monthly_budget", sum(BUDGET_LIMITS.values()))
            budget_by_category = user_settings.get("budget_by_category", BUDGET_LIMITS)
            
            # Overall monthly budget
            new_budget = st.number_input(
                "Set Monthly Budget (¥)",
                min_value=0.0,
                value=float(current_budget),
                step=10000.0,
                help="Set your total monthly budget target"
            )
            
            if new_budget != current_budget:
                try:
                    user_settings["monthly_budget"] = new_budget
                    update_user_settings(user_id, user_settings)
                    st.success("Monthly budget updated!")
                except Exception as e:
                    st.error(f"Error updating budget: {str(e)}")
            
            # Category budgets
            st.markdown("### Category Budgets")
            st.info("Set individual category budgets below. These will help track spending in each category.")
            
            categories = get_available_categories(user_id)
            for category in categories:
                current_cat_budget = budget_by_category.get(category, 0)
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_cat_budget = st.number_input(
                        f"{category} Budget (¥)",
                        min_value=0.0,
                        value=float(current_cat_budget),
                        step=1000.0,
                        key=f"budget_{category}"
                    )
                with col2:
                    if new_cat_budget != current_cat_budget:
                        if st.button("Save", key=f"save_{category}"):
                            try:
                                update_category_budget(user_id, category, new_cat_budget)
                                st.success(f"Budget for {category} updated!")
                            except Exception as e:
                                st.error(f"Error updating category budget: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading budget settings: {str(e)}")
    
    with settings_tab2:
        st.markdown("### 🏷️ Category Management")
        st.info("Add custom categories or modify existing ones. Note: Categories in use cannot be deleted.")
        
        # Add new category
        with st.form("add_category_form"):
            st.markdown("#### Add New Category")
            new_category = st.text_input("Category Name")
            color = st.color_picker("Category Color", "#4ECDC4")
            submit = st.form_submit_button("Add Category")
            
            if submit and new_category:
                try:
                    add_custom_category(user_id, new_category.strip(), color)
                    st.success(f"Category '{new_category}' added!")
                    st.rerun()
                except ValidationError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Error adding category: {str(e)}")
        
        # List and manage existing categories
        st.markdown("#### Existing Categories")
        try:
            categories = get_available_categories(user_id)
            for category in categories:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{category}**")
                with col2:
                    if category not in DEFAULT_CATEGORIES:
                        if st.button("Delete", key=f"delete_{category}"):
                            try:
                                # Check if category is in use
                                category_expenses = df[df['category'] == category]
                                if not category_expenses.empty:
                                    st.error(f"Cannot delete '{category}' - it is being used by {len(category_expenses)} expenses.")
                                else:
                                    delete_custom_category(user_id, category)
                                    st.success(f"Category '{category}' deleted!")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting category: {str(e)}")
        except Exception as e:
            st.error(f"Error loading categories: {str(e)}")
    
    st.markdown("---")
    
    # Import/Export Section
    st.subheader("📤 Import/Export")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Import from CSV")
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            try:
                # Save uploaded file temporarily
                temp_path = Path("temp_import.csv")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Import the file
                total_records, imported_count, errors = import_from_csv(user_id, temp_path)
                
                # Remove temporary file
                temp_path.unlink()
                
                if errors:
                    st.warning(f"Imported {imported_count} out of {total_records} records. {len(errors)} errors occurred.")
                    with st.expander("Show Errors"):
                        for error in errors:
                            st.error(error)
                else:
                    st.success(f"Successfully imported {imported_count} expenses!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error importing file: {str(e)}")
    
    with col2:
        st.markdown("### Export to CSV")
        export_all = st.checkbox("Export all data", value=True)
        if not export_all:
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input("Start Date", value=df['date'].min().date() if not df.empty else date.today())
            with col_end:
                end_date = st.date_input("End Date", value=df['date'].max().date() if not df.empty else date.today())
        
        if st.button("Export"):
            try:
                export_path = Path("expenses_export.csv")
                if export_all:
                    num_records = export_to_csv(user_id, export_path)
                else:
                    num_records = export_to_csv(user_id, export_path, start_date=start_date, end_date=end_date)
                
                if num_records > 0:
                    with open(export_path, "rb") as f:
                        st.download_button(
                            label="📥 Download CSV",
                            data=f,
                            file_name="expenses_export.csv",
                            mime="text/csv"
                        )
                    export_path.unlink()  # Clean up
                else:
                    st.warning("No expenses found for the selected period.")
            except Exception as e:
                st.error(f"Error exporting data: {str(e)}")
    
    # Backup/Restore Section
    st.markdown("---")
    st.subheader("💾 Backup & Restore")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### Create Backup")
        if st.button("Create Backup"):
            try:
                backup_dir = Path("backups")
                backup_file = create_backup(user_id, backup_dir)
                with open(backup_file, "rb") as f:
                    st.download_button(
                        label="📥 Download Backup",
                        data=f,
                        file_name=Path(backup_file).name,
                        mime="application/json"
                    )
                Path(backup_file).unlink()  # Clean up
                st.success("Backup created successfully!")
            except Exception as e:
                st.error(f"Error creating backup: {str(e)}")
    
    with col4:
        st.markdown("### Restore from Backup")
        uploaded_backup = st.file_uploader("Upload backup file", type=['json'])
        if uploaded_backup:
            try:
                # Save uploaded file temporarily
                temp_path = Path("temp_backup.json")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_backup.getvalue())
                
                if st.button("Restore"):
                    total_records, restored_count = restore_from_backup(user_id, temp_path)
                    temp_path.unlink()  # Clean up
                    
                    if restored_count > 0:
                        st.success(f"Successfully restored {restored_count} out of {total_records} expenses!")
                        st.rerun()
                    else:
                        st.warning("No expenses were restored. Please check the backup file.")
            except Exception as e:
                st.error(f"Error restoring backup: {str(e)}")
                if temp_path.exists():
                    temp_path.unlink()  # Clean up on error
