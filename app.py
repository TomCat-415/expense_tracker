"""
Expense Tracker Application - Supabase Multi-User Version
"""
import streamlit as st
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
import time
import plotly.graph_objects as go
from utils import stripe_client
import os

# Use Render's assigned port if available, fallback to 8501 for local
port = int(os.environ.get("PORT", 8501))
os.environ["STREAMLIT_SERVER_PORT"] = str(port)
os.environ["STREAMLIT_SERVER_ENABLECORS"] = "false"
os.environ["STREAMLIT_SERVER_ENABLEXSRFPROTECTION"] = "false"

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
    sync_default_categories_with_user_settings,
    DEFAULT_CATEGORIES,
    BUDGET_LIMITS,
    # Recurring expenses functions
    add_recurring_expense,
    get_recurring_expenses,
    update_recurring_expense,
    delete_recurring_expense,
    get_due_recurring_expenses,
    generate_expense_from_recurring,
    suggest_recurring_expenses,
    calculate_next_due_date,
    # Averaging functions
    calculate_monthly_equivalent,
    get_averaging_expenses,
    apply_expense_averaging
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

st.cache_data.clear()

# ---- Streamlit Page Config ----
st.set_page_config(**STREAMLIT_CONFIG)

# Force cache clear on app start to ensure fresh data
def clear_all_caches():
    """Clear all cached data."""
    st.cache_data.clear()
    if hasattr(st, 'cache_resource'):
        st.cache_resource.clear()

# Clear cache on startup
clear_all_caches()

# Add custom styles and JavaScript
st.markdown("""
<style>
    .stTabs [data-baseweb="tab"]:hover {
        border-bottom: 2px solid #4ECDC4;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 2px solid #4ECDC4;
        color: #4ECDC4;
    }
    .stSelectbox > div > div > div {
        background-color: rgba(78, 205, 196, 0.1);
    }
    .stButton > button {
        background-color: #4ECDC4;
        color: white;
        border: none;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #3BB5A8;
        transform: translateY(-2px);
    }
    .stButton > button[kind="primary"] {
        background-color: #4ECDC4;
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #3BB5A8;
    }
    .stMetric > div > div > div > div {
        color: #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

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
            # Try to refresh the session
            response = supabase.auth.refresh_session()
            if response and response.user:
                st.session_state.user = response.user
            else:
                st.session_state.user = None
    except Exception:
        st.session_state.user = None

def switch_auth_mode():
    st.session_state.auth_mode = "signup" if st.session_state.auth_mode == "login" else "login"

def login():
    # Display logo
    st.markdown('<div style="margin-left: -20px;">', unsafe_allow_html=True)
    
    # Load logo with error handling
    logo_path = Path("logo_optimized.png")
    if logo_path.exists():
        try:
            st.image(str(logo_path), width=600)
        except Exception as e:
            st.error(f"Error loading logo: {e}")
            st.markdown("### 💰 Expense Tracker")
    else:
        st.markdown("### 💰 Expense Tracker")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.header("🔐 Login")
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        remember_me = st.checkbox("Remember me", value=True)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
        
        with col2:
            st.form_submit_button("Don't have an account? Sign Up", on_click=switch_auth_mode, use_container_width=True)
        
        # Forgot password option - placed after main buttons to avoid conflicts with auto-fill
        forgot_password = st.form_submit_button("🔑 Forgot Password?", use_container_width=False)
        
        # Handle login
        if submitted:
            try:
                auth_response = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })
                if not auth_response.user:
                    st.error("Invalid email or password.")
                    return
                st.session_state.user = auth_response.user
                
                # If remember me is checked, we don't need to do anything special
                # as Supabase will handle the refresh token automatically
                
                st.success("Logged in!")
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")
        
        # Handle forgot password
        if forgot_password:
            if not email:
                st.error("Please enter your email address first, then click 'Forgot Password?'")
            else:
                try:
                    with st.spinner("Sending password reset email..."):
                        supabase.auth.reset_password_email(email)
                    st.success("✅ Password reset email sent!")
                    st.info("""
                    📧 **Check your email for password reset instructions.**
                    
                    **Next steps:**
                    1. Check your email inbox (and spam folder)
                    2. Click the password reset link
                    3. Create a new password
                    4. Return here to log in with your new password
                    """)
                except Exception as e:
                    st.error(f"Failed to send password reset email: {e}")

def signup():
    # Display logo
    st.markdown('<div style="margin-left: -20px;">', unsafe_allow_html=True)
    
    # Load logo with error handling
    logo_path = Path("logo_optimized.png")
    if logo_path.exists():
        try:
            st.image(str(logo_path), width=600)
        except Exception as e:
            st.error(f"Error loading logo: {e}")
            st.markdown("### 💰 Expense Tracker")
    else:
        st.markdown("### 💰 Expense Tracker")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.header("📝 Sign Up")
    
    # Show password requirements
    st.info("📋 Password requirements: At least 6 characters long")
    
    with st.form("signup_form", clear_on_submit=False):
        email = st.text_input("Email", key="signup_email", placeholder="Enter your email address")
        password = st.text_input("Password", type="password", key="signup_password", 
                                placeholder="Enter a strong password")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            submitted = st.form_submit_button("Sign Up", use_container_width=True)
            if submitted:
                # Input validation
                if not email or not password:
                    st.error("Please fill in both email and password fields.")
                    return
                
                # Basic email validation
                if "@" not in email or "." not in email:
                    st.error("Please enter a valid email address.")
                    return
                
                # Password length validation
                if len(password) < 6:
                    st.error("Password must be at least 6 characters long.")
                    return
                
                try:
                    with st.spinner("Creating your account..."):
                        auth_response = supabase.auth.sign_up({
                            "email": email,
                            "password": password
                        })
                    
                    # Check if user already exists - Supabase might still return a user object
                    # even if the email is already registered
                    if auth_response.user:
                        # Check if this is a new user or existing user
                        if hasattr(auth_response.user, 'email_confirmed_at') and auth_response.user.email_confirmed_at:
                            # User already exists and is confirmed
                            st.warning("⚠️ **This email is already registered!**")
                            st.info("""
                            🔑 **You already have an account with this email.**
                            
                            **Please log in instead:**
                            1. Click "Already have an account? Login" below
                            2. Use your existing password
                            3. If you forgot your password, use the "Forgot Password?" link on the login page
                            """)
                        else:
                            # New signup or unconfirmed user
                            st.balloons()
                            st.success("🎉 Signup successful!")
                            
                            # Prominent email confirmation message
                            st.info("""
                            📧 **Important: Check Your Email!**
                            
                            We've sent a confirmation link to your email address.
                            You must click the link in the email to activate your account before you can log in.
                            
                            **Next steps:**
                            1. Check your email inbox (and spam folder)
                            2. Click the confirmation link
                            3. Return here to log in
                            """)
                            
                            st.session_state.auth_mode = "login"
                            st.rerun()
                    else:
                        st.error("Signup failed. Please try again or contact support.")
                        
                except Exception as e:
                    error_message = str(e).lower()
                    if "already" in error_message or "exists" in error_message:
                        st.warning("⚠️ **This email is already registered!**")
                        st.info("""
                        🔑 **You already have an account with this email.**
                        
                        **Please log in instead:**
                        1. Click "Already have an account? Login" below
                        2. Use your existing password
                        3. If you forgot your password, use the "Forgot Password?" link on the login page
                        """)
                    elif "password" in error_message:
                        st.error("Password doesn't meet requirements. Please try a stronger password.")
                    elif "invalid" in error_message:
                        st.error("Invalid email format. Please check your email address.")
                    else:
                        st.error(f"Signup failed: {e}")
        
        with col2:
            st.form_submit_button("Already have an account? Login", on_click=switch_auth_mode, use_container_width=True)

def logout():
    try:
        supabase.auth.sign_out()
        # Clear any stored session data
        if 'user' in st.session_state:
            del st.session_state.user
    except Exception:
        pass
    st.success("Logged out.")
    st.rerun()

# 2. Show landing info + login/signup screen until authenticated
if st.session_state.user is None:

    # 🎯 Show Stripe-friendly info *before* login
    if st.session_state.auth_mode == "login":
        st.markdown("## 💸 Welcome to Expensei")
        st.markdown("""
        Track smarter. Save faster.  
        Expensei helps you manage spending, crush short-term goals, and grow your savings — effortlessly.

        **Features:**
        - 🧾 Receipt scanning  
        - 📊 Real-time analytics  
        - 🎯 Custom budget goals  
        - ☁️ Secure cloud backup  

        👉 Try [Expensei Pro](https://buy.stripe.com/test_bJe3cufIs1iF1lkcLH5AQ01) for just ¥500/month
        """)
        st.divider()
        login()
    
    else:
        signup()

    st.stop()

# 3. User is authenticated: show main app
# Display logo
st.markdown('<div style="margin-left: -20px;">', unsafe_allow_html=True)

# Load logo with error handling
logo_path = Path("logo_optimized.png")
if logo_path.exists():
    try:
        st.image(str(logo_path), width=600)
    except Exception as e:
        st.error(f"Error loading logo: {e}")
        st.markdown("### 💰 Expense Tracker")
else:
    st.markdown("### 💰 Expense Tracker")

st.markdown('</div>', unsafe_allow_html=True)

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
    "🗄️ Data Management",
    "🔄 Recurring Expenses",
    "☕️ Coffee"
])
tab_mapping = {
    "📊 Dashboard": 0,
    "➕ Add Expense": 1,
    "📸 Scan Receipt": 2,
    "📋 All Expenses": 3,
    "📊 Enhanced Analytics": 4,
    "🗄️ Data Management": 5,
    "🔄 Recurring Expenses": 6,
    "☕️ Coffee": 7
}
active_index = tab_mapping.get(st.session_state.active_tab, 0)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = tabs

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
        # Get user settings for excluded categories
        try:
            user_settings = get_user_settings(user_id)
            excluded_categories = user_settings.get('excluded_dashboard_categories', [])
            
            # Initialize session state for settings if not exists
            if 'settings_initialized' not in st.session_state:
                st.session_state.settings_initialized = False
            
            # Set rent to be excluded by default if no preferences are set (only once)
            if not excluded_categories and not st.session_state.settings_initialized:
                excluded_categories = ['🏠 Rent']
                user_settings['excluded_dashboard_categories'] = excluded_categories
                update_user_settings(user_id, user_settings)
                st.session_state.settings_initialized = True
            elif excluded_categories:
                # User has existing preferences, mark as initialized
                st.session_state.settings_initialized = True
                
        except Exception as e:
            excluded_categories = ['🏠 Rent']  # Default to excluding rent
        
        # Get available categories for this user
        available_categories = get_available_categories(user_id)
        
        # Filter dataframe based on excluded categories (silent filtering)
        if excluded_categories:
            filtered_df = df[~df['category'].isin(excluded_categories)]
        else:
            filtered_df = df
        
        # Check if filtered dataframe is empty
        if filtered_df.empty:
            st.warning("⚠️ All categories are excluded. Please enable at least one category to view analytics.")
            st.stop()
        
        # ==================== APPLY DEFAULT AVERAGING ====================
        # Default to averaging for quarterly/yearly expenses
        df_to_use = apply_expense_averaging(filtered_df, user_id, include_averaging=True)
        
        # Make sure date is datetime
        df_to_use['date'] = pd.to_datetime(df_to_use['date'])

    # ==================== KPI CALCULATIONS ====================
    # Calculate KPIs after all filters are applied
    today = pd.Timestamp.now()
    current_month_df = df_to_use[df_to_use['date'].dt.to_period('M') == today.to_period('M')]
    current_year_df = df_to_use[df_to_use['date'].dt.year == today.year]
    current_week = today.isocalendar().week
    current_year_num = today.isocalendar().year

    current_week_df = df_to_use[
        (df_to_use['date'].dt.isocalendar().week == current_week) &
        (df_to_use['date'].dt.isocalendar().year == current_year_num)
    ]

    monthly_spending = current_month_df['amount'].sum()
    weekly_spending = current_week_df['amount'].sum()
    daily_avg = df_to_use.groupby('date')['amount'].sum().mean()
    yearly_spending = current_year_df['amount'].sum()

    # KPI Columns: Monthly, Weekly, Daily Average, Yearly
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric("Monthly Spending", f"¥{monthly_spending:,.0f}")
    with kpi2:
        st.metric("Weekly Spending", f"¥{weekly_spending:,.0f}")
    with kpi3:
        st.metric("Daily Average", f"¥{daily_avg:,.0f}")
    with kpi4:
        st.metric("Yearly Spending", f"¥{yearly_spending:,.0f}")

    st.markdown("---")
    
    # Create daily spending trend with daily totals
    daily_totals = df_to_use.groupby('date')['amount'].sum().reset_index()
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
    
    # Category pie chart - full width for better visibility
    st.plotly_chart(create_spending_by_category_pie(df_to_use.to_dict('records'), user_id=user_id), use_container_width=True, key="dashboard_category_pie")
    
    # Top merchants chart - full width for better visibility
    st.plotly_chart(create_top_merchants_chart(df_to_use), use_container_width=True, key="dashboard_merchants")
    
    # Monthly average chart
    st.plotly_chart(create_monthly_average_chart(df_to_use), use_container_width=True, key="monthly_average")

    # ==================== CATEGORY FILTERS SECTION ====================
    st.markdown("---")
    
    # Show current filter status
    if excluded_categories:
        excluded_total = df[df['category'].isin(excluded_categories)]['amount'].sum()
        st.info(f"📊 **Filtered View**: Excluding {len(excluded_categories)} categories (¥{excluded_total:,.0f} hidden)")
    
    # Category filter dropdown
    with st.expander("🎯 **Category Filter - Control what categories to show**", expanded=False):
        st.markdown("**💡 Tip**: Rent is filtered out by default to focus on day-to-day spending. Check the Rent box below to include it in your analysis.")
        
        # Show categories as checkboxes in a cleaner layout
        st.markdown("**Select categories to show:**")
        
        # Define callback function to track changes
        def on_category_change():
            st.session_state.category_filter_changed = True
        
        # Create columns for better layout
        num_categories = len(available_categories)
        cols_per_row = 3
        num_rows = (num_categories + cols_per_row - 1) // cols_per_row
        
        updated_excluded = []
        category_index = 0
        
        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                if category_index < num_categories:
                    category = available_categories[category_index]
                    with cols[col_idx]:
                        # Special styling for rent category
                        if category == "🏠 Rent":
                            is_shown = st.checkbox(
                                f"✅ {category}",
                                value=category not in excluded_categories,
                                key=f"show_{category}",
                                help="💡 Include rent to see total spending including fixed costs",
                                on_change=on_category_change
                            )
                        else:
                            is_shown = st.checkbox(
                                f"{category}",
                                value=category not in excluded_categories,
                                key=f"show_{category}",
                                on_change=on_category_change
                            )
                        
                        if not is_shown:
                            updated_excluded.append(category)
                    category_index += 1
        
        # Quick preset buttons
        st.markdown("**Quick Filters:**")
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        with preset_col1:
            if st.button("🏠 Hide Rent Only", key="preset_rent_only"):
                try:
                    user_settings['excluded_dashboard_categories'] = ['🏠 Rent']
                    update_user_settings(user_id, user_settings)
                    st.session_state.category_filter_changed = False  # Reset flag
                    st.rerun()
                except Exception as e:
                    st.error("Error updating settings")
        
        with preset_col2:
            if st.button("🎉 Show All", key="preset_show_all"):
                try:
                    user_settings['excluded_dashboard_categories'] = []
                    update_user_settings(user_id, user_settings)
                    st.session_state.category_filter_changed = False  # Reset flag
                    st.rerun()
                except Exception as e:
                    st.error("Error updating settings")
        
        with preset_col3:
            if st.button("🍽️ Food & Fun Only", key="preset_discretionary"):
                try:
                    keep_categories = [cat for cat in available_categories if any(keyword in cat.lower() for keyword in ['food', 'dining', 'shopping', 'entertainment', 'coffee', 'fun', 'retail', 'treats', 'eats', 'groceries'])]
                    excluded = [cat for cat in available_categories if cat not in keep_categories]
                    user_settings['excluded_dashboard_categories'] = excluded
                    update_user_settings(user_id, user_settings)
                    st.session_state.category_filter_changed = False  # Reset flag
                    st.rerun()
                except Exception as e:
                    st.error("Error updating settings")
        
        # Update settings if changed
        # Only update if the user actually interacted with checkboxes
        # Use session state to track if we should update
        if 'category_filter_changed' in st.session_state and st.session_state.category_filter_changed:
            if updated_excluded != excluded_categories:
                try:
                    user_settings['excluded_dashboard_categories'] = updated_excluded
                    update_user_settings(user_id, user_settings)
                    st.session_state.category_filter_changed = False  # Reset the flag
                    st.rerun()
                except Exception as e:
                    st.error("⚠️ **Database Setup Required**: Please add the excluded_dashboard_categories column to your database.")
                    st.info("📝 **To fix this**: Go to your Supabase SQL Editor and run the `add_excluded_categories_column.sql` script.")
                    st.code("""
-- Copy and paste this into Supabase SQL Editor:
ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS excluded_dashboard_categories JSONB DEFAULT '[]'::jsonb;
CREATE INDEX IF NOT EXISTS idx_user_settings_excluded_categories ON user_settings USING GIN (excluded_dashboard_categories);
COMMENT ON COLUMN user_settings.excluded_dashboard_categories IS 'Array of category names to exclude from dashboard view';
                    """, language="sql")
            else:
                st.session_state.category_filter_changed = False  # Reset the flag even if no change needed

# Initialize session state for amount field
if 'expense_amount' not in st.session_state:
    st.session_state.expense_amount = ""

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
            amount_str = st.text_input(
                "Amount (¥)",
                value="",
                placeholder="Enter amount",
                key="amount_input",
                help="Enter the expense amount"
            )
        with col2:
            merchant_options = sorted(df['merchant'].dropna().unique().tolist())
            
            # Field 1: Select from existing merchants
            selected_existing = st.selectbox(
                "Select existing merchant:",
                options=[""] + merchant_options,
                index=0,
                help="Type to search or select from saved merchants",
                key="existing_merchant_selectbox"
            )
            
            # Field 2: Add new merchant
            new_merchant_input = st.text_input(
                "Or add new merchant:",
                value="",
                placeholder="Type new merchant name...",
                help="Enter a new merchant name",
                key="new_merchant_textinput"
            )
            
            # Determine which merchant to use
            if new_merchant_input.strip():
                merchant = new_merchant_input.strip()
            else:
                merchant = selected_existing

        # Category and payment method in same row
        col3, col4 = st.columns(2)
        with col3:
            # Get user's available categories
            available_categories = get_available_categories(user_id)
            category = st.selectbox(
                "Category",
                options=available_categories,
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
                # Validate and parse amount
                try:
                    amount = float(amount_str.strip())
                    if amount <= 0:
                        raise ValidationError("Amount must be greater than 0")
                except ValueError:
                    show_error("Please enter a valid number for Amount")
                    st.stop()

                if not merchant:
                    show_error("Merchant name is required")
                    st.stop()

                expense_data = {
                    "date": expense_date,
                    "merchant": merchant,
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
            except Exception as e:
                show_error(f"Error adding expense: {str(e)}")

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
                                # Get user's available categories
                                available_categories = get_available_categories(user_id)
                                category = st.selectbox(
                                    "Category",
                                    available_categories,
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
        # Clean search and filter section
        st.markdown("### 🔍 Search & Filter")
        
        col_search, col_filter = st.columns([3, 1])
        with col_search:
            search = st.text_input(
                "🔍 Search expenses",
                value=st.session_state.search_query,
                help="Search by merchant, category, or description",
                placeholder="Type to search merchants, categories, or descriptions..."
            )
        with col_filter:
            # Find the index of the current expenses_per_page value
            page_options = [10, 20, 50, 100]
            try:
                current_index = page_options.index(st.session_state.expenses_per_page)
            except ValueError:
                current_index = 1  # Default to 20 if not found
            
            st.session_state.expenses_per_page = st.selectbox(
                "Show per page",
                options=page_options,
                index=current_index,
                key="items_per_page"
            )
        
        if search != st.session_state.search_query:
            st.session_state.search_query = search
            st.session_state.page_number = 1
            st.rerun()
            
        # Apply search filter
        filtered_df = df.copy()
        if search:
            search_lower = search.lower()
            filtered_df = filtered_df[
                filtered_df['merchant'].str.lower().str.contains(search_lower) |
                filtered_df['category'].str.lower().str.contains(search_lower) |
                filtered_df['description'].str.lower().str.contains(search_lower, na=False)
            ]
        
        # Group by date and paginate by days instead of individual expenses
        filtered_df['date'] = pd.to_datetime(filtered_df['date'])
        daily_groups = filtered_df.groupby(filtered_df['date'].dt.date).apply(lambda x: x.sort_values('date')).reset_index(drop=True)
        unique_dates = sorted(filtered_df['date'].dt.date.unique(), reverse=True)
        
        # Pagination by days
        days_per_page = 7  # Show 7 days per page
        total_pages = max(1, -(-len(unique_dates) // days_per_page))
        
        # Clean pagination controls
        if total_pages > 1:
            st.markdown("### 📄 Navigation")
            
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            
            with col1:
                if st.button("⏮️ First", disabled=st.session_state.page_number == 1, type="primary"):
                    st.session_state.page_number = 1
                    st.rerun()
            
            with col2:
                if st.button("◀️ Previous", disabled=st.session_state.page_number == 1, type="primary"):
                    st.session_state.page_number = max(1, st.session_state.page_number - 1)
                    st.rerun()
            
            with col3:
                # Page number selector
                page_options = list(range(1, total_pages + 1))
                current_page_index = min(st.session_state.page_number, total_pages) - 1  # Convert to 0-based index
                
                current_page = st.selectbox(
                    "Page",
                    options=page_options,
                    index=current_page_index,
                    key="page_selector",
                    format_func=lambda x: f"Page {x} of {total_pages}"
                )
                if current_page != st.session_state.page_number:
                    st.session_state.page_number = current_page
                    st.rerun()
            
            with col4:
                if st.button("Next ▶️", disabled=st.session_state.page_number == total_pages, type="primary"):
                    st.session_state.page_number = min(total_pages, st.session_state.page_number + 1)
                    st.rerun()
            
            with col5:
                if st.button("Last ⏭️", disabled=st.session_state.page_number == total_pages, type="primary"):
                    st.session_state.page_number = total_pages
                    st.rerun()
        else:
            st.session_state.page_number = 1
        
        # Get dates for current page
        start_day_idx = (st.session_state.page_number - 1) * days_per_page
        end_day_idx = min(start_day_idx + days_per_page, len(unique_dates))
        page_dates = unique_dates[start_day_idx:end_day_idx]
        
        # Display summary with better styling
        total_expenses = len(filtered_df)
        if total_expenses > 0:
            st.markdown(f"""
            <div style="background: rgba(78, 205, 196, 0.1); 
                        border-radius: 8px; 
                        padding: 10px 15px; 
                        margin: 10px 0; 
                        border-left: 4px solid #4ECDC4;">
                <strong>📊 Showing {len(page_dates)} day{'s' if len(page_dates) != 1 else ''} • {total_expenses} total expense{'s' if total_expenses != 1 else ''}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Display expenses grouped by day
        for date in page_dates:
            daily_expenses = filtered_df[filtered_df['date'].dt.date == date]
            daily_total = daily_expenses['amount'].sum()
            
            # Day header with total
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #4ECDC4 0%, #44A08D 100%); 
                        color: white; 
                        padding: 12px 20px; 
                        border-radius: 10px; 
                        margin: 20px 0 10px 0;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; font-size: 18px;">
                    📅 {date.strftime('%A, %B %d, %Y')} 
                    <span style="float: right; font-weight: normal;">
                        💰 ¥{daily_total:,.0f} ({len(daily_expenses)} expense{'s' if len(daily_expenses) != 1 else ''})
                    </span>
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display each expense for this day
            for _, row in daily_expenses.iterrows():
                edit_key = f"edit_{row['id']}"
                confirm_key = f"confirm_{row['id']}"
                delete_key = f"delete_{row['id']}"
                confirm_delete = st.session_state.get(confirm_key, False)
                is_editing = st.session_state.get(edit_key, False)
                
                if not is_editing:
                    # Create a card-like layout for each expense
                    with st.container():
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.05); 
                                    border-left: 4px solid #4ECDC4; 
                                    padding: 15px; 
                                    margin: 8px 0; 
                                    border-radius: 8px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-size: 16px; font-weight: bold;">🏪 {row['merchant']}</span>
                                    <span style="margin-left: 15px; font-size: 18px; color: #4ECDC4; font-weight: bold;">¥{row['amount']:,.0f}</span>
                                </div>
                                <div style="text-align: right; font-size: 14px; color: #888;">
                                    🗂️ {row['category']} • 💳 {row['payment_method'] or '—'}
                                </div>
                            </div>
                            {f'<div style="margin-top: 8px; font-size: 14px; color: #666;">📝 {row["description"]}</div>' if row['description'] else ''}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Action buttons
                        col1, col2 = st.columns([6, 1])
                        with col2:
                            btn_col1, btn_col2 = st.columns(2)
                            with btn_col1:
                                if st.button("✏️", key=f"edit_btn_{row['id']}", help="Edit expense"):
                                    st.session_state[edit_key] = True
                                    st.rerun()
                            with btn_col2:
                                if not confirm_delete:
                                    if st.button("🗑️", key=delete_key, help="Delete expense"):
                                        st.session_state[confirm_key] = True
                                        st.rerun()
                        
                        # Confirm delete warning with proper button layout
                        if confirm_delete:
                            st.warning("⚠️ Are you sure you want to delete this expense?")
                            col_cancel, col_confirm = st.columns(2)
                            with col_cancel:
                                if st.button("❌ Cancel", key=f"cancel_{row['id']}", use_container_width=True):
                                    del st.session_state[confirm_key]
                                    st.rerun()
                            with col_confirm:
                                if st.button("✅ Confirm Delete", key=f"yes_{row['id']}", use_container_width=True, type="primary"):
                                    try:
                                        with st.spinner("Deleting expense..."):
                                            delete_expense(user_id, row['id'])
                                            del st.session_state[confirm_key]
                                            show_success("Expense deleted successfully!")
                                            st.rerun()
                                    except DatabaseError as e:
                                        show_error(f"Error deleting expense: {str(e)}")
                
                else:
                    # Edit mode
                    st.markdown(f"### ✏️ Edit Expense")
                    with st.form(f"edit_form_{row['id']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            edited_date = st.date_input("Date", value=row['date'].date())
                            edited_merchant = st.text_input("Merchant", value=row['merchant'])
                            edited_amount = st.number_input("Amount (¥)", min_value=0.0, value=float(row['amount']), step=100.0)
                        with col2:
                            available_categories = get_available_categories(user_id)
                            edited_category = st.selectbox("Category", options=available_categories, index=available_categories.index(row['category']) if row['category'] in available_categories else 0)
                            edited_payment_method = st.selectbox("Payment Method", options=["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Other"], index=0 if not row['payment_method'] else ["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Other"].index(row['payment_method']) if row['payment_method'] in ["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Other"] else 0)
                            edited_description = st.text_input("Description", value=row['description'] if row['description'] else "")
                        
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

        # Summary section with cleaner styling
        st.markdown("---")
        st.markdown("### 📊 Summary")
        
        # Summary statistics
        if not filtered_df.empty:
            total_amount = filtered_df['amount'].sum()
            avg_daily = filtered_df.groupby(filtered_df['date'].dt.date)['amount'].sum().mean()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Amount", f"¥{total_amount:,.0f}")
            with col2:
                st.metric("📈 Daily Average", f"¥{avg_daily:,.0f}")
            with col3:
                st.metric("📅 Days with Expenses", f"{len(unique_dates)}")
        
        # Download button with better styling
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Expenses",
            data=csv,
            file_name="expenses_filtered.csv",
            mime="text/csv",
            help="Download all filtered expenses as CSV",
            type="primary"
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
        
        # Create sub-tabs for different analytics views
        st.markdown("### 📈 Detailed Analysis")
        analysis_tabs = st.tabs([
            "💰 Budget Allocation",
            "🔮 Forecast",
            "📅 Weekly Analysis",
            "🗂️ Categories"
        ])
        
        # Budget Allocation Tab
        with analysis_tabs[0]:
            st.plotly_chart(create_category_budget_allocation(df), use_container_width=True)
            st.plotly_chart(create_payment_method_chart(df), use_container_width=True)
        
        # Forecast Tab
        with analysis_tabs[1]:
            forecast_days = st.slider("Forecast Days", min_value=7, max_value=90, value=30)
            forecast_chart, predicted_total = create_spending_forecast(df, days_ahead=forecast_days)
            st.plotly_chart(forecast_chart, use_container_width=True)
            st.info(f"Predicted spending for next {forecast_days} days: ¥{predicted_total:,.0f}")
        
        # Weekly Analysis Tab
        with analysis_tabs[2]:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_weekly_comparison_chart(df), use_container_width=True)
            with col2:
                st.plotly_chart(create_weekly_spending_heatmap(df), use_container_width=True)
        
        # Categories Tab
        with analysis_tabs[3]:
            # Get user's available categories
            available_categories = get_available_categories(user_id)
            categories = ["All"] + available_categories
            selected_category = st.selectbox("Select Category", categories)
            if selected_category == "All":
                st.plotly_chart(create_category_trend_chart(df, user_id=user_id), use_container_width=True)
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
        
        # Category Presets Section
        st.markdown("### 🎯 Category Presets")
        st.info("Choose a category style that fits your lifestyle. You can still customize it later!")
        
        # Define the presets
        CATEGORY_PRESETS = {
            "Minimalist Mode": {
                "description": "Core essentials only — 8 categories max",
                "ideal_for": "busy users, new budgeters, folks who hate clutter",
                "categories": {
                    "🏠 Housing": "#FF6B6B",
                    "🍽 Food & Dining": "#4ECDC4", 
                    "🚗 Transportation": "#45B7D1",
                    "💡 Utilities": "#FFA07A",
                    "🛍 Shopping": "#98D8C8",
                    "🎭 Entertainment": "#F7DC6F",
                    "🌱 Health & Wellness": "#BB8FCE",
                    "🌀 Miscellaneous": "#747D8C"
                },
                "_metadata": {
                    "categories": {
                        "🏠 Housing": {
                            "color": "#FF6B6B",
                            "keywords": ["rent", "mortgage", "maintenance", "repair", "furniture", "home"]
                        },
                        "🍽 Food & Dining": {
                            "color": "#4ECDC4",
                            "keywords": ["restaurant", "cafe", "coffee", "dining", "food", "takeout", "starbucks"]
                        },
                        "🚗 Transportation": {
                            "color": "#45B7D1",
                            "keywords": ["train", "bus", "taxi", "parking", "gas", "toll", "suica", "pasmo", "uber"]
                        },
                        "💡 Utilities": {
                            "color": "#FFA07A",
                            "keywords": ["electric", "gas", "water", "internet", "phone", "mobile", "utility", "bill"]
                        },
                        "🛍 Shopping": {
                            "color": "#98D8C8",
                            "keywords": ["shopping", "clothing", "amazon", "rakuten", "retail", "store", "mall"]
                        },
                        "🎭 Entertainment": {
                            "color": "#F7DC6F",
                            "keywords": ["movie", "game", "concert", "theater", "netflix", "spotify", "entertainment"]
                        },
                        "🌱 Health & Wellness": {
                            "color": "#BB8FCE",
                            "keywords": ["doctor", "dentist", "pharmacy", "medical", "gym", "fitness", "health", "wellness"]
                        },
                        "🌀 Miscellaneous": {
                            "color": "#747D8C",
                            "keywords": ["miscellaneous", "other", "random", "misc"]
                        }
                    },
                    "auto_categorization_rules": {
                        "min_confidence": 0.6,
                        "fallback_category": "🌀 Miscellaneous",
                        "case_sensitive": False
                    }
                }
            },
            "Family Mode": {
                "description": "Adds child-related, home, and wellness categories",
                "ideal_for": "families, caregivers, or folks with dependents",
                "categories": {
                    "🏠 Rent & Housing": "#FF6B6B",
                    "🍽 Eats & Treats": "#4ECDC4",
                    "👶 Kids & Baby": "#FFB6C1",
                    "💡 Bills & Utilities": "#FFA07A",
                    "🛍 Retail Therapy": "#98D8C8",
                    "💅 Self-Care": "#DDA0DD",
                    "🌱 Wellness": "#BB8FCE",
                    "🧾 Medical & Care": "#87CEEB",
                    "🚗 Transport": "#45B7D1",
                    "🎁 Gifts": "#F7DC6F",
                    "🌀 Misc & One-Offs": "#747D8C"
                },
                "_metadata": {
                    "categories": {
                        "🏠 Rent & Housing": {
                            "color": "#FF6B6B",
                            "keywords": ["rent", "mortgage", "maintenance", "repair", "furniture", "home", "house"]
                        },
                        "🍽 Eats & Treats": {
                            "color": "#4ECDC4",
                            "keywords": ["restaurant", "cafe", "dining", "food", "takeout", "delivery", "starbucks"]
                        },
                        "👶 Kids & Baby": {
                            "color": "#FFB6C1",
                            "keywords": ["toys", "kids", "baby", "child", "daughter", "son", "childcare", "school"]
                        },
                        "💡 Bills & Utilities": {
                            "color": "#FFA07A",
                            "keywords": ["electric", "gas", "water", "internet", "phone", "mobile", "utility", "bill"]
                        },
                        "🛍 Retail Therapy": {
                            "color": "#98D8C8",
                            "keywords": ["shopping", "clothing", "amazon", "rakuten", "retail", "store", "mall"]
                        },
                        "💅 Self-Care": {
                            "color": "#DDA0DD",
                            "keywords": ["spa", "massage", "beauty", "salon", "skincare", "cosmetics", "wellness"]
                        },
                        "🌱 Wellness": {
                            "color": "#BB8FCE",
                            "keywords": ["gym", "fitness", "yoga", "health", "exercise", "wellness", "meditation"]
                        },
                        "🧾 Medical & Care": {
                            "color": "#87CEEB",
                            "keywords": ["doctor", "dentist", "pharmacy", "medical", "hospital", "clinic", "medicine"]
                        },
                        "🚗 Transport": {
                            "color": "#45B7D1",
                            "keywords": ["train", "bus", "taxi", "parking", "gas", "toll", "suica", "pasmo", "uber"]
                        },
                        "🎁 Gifts": {
                            "color": "#F7DC6F",
                            "keywords": ["gift", "present", "birthday", "holiday", "anniversary", "celebration"]
                        },
                        "🌀 Misc & One-Offs": {
                            "color": "#747D8C",
                            "keywords": ["miscellaneous", "other", "random", "one-off", "misc"]
                        }
                    },
                    "auto_categorization_rules": {
                        "min_confidence": 0.6,
                        "fallback_category": "🌀 Misc & One-Offs",
                        "case_sensitive": False
                    }
                }
            },
            "Freelancer Mode": {
                "description": "Adds work-related, tax, and self-investment categories",
                "ideal_for": "creators, small business folks, consultants, remote workers",
                "categories": {
                    "🏠 Rent & Workspace": "#FF6B6B",
                    "🍽 Eats & Coffee": "#4ECDC4",
                    "💻 Tech & Gear": "#20B2AA",
                    "🧠 Education": "#DDA0DD",
                    "💅 Self-Care": "#FFB6C1",
                    "🚗 Transport": "#45B7D1",
                    "💼 Business Expenses": "#F4A460",
                    "🧾 Medical & Insurance": "#87CEEB",
                    "🎁 Client Gifts": "#F7DC6F",
                    "🌀 Misc & One-Offs": "#747D8C"
                },
                "_metadata": {
                    "categories": {
                        "🏠 Rent & Workspace": {
                            "color": "#FF6B6B",
                            "keywords": ["rent", "mortgage", "office", "workspace", "coworking", "furniture", "home"]
                        },
                        "🍽 Eats & Coffee": {
                            "color": "#4ECDC4",
                            "keywords": ["restaurant", "cafe", "coffee", "dining", "food", "takeout", "starbucks"]
                        },
                        "💻 Tech & Gear": {
                            "color": "#20B2AA",
                            "keywords": ["computer", "laptop", "phone", "electronics", "tech", "software", "hardware"]
                        },
                        "🧠 Education": {
                            "color": "#DDA0DD",
                            "keywords": ["course", "training", "learning", "education", "books", "skill", "certification"]
                        },
                        "💅 Self-Care": {
                            "color": "#FFB6C1",
                            "keywords": ["spa", "massage", "beauty", "salon", "skincare", "cosmetics", "wellness"]
                        },
                        "🚗 Transport": {
                            "color": "#45B7D1",
                            "keywords": ["train", "bus", "taxi", "parking", "gas", "toll", "suica", "pasmo", "uber"]
                        },
                        "💼 Business Expenses": {
                            "color": "#F4A460",
                            "keywords": ["business", "office", "supplies", "meeting", "client", "expense", "tax"]
                        },
                        "🧾 Medical & Insurance": {
                            "color": "#87CEEB",
                            "keywords": ["doctor", "dentist", "pharmacy", "medical", "insurance", "health", "clinic"]
                        },
                        "🎁 Client Gifts": {
                            "color": "#F7DC6F",
                            "keywords": ["gift", "present", "client", "business", "celebration", "thank you"]
                        },
                        "🌀 Misc & One-Offs": {
                            "color": "#747D8C",
                            "keywords": ["miscellaneous", "other", "random", "one-off", "misc"]
                        }
                    },
                    "auto_categorization_rules": {
                        "min_confidence": 0.6,
                        "fallback_category": "🌀 Misc & One-Offs",
                        "case_sensitive": False
                    }
                }
            },
            "Full Lifestyle Mode": {
                "description": "Well-rounded, perfect for most everyday users with a mix of fun, practical, and personal",
                "ideal_for": "users who want comprehensive tracking",
                "categories": {
                    "🏠 Rent & Housing": "#FF6B6B",
                    "🍽 Eats & Treats": "#4ECDC4",
                    "☕ Coffee": "#8B4513",
                    "🍺 Alcohol": "#FFD700",
                    "🛍 Retail Therapy": "#98D8C8",
                    "💡 Bills & Utilities": "#FFA07A",
                    "💻 Tech & Gear": "#20B2AA",
                    "🧠 Education": "#DDA0DD",
                    "💅 Self-Care": "#FFB6C1",
                    "👶 Kids & Baby": "#FFB6C1",
                    "🚗 Transport": "#45B7D1",
                    "🌱 Wellness": "#BB8FCE",
                    "🧾 Medical & Care": "#87CEEB",
                    "🎭 Fun & Entertainment": "#F7DC6F",
                    "🎁 Gifts": "#F7DC6F",
                    "🌀 Misc & One-Offs": "#747D8C"
                },
                "_metadata": {
                    "categories": {
                        "🏠 Rent & Housing": {
                            "color": "#FF6B6B",
                            "keywords": ["rent", "mortgage", "maintenance", "repair", "furniture", "home", "house"]
                        },
                        "🍽 Eats & Treats": {
                            "color": "#4ECDC4",
                            "keywords": ["restaurant", "cafe", "dining", "food", "takeout", "delivery", "meal"]
                        },
                        "☕ Coffee": {
                            "color": "#8B4513",
                            "keywords": ["coffee", "starbucks", "cafe", "espresso", "latte", "cappuccino", "bean"]
                        },
                        "🍺 Alcohol": {
                            "color": "#FFD700",
                            "keywords": ["beer", "wine", "alcohol", "bar", "pub", "sake", "whiskey", "vodka"]
                        },
                        "🛍 Retail Therapy": {
                            "color": "#98D8C8",
                            "keywords": ["shopping", "clothing", "amazon", "rakuten", "retail", "store", "mall"]
                        },
                        "💡 Bills & Utilities": {
                            "color": "#FFA07A",
                            "keywords": ["electric", "gas", "water", "internet", "phone", "mobile", "utility", "bill"]
                        },
                        "💻 Tech & Gear": {
                            "color": "#20B2AA",
                            "keywords": ["computer", "laptop", "phone", "electronics", "tech", "software", "hardware"]
                        },
                        "🧠 Education": {
                            "color": "#DDA0DD",
                            "keywords": ["school", "tuition", "books", "course", "training", "learning", "education"]
                        },
                        "💅 Self-Care": {
                            "color": "#FFB6C1",
                            "keywords": ["spa", "massage", "beauty", "salon", "skincare", "cosmetics", "wellness"]
                        },
                        "👶 Kids & Baby": {
                            "color": "#FFB6C1",
                            "keywords": ["toys", "kids", "baby", "child", "daughter", "son", "childcare", "school"]
                        },
                        "🚗 Transport": {
                            "color": "#45B7D1",
                            "keywords": ["train", "bus", "taxi", "parking", "gas", "toll", "suica", "pasmo", "uber"]
                        },
                        "🌱 Wellness": {
                            "color": "#BB8FCE",
                            "keywords": ["gym", "fitness", "yoga", "health", "exercise", "wellness", "meditation"]
                        },
                        "🧾 Medical & Care": {
                            "color": "#87CEEB",
                            "keywords": ["doctor", "dentist", "pharmacy", "medical", "hospital", "clinic", "medicine", "seims", "matsumoto kiyoshi"]
                        },
                        "🎭 Fun & Entertainment": {
                            "color": "#F7DC6F",
                            "keywords": ["movie", "game", "concert", "theater", "netflix", "spotify", "entertainment", "fun"]
                        },
                        "🎁 Gifts": {
                            "color": "#F7DC6F",
                            "keywords": ["gift", "present", "birthday", "holiday", "anniversary", "celebration"]
                        },
                        "🌀 Misc & One-Offs": {
                            "color": "#747D8C",
                            "keywords": ["miscellaneous", "other", "random", "one-off", "misc"]
                        }
                    },
                    "auto_categorization_rules": {
                        "min_confidence": 0.6,
                        "fallback_category": "🌀 Misc & One-Offs",
                        "case_sensitive": False
                    }
                }
            }
        }
        
        # Preset selection
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_preset = st.selectbox(
                "Choose a preset:",
                options=list(CATEGORY_PRESETS.keys()),
                index=0,
                help="Select a category preset that matches your lifestyle"
            )
        
        with col2:
            if st.button("Preview Categories", key="preview_preset"):
                st.session_state.show_preview = True
        
        # Show preset description
        preset_info = CATEGORY_PRESETS[selected_preset]
        st.markdown(f"**Description:** {preset_info['description']}")
        st.markdown(f"**👉 Ideal for:** {preset_info['ideal_for']}")
        st.markdown(f"**📊 Categories:** {len(preset_info['categories'])} categories")
        
        # Show preview if requested
        if st.session_state.get('show_preview', False):
            st.markdown("### 👀 Preview Categories")
            preview_cols = st.columns(3)
            for i, (cat_name, cat_color) in enumerate(preset_info['categories'].items()):
                with preview_cols[i % 3]:
                    st.markdown(f"<div style='background-color: {cat_color}; padding: 8px; border-radius: 4px; margin: 2px; text-align: center; color: white; font-weight: bold;'>{cat_name}</div>", unsafe_allow_html=True)
        
        # Apply preset button with warning
        if st.button("🎯 Apply Preset", key="apply_preset", type="primary"):
            # Get current categories to check if they exist
            current_categories = get_available_categories(user_id)
            
            if len(current_categories) > 0:
                # Show warning
                st.warning("⚠️ **Warning:** Applying a preset will replace your current categories. Are you sure?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Yes, Apply Preset", key="confirm_apply"):
                        try:
                            # Apply the preset
                            settings = get_user_settings(user_id)
                            
                            # Handle presets with metadata vs simple presets
                            if "_metadata" in preset_info:
                                # Full format with metadata
                                settings["custom_categories"] = preset_info["categories"]
                                settings["custom_categories"]["_metadata"] = preset_info["_metadata"]
                            else:
                                # Simple format - just categories
                                settings["custom_categories"] = preset_info["categories"]
                            
                            update_user_settings(user_id, settings)
                            
                            # Clear cache to ensure UI updates
                            st.cache_data.clear()
                            
                            st.success(f"✅ {selected_preset} applied successfully!")
                            st.balloons()
                            time.sleep(2)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error applying preset: {str(e)}")
                with col2:
                    if st.button("❌ Cancel", key="cancel_apply"):
                        st.rerun()
            else:
                # No existing categories, apply directly
                try:
                    settings = get_user_settings(user_id)
                    
                    # Handle presets with metadata vs simple presets
                    if "_metadata" in preset_info:
                        # Full format with metadata
                        settings["custom_categories"] = preset_info["categories"]
                        settings["custom_categories"]["_metadata"] = preset_info["_metadata"]
                    else:
                        # Simple format - just categories
                        settings["custom_categories"] = preset_info["categories"]
                    
                    update_user_settings(user_id, settings)
                    
                    # Clear cache to ensure UI updates
                    st.cache_data.clear()
                    
                    st.success(f"✅ {selected_preset} applied successfully!")
                    st.balloons()
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error applying preset: {str(e)}")
        
        st.markdown("---")
        
        # Debug section to clear custom categories
        st.markdown("### 🔧 Debug & Sync")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Sync with Latest Categories", type="primary"):
                try:
                    settings = get_user_settings(user_id)
                    settings["custom_categories"] = {}  # Clear to force sync
                    update_user_settings(user_id, settings)
                    sync_default_categories_with_user_settings(user_id)
                    st.cache_data.clear()  # Clear cache
                    st.success("✅ Categories synced with latest defaults! Now you should see Rent, Housing, and Groceries as separate categories.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error syncing categories: {str(e)}")
        
        with col2:
            if st.button("🗑️ Clear All Custom Categories", type="secondary"):
                try:
                    settings = get_user_settings(user_id)
                    settings["custom_categories"] = {}  # Clear custom categories
                    update_user_settings(user_id, settings)
                    st.cache_data.clear()  # Clear cache
                    st.success("✅ All custom categories cleared! Default categories restored.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing categories: {str(e)}")
        
        st.markdown("---")
        
        # Existing category management
        st.markdown("### ➕ Custom Categories")
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
                    # Check if this is a custom category (not in default categories)
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

# ---------- Recurring Expenses ----------
with tab7:
    st.header("🔄 Recurring Expenses")
    
    # Get user's recurring expenses
    try:
        recurring_expenses = get_recurring_expenses(user_id)
        due_expenses = get_due_recurring_expenses(user_id)
        suggestions = suggest_recurring_expenses(user_id)
    except Exception as e:
        st.error(f"Error loading recurring expenses: {str(e)}")
        recurring_expenses = []
        due_expenses = []
        suggestions = []
    
    # Show due expenses notification if any
    if due_expenses:
        st.info(f"💡 You have {len(due_expenses)} recurring expense(s) due for approval!")
        
        for due_expense in due_expenses:
            with st.expander(f"Due: {due_expense['name']} - ¥{due_expense['amount']:,.0f}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Merchant:** {due_expense['merchant']}")
                    st.write(f"**Category:** {due_expense['category']}")
                    st.write(f"**Amount:** ¥{due_expense['amount']:,.0f}")
                    st.write(f"**Due Date:** {due_expense['next_due_date']}")
                
                with col2:
                    if st.button("✅ Approve", key=f"approve_{due_expense['id']}"):
                        try:
                            expense_id = generate_expense_from_recurring(user_id, due_expense['id'])
                            st.success(f"✅ Expense approved and added!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error approving expense: {str(e)}")
                
                with col3:
                    if st.button("✏️ Edit & Approve", key=f"edit_approve_{due_expense['id']}"):
                        st.session_state[f"edit_due_{due_expense['id']}"] = True
                        st.rerun()
                
                # Edit form for due expense
                if st.session_state.get(f"edit_due_{due_expense['id']}", False):
                    with st.form(f"edit_due_form_{due_expense['id']}"):
                        st.markdown("### Edit & Approve Expense")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            edit_amount = st.number_input(
                                "Amount (¥)",
                                value=float(due_expense['amount']),
                                min_value=0.0,
                                step=100.0
                            )
                        with col2:
                            edit_merchant = st.text_input(
                                "Merchant",
                                value=due_expense['merchant']
                            )
                        
                        col3, col4 = st.columns(2)
                        with col3:
                            available_categories = get_available_categories(user_id)
                            try:
                                current_index = available_categories.index(due_expense['category'])
                            except ValueError:
                                current_index = 0
                            
                            edit_category = st.selectbox(
                                "Category",
                                options=available_categories,
                                index=current_index
                            )
                        with col4:
                            edit_payment_method = st.selectbox(
                                "Payment Method",
                                options=["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Other"],
                                index=["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Other"].index(due_expense['payment_method']) if due_expense['payment_method'] in ["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Other"] else 0
                            )
                        
                        edit_description = st.text_area(
                            "Description (optional)",
                            value=due_expense['description'] if due_expense['description'] else ""
                        )
                        
                        col5, col6 = st.columns(2)
                        with col5:
                            if st.form_submit_button("💾 Save & Approve"):
                                try:
                                    override_data = {
                                        "amount": edit_amount,
                                        "merchant": edit_merchant,
                                        "category": edit_category,
                                        "payment_method": edit_payment_method,
                                        "description": edit_description
                                    }
                                    
                                    expense_id = generate_expense_from_recurring(user_id, due_expense['id'], override_data)
                                    del st.session_state[f"edit_due_{due_expense['id']}"]
                                    st.success("✅ Expense updated and approved!")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error updating expense: {str(e)}")
                        
                        with col6:
                            if st.form_submit_button("❌ Cancel"):
                                del st.session_state[f"edit_due_{due_expense['id']}"]
                                st.rerun()
    
    # Main content tabs
    recurring_tabs = st.tabs(["📝 Manage Recurring", "➕ Add New", "💡 Suggestions"])
    
    # Manage Recurring tab
    with recurring_tabs[0]:
        st.markdown("### 📝 Your Recurring Expenses")
        
        if not recurring_expenses:
            st.info("No recurring expenses set up yet. Click 'Add New' to create your first recurring expense!")
        else:
            for recurring in recurring_expenses:
                # Check if this recurring expense is in edit mode
                edit_mode = st.session_state.get(f"edit_recurring_mode_{recurring['id']}", False)
                
                with st.expander(f"{recurring['name']} - ¥{recurring['amount']:,.0f} ({recurring['frequency']})"):
                    if not edit_mode:
                        # Display mode
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        
                        with col1:
                            st.write(f"**Merchant:** {recurring['merchant']}")
                            st.write(f"**Category:** {recurring['category']}")
                            st.write(f"**Amount:** ¥{recurring['amount']:,.0f}")
                            st.write(f"**Frequency:** {recurring['frequency'].title()}")
                            st.write(f"**Next Due:** {recurring['next_due_date']}")
                            st.write(f"**Status:** {'🟢 Active' if recurring['is_active'] else '🔴 Inactive'}")
                            if recurring['averaging_type'] != 'none':
                                st.write(f"**Averaging:** {recurring['averaging_type'].title()}")
                        
                        with col2:
                            if st.button("✏️ Edit", key=f"edit_recurring_{recurring['id']}"):
                                st.session_state[f"edit_recurring_mode_{recurring['id']}"] = True
                                st.rerun()
                        
                        with col3:
                            # Add Generate Now button
                            if st.button("⚡ Generate Now", key=f"generate_now_{recurring['id']}", help="Create an expense from this recurring template"):
                                try:
                                    expense_id = generate_expense_from_recurring(user_id, recurring['id'])
                                    st.success(f"✅ Expense created from {recurring['name']}!")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error generating expense: {str(e)}")
                        
                        with col4:
                            if st.button("🗑️ Delete", key=f"delete_recurring_{recurring['id']}"):
                                try:
                                    delete_recurring_expense(user_id, recurring['id'])
                                    st.success("✅ Recurring expense deleted!")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting recurring expense: {str(e)}")
                    else:
                        # Edit mode - show edit form in place
                        with st.form(f"edit_recurring_form_{recurring['id']}"):
                            st.markdown("### ✏️ Edit Recurring Expense")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                edit_name = st.text_input("Name", value=recurring['name'])
                                edit_merchant = st.text_input("Merchant", value=recurring['merchant'])
                                edit_amount = st.number_input("Amount (¥)", value=float(recurring['amount']), min_value=0.0, step=100.0)
                            with col2:
                                edit_frequency = st.selectbox(
                                    "Frequency",
                                    options=["monthly", "weekly", "quarterly", "yearly"],
                                    index=["monthly", "weekly", "quarterly", "yearly"].index(recurring['frequency']) if recurring['frequency'] in ["monthly", "weekly", "quarterly", "yearly"] else 0,
                                    format_func=lambda x: x.title()
                                )
                                
                                available_categories = get_available_categories(user_id)
                                try:
                                    category_index = available_categories.index(recurring['category'])
                                except ValueError:
                                    category_index = 0
                                edit_category = st.selectbox("Category", options=available_categories, index=category_index)
                                
                                edit_payment_method = st.selectbox(
                                    "Payment Method",
                                    options=["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Other"],
                                    index=["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Other"].index(recurring['payment_method']) if recurring['payment_method'] in ["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Other"] else 0
                                )
                            
                            col3, col4 = st.columns(2)
                            with col3:
                                edit_start_date = st.date_input("Start Date", value=pd.to_datetime(recurring['start_date']).date())
                            with col4:
                                edit_end_date = st.date_input("End Date (Optional)", value=pd.to_datetime(recurring['end_date']).date() if recurring['end_date'] else None)
                            
                            # Averaging option for quarterly/yearly
                            if edit_frequency in ['quarterly', 'yearly']:
                                edit_averaging = st.selectbox(
                                    "Averaging Type",
                                    options=['monthly', 'none'],
                                    index=['monthly', 'none'].index(recurring['averaging_type']) if recurring['averaging_type'] in ['monthly', 'none'] else 0,
                                    format_func=lambda x: 'Average across months (Recommended)' if x == 'monthly' else 'Show as actual payments',
                                    help="Quarterly and yearly expenses are averaged by default for better budgeting"
                                )
                            else:
                                edit_averaging = 'none'
                            
                            edit_description = st.text_area("Description (optional)", value=recurring['description'] if recurring['description'] else "")
                            
                            col5, col6 = st.columns(2)
                            with col5:
                                if st.form_submit_button("💾 Save Changes"):
                                    try:
                                        if not edit_name.strip():
                                            st.error("Name is required")
                                        elif not edit_merchant.strip():
                                            st.error("Merchant is required")
                                        elif edit_amount <= 0:
                                            st.error("Amount must be greater than 0")
                                        else:
                                            recurring_data = {
                                                "name": edit_name.strip(),
                                                "merchant": edit_merchant.strip(),
                                                "amount": edit_amount,
                                                "category": edit_category,
                                                "payment_method": edit_payment_method,
                                                "frequency": edit_frequency,
                                                "start_date": edit_start_date,
                                                "end_date": edit_end_date,
                                                "averaging_type": edit_averaging,
                                                "description": edit_description.strip() if edit_description else None
                                            }
                                            
                                            update_recurring_expense(user_id, recurring['id'], recurring_data)
                                            del st.session_state[f"edit_recurring_mode_{recurring['id']}"]
                                            st.success("✅ Recurring expense updated successfully!")
                                            time.sleep(1)
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Error updating recurring expense: {str(e)}")
                            
                            with col6:
                                if st.form_submit_button("❌ Cancel"):
                                    del st.session_state[f"edit_recurring_mode_{recurring['id']}"]
                                    st.rerun()
    
    # Add New tab
    with recurring_tabs[1]:
        st.markdown("### ➕ Add New Recurring Expense")
        
        # First, get the frequency to determine if averaging options should be shown
        col1, col2 = st.columns(2)
        with col1:
            recurring_name = st.text_input("Name", placeholder="e.g., Monthly Rent", key="recurring_name")
            recurring_merchant = st.text_input("Merchant", placeholder="e.g., Property Management", key="recurring_merchant")
        with col2:
            recurring_amount = st.number_input("Amount (¥)", min_value=0.0, step=100.0, key="recurring_amount")
            recurring_frequency = st.selectbox(
                "Frequency",
                options=["monthly", "weekly", "quarterly", "yearly"],
                format_func=lambda x: x.title(),
                key="recurring_frequency"
            )
        
        col3, col4 = st.columns(2)
        with col3:
            available_categories = get_available_categories(user_id)
            recurring_category = st.selectbox("Category", options=available_categories, key="recurring_category")
        with col4:
            recurring_payment_method = st.selectbox(
                "Payment Method",
                options=["Credit Card", "Debit Card", "Cash", "Digital Wallet", "Other"],
                key="recurring_payment_method"
            )
        
        col5, col6 = st.columns(2)
        with col5:
            recurring_start_date = st.date_input("Start Date", value=date.today(), key="recurring_start_date")
        with col6:
            recurring_end_date = st.date_input("End Date (Optional)", value=None, key="recurring_end_date")
        
        # Averaging option for quarterly/yearly - now reactive!
        if recurring_frequency in ['quarterly', 'yearly']:
            st.markdown("**💡 Averaging (Default for Quarterly/Yearly):**")
            
            # Calculate example amounts
            if recurring_amount > 0:
                monthly_equiv = calculate_monthly_equivalent(recurring_amount, recurring_frequency)
                example_text = f"¥{recurring_amount:,.0f} {recurring_frequency} = ¥{monthly_equiv:,.0f} per month"
            else:
                example_text = "Enter an amount to see the monthly equivalent"
            
            st.info(f"📊 {example_text}")
            
            recurring_averaging = st.selectbox(
                "Averaging Type",
                options=['monthly', 'none'],
                index=0,  # Default to 'monthly' (first option)
                format_func=lambda x: 'Average across months (Recommended)' if x == 'monthly' else 'Show as actual payments',
                help="Quarterly and yearly expenses are averaged by default for better budgeting",
                key="recurring_averaging"
            )
        else:
            recurring_averaging = 'none'
        
        recurring_description = st.text_area("Description (optional)", key="recurring_description")
        
        if st.button("💾 Add Recurring Expense", type="primary", use_container_width=True):
            try:
                if not recurring_name.strip():
                    st.error("Name is required")
                elif not recurring_merchant.strip():
                    st.error("Merchant is required")
                elif recurring_amount <= 0:
                    st.error("Amount must be greater than 0")
                else:
                    # Check for potential duplicates
                    existing_recurring = get_recurring_expenses(user_id)
                    duplicate_found = False
                    
                    for existing in existing_recurring:
                        if (existing['merchant'].lower() == recurring_merchant.strip().lower() and
                            existing['amount'] == recurring_amount and
                            existing['frequency'] == recurring_frequency and
                            existing['is_active']):
                            duplicate_found = True
                            st.warning(f"⚠️ **Potential duplicate found**: A similar recurring expense already exists for {existing['merchant']} (¥{existing['amount']:,.0f} {existing['frequency']})")
                            
                            col_cancel, col_add_anyway = st.columns(2)
                            with col_cancel:
                                if st.button("❌ Cancel", key="cancel_duplicate"):
                                    st.rerun()
                            with col_add_anyway:
                                if st.button("➕ Add Anyway", key="add_anyway", type="secondary"):
                                    duplicate_found = False  # Allow adding
                                    break
                    
                    if not duplicate_found:
                        recurring_data = {
                            "name": recurring_name,
                            "merchant": recurring_merchant,
                            "amount": recurring_amount,
                            "category": recurring_category,
                            "payment_method": recurring_payment_method,
                            "frequency": recurring_frequency,
                            "start_date": recurring_start_date,
                            "end_date": recurring_end_date,
                            "averaging_type": recurring_averaging,
                            "description": recurring_description
                        }
                        
                        recurring_id = add_recurring_expense(user_id, recurring_data)
                        
                        # Auto-generate the first expense for better UX
                        try:
                            first_expense_id = generate_expense_from_recurring(user_id, recurring_id)
                            st.success("✅ Recurring expense added and first payment recorded!")
                            st.info(f"💡 Your {recurring_name} has been added to this month's expenses")
                            
                            # Clear form fields
                            for key in ["recurring_name", "recurring_merchant", "recurring_amount", "recurring_frequency", 
                                       "recurring_category", "recurring_payment_method", "recurring_start_date", 
                                       "recurring_end_date", "recurring_averaging", "recurring_description"]:
                                if key in st.session_state:
                                    del st.session_state[key]
                            
                        except Exception as e:
                            st.warning(f"✅ Recurring expense added, but couldn't auto-generate first expense: {str(e)}")
                        
                        time.sleep(1)
                        st.rerun()
            except Exception as e:
                st.error(f"Error adding recurring expense: {str(e)}")
    
    # Suggestions tab
    with recurring_tabs[2]:
        st.markdown("### 💡 Recurring Expense Suggestions")
        st.info("Based on your spending patterns, we've identified potential recurring expenses:")
        
        if not suggestions:
            st.info("No suggestions available. Add more expenses to see patterns!")
        else:
            for suggestion in suggestions:
                with st.expander(f"{suggestion['merchant']} - ¥{suggestion['amount']:,.0f} ({suggestion['frequency']}) - {suggestion['confidence']:.0%} confidence"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Merchant:** {suggestion['merchant']}")
                        st.write(f"**Amount:** ¥{suggestion['amount']:,.0f}")
                        st.write(f"**Suggested Frequency:** {suggestion['frequency'].title()}")
                        st.write(f"**Category:** {suggestion['category']}")
                        st.write(f"**Occurrences:** {suggestion['occurrences']}")
                        st.write(f"**Confidence:** {suggestion['confidence']:.0%}")
                    
                    with col2:
                        if st.button("➕ Add as Recurring", key=f"add_suggestion_{suggestion['merchant']}_{suggestion['amount']}"):
                            recurring_data = {
                                "name": f"{suggestion['merchant']} - {suggestion['frequency'].title()}",
                                "merchant": suggestion['merchant'],
                                "amount": suggestion['amount'],
                                "category": suggestion['category'],
                                "payment_method": suggestion['payment_method'],
                                "frequency": suggestion['frequency'],
                                "start_date": date.today(),
                                "end_date": None,
                                "averaging_type": 'monthly' if suggestion['frequency'] in ['quarterly', 'yearly'] else 'none',
                                "description": f"Auto-created from pattern analysis"
                            }
                            
                            try:
                                recurring_id = add_recurring_expense(user_id, recurring_data)
                                
                                # Auto-generate the first expense for better UX
                                try:
                                    first_expense_id = generate_expense_from_recurring(user_id, recurring_id)
                                    st.success("✅ Recurring expense added and first payment recorded!")
                                except Exception as e:
                                    st.warning(f"✅ Recurring expense added, but couldn't auto-generate first expense: {str(e)}")
                                
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error adding recurring expense: {str(e)}")

# ---------- Coffee ----------
import streamlit.components.v1 as components  # Add this at the top if not already imported

with tab8:
    st.header("☕️ Coffee")

    st.markdown("If you find this app useful, feel free to buy me a flat white with an extra espresso! 🙏")

    COFFEE_PRICE = 200  # Yen

    def create_checkout_session():
        session = stripe_client.stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency": "jpy",
                    "product_data": {"name": "Buy Me a Coffee"},
                    "unit_amount": COFFEE_PRICE,
                },
                "quantity": 1,
            }],
            mode="payment",
            success_url="https://expense-tracker-cwrs.onrender.com?success=true",
            cancel_url="https://expense-tracker-cwrs.onrender.com?canceled=true",
        )
        return session.url

    if st.button("Buy Me a Coffee (¥200)"):
        url = create_checkout_session()
        st.success("Redirecting you to Stripe Checkout...")

        # Auto open in new tab
        components.html(
            f"""
            <script>
                window.open("{url}", "_blank");
            </script>
            """,
            height=0
        )

        # Fallback for users with popup blockers
        st.markdown(f"[Click here if not redirected]({url})", unsafe_allow_html=True)
