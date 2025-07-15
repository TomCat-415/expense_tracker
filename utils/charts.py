"""
Chart generation utilities for the expense tracker.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import calendar
from datetime import datetime, date, timedelta
import numpy as np
import json
from pathlib import Path
import logging

from config.settings import DEFAULT_CATEGORIES, CURRENCY_SYMBOL, BUDGET_LIMITS
from .database import get_expenses_df, get_user_settings

logger = logging.getLogger(__name__)


def create_category_pie_chart(category_totals: pd.DataFrame, title: str = "Spending by Category", user_id: str = None) -> go.Figure:
    """Create a pie chart for category spending breakdown."""
    # Get user's category colors
    category_colors = get_user_category_colors(user_id)
    
    fig = px.pie(
        category_totals,
        values='amount',
        names='category',
        title=title,
        color='category',
        color_discrete_map=category_colors
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Amount: ' + CURRENCY_SYMBOL + '%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        showlegend=True,
        height=400,
        font=dict(size=12)
    )
    
    return fig


def create_daily_spending_chart(daily_spending: pd.DataFrame, title: str = "Daily Spending Trend") -> go.Figure:
    """Create a line chart showing daily spending trends."""
    # Calculate y-axis range with 20% padding
    max_amount = daily_spending['amount'].max()
    y_max = max_amount * 1.2  # Add 20% padding to the top
    
    fig = px.line(
        daily_spending,
        x='date',
        y='amount',
        title=title,
        markers=True,
        line_shape='linear'  # Use straight lines between points
    )
    
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>Amount: ' + CURRENCY_SYMBOL + '%{y:,.0f}<extra></extra>',
        line=dict(color='#45B7D1', width=2),
        marker=dict(size=8, color='#FF6B6B')
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=f"Amount ({CURRENCY_SYMBOL})",
        height=400,
        hovermode='x unified',
        yaxis=dict(
            range=[0, y_max],  # Set y-axis range from 0 to max + padding
            tickformat=',d'  # Format large numbers with commas
        )
    )
    
    return fig


def create_monthly_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Create a chart comparing monthly spending across years."""
    if df.empty:
        return go.Figure()

    # Prepare data
    df['Year'] = df['date'].dt.year
    df['Month'] = df['date'].dt.month
    monthly_by_year = df.pivot_table(
        values='amount',
        index='Month',
        columns='Year',
        aggfunc='sum'
    ).fillna(0)

    # Create figure
    fig = go.Figure()
    
    # Add line for each year
    for year in monthly_by_year.columns:
        fig.add_trace(go.Scatter(
            x=monthly_by_year.index,
            y=monthly_by_year[year],
            name=str(year),
            mode='lines+markers'
        ))

    fig.update_layout(
        title="Monthly Spending by Year",
        xaxis_title="Month",
        yaxis_title="Total Spending (¥)",
        hovermode='x unified',
        showlegend=True
    )

    return fig


def create_category_trend_chart(expenses_df: pd.DataFrame, categories: List[str] = None, user_id: str = None) -> go.Figure:
    """Create a multi-line chart showing spending trends by category over time."""
    if categories is None:
        categories = expenses_df['category'].unique()
    
    # Get user's category colors
    category_colors = get_user_category_colors(user_id)
    
    # Group by month and category
    expenses_df['month'] = expenses_df['date'].dt.to_period('M')
    monthly_category = expenses_df.groupby(['month', 'category'])['amount'].sum().reset_index()
    monthly_category['month'] = monthly_category['month'].astype(str)
    
    fig = go.Figure()
    
    for category in categories:
        category_data = monthly_category[monthly_category['category'] == category]
        
        fig.add_trace(go.Scatter(
            x=category_data['month'],
            y=category_data['amount'],
            mode='lines+markers',
            name=category,
            line=dict(color=category_colors.get(category, '#747D8C')),
            hovertemplate=f'<b>{category}</b><br>%{{x}}<br>Amount: ' + CURRENCY_SYMBOL + '%{y:,.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Category Spending Trends",
        xaxis_title="Month",
        yaxis_title=f"Amount ({CURRENCY_SYMBOL})",
        height=500,
        hovermode='x unified'
    )
    
    return fig


def create_budget_vs_actual_chart(budget_data: Dict, actual_data: Dict) -> go.Figure:
    """Create a comparison chart of budget vs actual spending."""
    categories = list(set(budget_data.keys()) | set(actual_data.keys()))
    
    budget_amounts = [budget_data.get(cat, 0) for cat in categories]
    actual_amounts = [actual_data.get(cat, 0) for cat in categories]
    
    fig = go.Figure(data=[
        go.Bar(name='Budget', x=categories, y=budget_amounts, marker_color='#96CEB4'),
        go.Bar(name='Actual', x=categories, y=actual_amounts, marker_color='#FF6B6B')
    ])
    
    fig.update_layout(
        title="Budget vs Actual Spending",
        xaxis_title="Category",
        yaxis_title=f"Amount ({CURRENCY_SYMBOL})",
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    
    return fig


def create_monthly_average_chart(expenses_df: pd.DataFrame) -> go.Figure:
    """Create a combination chart showing monthly total and average daily spending."""
    if expenses_df.empty:
        return go.Figure()

    # Add month column and group by month
    expenses_df['month'] = expenses_df['date'].dt.to_period('M')
    
    # Calculate days in each month
    expenses_df['days_in_month'] = expenses_df['date'].dt.days_in_month
    
    # Group by month
    monthly_stats = expenses_df.groupby('month').agg({
        'amount': ['sum', 'count'],
        'days_in_month': 'first'  # Take the first value since it's the same for the whole month
    }).reset_index()
    
    # Calculate daily average
    monthly_stats['daily_average'] = monthly_stats[('amount', 'sum')] / monthly_stats[('days_in_month', 'first')]
    
    # Convert month to string for better display
    monthly_stats['month'] = monthly_stats['month'].astype(str)
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add monthly total bars
    fig.add_trace(
        go.Bar(
            x=monthly_stats['month'],
            y=monthly_stats[('amount', 'sum')],
            name="Monthly Total",
            marker_color='#4ECDC4',
            hovertemplate='<b>%{x}</b><br>' +
            'Total: ' + CURRENCY_SYMBOL + '%{y:,.2f}<br>' +
            'Transactions: %{text}<extra></extra>',
            text=monthly_stats[('amount', 'count')]
        ),
        secondary_y=False
    )
    
    # Add daily average line
    fig.add_trace(
        go.Scatter(
            x=monthly_stats['month'],
            y=monthly_stats['daily_average'],
            name="Daily Average",
            line=dict(color='#FF6B6B', width=3),
            mode='lines+markers',
            hovertemplate='<b>%{x}</b><br>' +
            'Daily Avg: ' + CURRENCY_SYMBOL + '%{y:,.2f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title="Monthly Spending Overview",
        height=400,
        hovermode='x unified',
        barmode='relative',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes
    fig.update_yaxes(
        title_text=f"Monthly Total ({CURRENCY_SYMBOL})", 
        secondary_y=False
    )
    fig.update_yaxes(
        title_text=f"Daily Average ({CURRENCY_SYMBOL})", 
        secondary_y=True
    )
    fig.update_xaxes(title_text="Month")
    
    return fig


def create_payment_method_chart(expenses_df: pd.DataFrame) -> go.Figure:
    """Create a donut chart showing payment method usage."""
    payment_totals = expenses_df.groupby('payment_method')['amount'].sum().reset_index()
    
    fig = px.pie(
        payment_totals,
        values='amount',
        names='payment_method',
        title="Spending by Payment Method",
        hole=0.4
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Amount: ' + CURRENCY_SYMBOL + '%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(height=400)
    
    return fig


def create_top_merchants_chart(expenses_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Create a bar chart showing top merchants by spending."""
    if expenses_df.empty:
        return go.Figure()

    # Get top merchants
    top_merchants = expenses_df.groupby('merchant').agg({
        'amount': 'sum',
        'id': 'count'
    }).reset_index()
    
    top_merchants = top_merchants.nlargest(top_n, 'amount')
    top_merchants = top_merchants.sort_values('amount', ascending=True)  # For bottom-to-top display

    # Generate a color palette
    colors = px.colors.qualitative.Set3[:top_n]  # Using Plotly's Set3 palette for distinct colors
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_merchants['amount'],
        y=top_merchants['merchant'],
        orientation='h',
        marker_color=colors,
        text=top_merchants['amount'].apply(lambda x: f'¥{x:,.0f}'),
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>' +
                      'Total: ¥%{x:,.0f}<br>' +
                      '<extra></extra>'
    ))

    fig.update_layout(
        title=f"Top {top_n} Merchants by Spending",
        xaxis_title="Total Spending (¥)",
        yaxis_title="Merchant",
        height=max(400, 50 * top_n),  # Dynamic height based on number of merchants
        showlegend=False,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=False)
    )

    return fig


def create_weekly_spending_heatmap(expenses_df: pd.DataFrame) -> go.Figure:
    """Create a heatmap showing spending patterns by day of week and hour."""
    # Add day of week and hour columns
    expenses_df['day_of_week'] = expenses_df['date'].dt.day_name()
    expenses_df['week_number'] = expenses_df['date'].dt.isocalendar().week
    
    # Group by week and day of week
    weekly_spending = expenses_df.groupby(['week_number', 'day_of_week'])['amount'].sum().reset_index()
    
    # Pivot for heatmap
    heatmap_data = weekly_spending.pivot(index='week_number', columns='day_of_week', values='amount').fillna(0)
    
    # Reorder columns to start with Monday
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(columns=day_order, fill_value=0)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Blues',
        hovertemplate='<b>Week %{y}</b><br>%{x}<br>Amount: ' + CURRENCY_SYMBOL + '%{z:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Weekly Spending Heatmap",
        xaxis_title="Day of Week",
        yaxis_title="Week Number",
        height=400
    )
    
    return fig


def create_budget_tracking_chart(df: pd.DataFrame) -> go.Figure:
    """Create a chart showing spending vs budget by category."""
    if df.empty:
        return go.Figure()

    # Get current month's spending by category
    current_month = datetime.now().replace(day=1)
    month_mask = (df['date'] >= current_month)
    monthly_by_category = df[month_mask].groupby('category')['amount'].sum()

    # Create figure
    fig = go.Figure()

    # Add bars for actual spending and budget
    categories = list(BUDGET_LIMITS.keys())
    actual_spending = [monthly_by_category.get(cat, 0) for cat in categories]
    budget_limits = [BUDGET_LIMITS[cat] for cat in categories]

    # Calculate percentage of budget used
    budget_used = [
        (spent / budget * 100) if budget > 0 else 0 
        for spent, budget in zip(actual_spending, budget_limits)
    ]

    # Add bars
    fig.add_trace(go.Bar(
        name='Actual Spending',
        x=categories,
        y=actual_spending,
        marker_color=['red' if spent > budget else 'green' 
                     for spent, budget in zip(actual_spending, budget_limits)]
    ))

    fig.add_trace(go.Bar(
        name='Budget Limit',
        x=categories,
        y=budget_limits,
        marker_color='rgba(0, 0, 0, 0.2)'
    ))

    # Add percentage labels
    for i, (cat, spent, budget, pct) in enumerate(zip(categories, actual_spending, budget_limits, budget_used)):
        fig.add_annotation(
            x=cat,
            y=max(spent, budget),
            text=f"{pct:.1f}%",
            showarrow=False,
            yshift=10
        )

    fig.update_layout(
        title="Monthly Budget Tracking",
        barmode='overlay',
        yaxis_title="Amount (¥)",
        hovermode='x unified'
    )

    return fig


def create_spending_forecast(df: pd.DataFrame, days_ahead: int = 30) -> Tuple[go.Figure, float]:
    """Create a spending forecast chart and return predicted total."""
    if df.empty:
        return go.Figure(), 0

    # Prepare daily spending data
    df['date'] = pd.to_datetime(df['date'])
    daily_spending = df.groupby('date')['amount'].sum().reset_index()
    
    # Calculate moving averages and baseline metrics
    daily_spending['MA7'] = daily_spending['amount'].rolling(window=7, min_periods=1).mean()
    daily_spending['MA30'] = daily_spending['amount'].rolling(window=30, min_periods=1).mean()
    
    # Calculate recent average daily spending (last 7 days)
    recent_daily_avg = daily_spending['amount'].tail(7).mean()
    if pd.isna(recent_daily_avg) or recent_daily_avg == 0:
        recent_daily_avg = daily_spending['amount'].mean()  # Fallback to overall average
    
    # Generate future dates
    last_date = daily_spending['date'].max()
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=days_ahead
    )

    # Calculate volatility from recent data
    volatility = daily_spending['amount'].tail(30).std() * 0.3  # Reduced volatility impact
    if pd.isna(volatility):
        volatility = daily_spending['amount'].std() * 0.3
    
    # Create forecast array based on recent daily average
    base_forecast = np.full(days_ahead, recent_daily_avg)
    
    # Add controlled randomness
    np.random.seed(42)  # For consistent results
    random_factors = np.random.normal(0, volatility, size=days_ahead)
    future_y = np.maximum(0, base_forecast + random_factors)  # Ensure no negative values

    # Create figure
    fig = go.Figure()

    # Add actual spending
    fig.add_trace(go.Scatter(
        x=daily_spending['date'],
        y=daily_spending['amount'],
        name='Daily Spending',
        mode='markers',
        marker=dict(size=6, color='#45B7D1')
    ))

    # Add moving averages
    fig.add_trace(go.Scatter(
        x=daily_spending['date'],
        y=daily_spending['MA7'],
        name='7-day Moving Average',
        line=dict(width=2, color='#FF6B6B')
    ))

    fig.add_trace(go.Scatter(
        x=daily_spending['date'],
        y=daily_spending['MA30'],
        name='30-day Moving Average',
        line=dict(width=2, color='#4ECDC4')
    ))

    # Add forecast
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_y,
        name='Forecast',
        line=dict(dash='dash', color='#FFB900')
    ))

    # Add confidence interval
    upper_bound = future_y + volatility
    lower_bound = np.maximum(0, future_y - volatility)

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=upper_bound,
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=lower_bound,
        name='Lower Bound',
        fill='tonexty',
        fillcolor='rgba(255, 185, 0, 0.2)',
        line=dict(width=0),
        showlegend=False
    ))

    # Calculate y-axis range with padding
    y_max = max(
        daily_spending['amount'].max(),
        upper_bound.max()
    ) * 1.2  # Add 20% padding

    # Update layout
    fig.update_layout(
        title="Spending Forecast",
        xaxis_title="Date",
        yaxis_title="Daily Spending (¥)",
        hovermode='x unified',
        yaxis=dict(
            range=[0, y_max],
            tickformat=',d'  # Format large numbers with commas
        ),
        # Improve legend
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Calculate predicted total using the base forecast (without volatility)
    # This gives a more stable prediction
    predicted_total = float(recent_daily_avg * days_ahead)

    return fig, predicted_total


def create_weekly_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Create a week-over-week spending comparison chart."""
    if df.empty:
        return go.Figure()

    # Prepare weekly data
    df['Week'] = df['date'].dt.isocalendar().week
    df['Year'] = df['date'].dt.isocalendar().year
    
    weekly_spending = df.groupby(['Year', 'Week'])['amount'].agg([
        ('total', 'sum'),
        ('count', 'count'),
        ('avg', 'mean')
    ]).reset_index()

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add total spending bars
    fig.add_trace(
        go.Bar(
            x=weekly_spending.apply(lambda x: f"{x['Year']}-W{int(x['Week']):02d}", axis=1),
            y=weekly_spending['total'],
            name="Total Spending",
            marker_color='rgb(55, 83, 109)'
        ),
        secondary_y=False
    )

    # Add transaction count line
    fig.add_trace(
        go.Scatter(
            x=weekly_spending.apply(lambda x: f"{x['Year']}-W{int(x['Week']):02d}", axis=1),
            y=weekly_spending['count'],
            name="Transaction Count",
            mode='lines+markers',
            marker_color='rgb(26, 118, 255)'
        ),
        secondary_y=True
    )

    # Update layout
    fig.update_layout(
        title="Week-over-Week Spending Analysis",
        xaxis_title="Week",
        barmode='group',
        hovermode='x unified'
    )

    fig.update_yaxes(
        title_text="Total Spending (¥)",
        secondary_y=False
    )
    fig.update_yaxes(
        title_text="Transaction Count",
        secondary_y=True
    )

    return fig


def create_category_budget_allocation(df: pd.DataFrame) -> go.Figure:
    """Create a chart showing budget allocation and actual spending by category."""
    if df.empty:
        return go.Figure()

    # Get current month's data
    current_month = datetime.now().replace(day=1)
    month_mask = (df['date'] >= current_month)
    monthly_by_category = df[month_mask].groupby('category').agg({
        'amount': ['sum', 'count', 'mean']
    }).reset_index()
    monthly_by_category.columns = ['category', 'total', 'count', 'avg']

    # Calculate budget percentages
    total_budget = sum(BUDGET_LIMITS.values())
    budget_percentages = {
        cat: (limit / total_budget * 100) 
        for cat, limit in BUDGET_LIMITS.items()
    }

    # Calculate actual percentages
    total_spending = monthly_by_category['total'].sum()
    actual_percentages = {
        row['category']: (row['total'] / total_spending * 100)
        for _, row in monthly_by_category.iterrows()
    }

    # Create figure
    fig = go.Figure()

    # Add budget allocation
    fig.add_trace(go.Bar(
        name='Budget Allocation',
        x=list(BUDGET_LIMITS.keys()),
        y=[budget_percentages.get(cat, 0) for cat in BUDGET_LIMITS.keys()],
        marker_color='rgba(0, 0, 0, 0.2)'
    ))

    # Add actual spending
    fig.add_trace(go.Bar(
        name='Actual Spending',
        x=list(BUDGET_LIMITS.keys()),
        y=[actual_percentages.get(cat, 0) for cat in BUDGET_LIMITS.keys()],
        marker_color='rgba(55, 83, 109, 0.7)'
    ))

    fig.update_layout(
        title="Category Budget Allocation vs Actual Spending",
        xaxis_title="Category",
        yaxis_title="Percentage of Total (%)",
        barmode='group',
        hovermode='x unified'
    )

    return fig


def create_spending_by_category_pie(expenses: List[Dict], start_date: Optional[str] = None, end_date: Optional[str] = None, user_id: str = None) -> go.Figure:
    """Create a pie chart showing spending distribution by category."""
    df = pd.DataFrame(expenses)
    
    # Apply date filters if provided
    if start_date:
        df = df[df['date'] >= start_date]
    if end_date:
        df = df[df['date'] <= end_date]
    
    # Group by category and sum amounts
    category_totals = df.groupby('category')['amount'].sum().reset_index()
    
    # Get user's category colors
    color_map = get_user_category_colors(user_id)
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=category_totals['category'],
        values=category_totals['amount'],
        marker_colors=[color_map.get(cat, '#747D8C') for cat in category_totals['category']],
        textinfo='label+percent',
        hovertemplate="<b>%{label}</b><br>" +
                      "Amount: ¥%{value:,.0f}<br>" +
                      "<extra></extra>"
    )])
    
    fig.update_layout(
        title="Spending by Category",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_spending_trend(expenses: List[Dict], category_filter: Optional[str] = None) -> go.Figure:
    """Create a line chart showing spending trends over time with optional category filtering."""
    df = pd.DataFrame(expenses)
    
    # Apply category filter if provided
    if category_filter:
        df = df[df['category'] == category_filter]
    
    # Group by date and sum amounts
    daily_totals = df.groupby('date')['amount'].sum().reset_index()
    
    # Sort by date
    daily_totals = daily_totals.sort_values('date')
    
    # Calculate cumulative sum
    daily_totals['cumulative'] = daily_totals['amount'].cumsum()
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add daily spending bars
    fig.add_trace(go.Bar(
        x=daily_totals['date'],
        y=daily_totals['amount'],
        name='Daily Spending',
        hovertemplate="Date: %{x}<br>" +
                      "Amount: ¥%{y:,.0f}<br>" +
                      "<extra></extra>"
    ))
    
    # Add cumulative line
    fig.add_trace(go.Scatter(
        x=daily_totals['date'],
        y=daily_totals['cumulative'],
        name='Cumulative',
        yaxis='y2',
        line=dict(color='red'),
        hovertemplate="Date: %{x}<br>" +
                      "Cumulative: ¥%{y:,.0f}<br>" +
                      "<extra></extra>"
    ))
    
    title = "Spending Trend"
    if category_filter:
        title += f" - {category_filter}"
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Daily Amount (¥)",
        yaxis2=dict(
            title="Cumulative Amount (¥)",
            overlaying='y',
            side='right'
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        barmode='stack'
    )
    
    return fig


def get_available_categories(user_id: str = None) -> List[str]:
    """Get list of available expense categories."""
    if user_id is not None:
        try:
            settings = get_user_settings(user_id)
            categories = settings.get("custom_categories", DEFAULT_CATEGORIES)
            return [cat for cat in categories.keys() if cat != "_metadata"]
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
    
    # Fallback to default categories
    with open(Path(__file__).parent.parent / 'config' / 'categories.json', 'r') as f:
        categories = json.load(f)
    return [cat for cat in categories.keys() if cat != "_metadata"]


def get_user_category_colors(user_id: str = None) -> Dict[str, str]:
    """Get user's category colors mapping."""
    if user_id is not None:
        try:
            settings = get_user_settings(user_id)
            categories = settings.get("custom_categories", DEFAULT_CATEGORIES)
            # Return color mapping, filtering out _metadata
            return {cat: color for cat, color in categories.items() if cat != "_metadata"}
        except Exception as e:
            logger.error(f"Error getting category colors: {e}")
    
    # Fallback to default categories
    with open(Path(__file__).parent.parent / 'config' / 'categories.json', 'r') as f:
        categories = json.load(f)
    return {cat: color for cat, color in categories.items() if cat != "_metadata"}
    