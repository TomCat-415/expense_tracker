# 💰 Expensei - Your Expense Sensei

A sophisticated, multi-user expense tracking application built with **Streamlit** and **Supabase**. Features intelligent OCR receipt scanning, recurring expense management with averaging, real-time analytics, and secure multi-user authentication.

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://expense-tracker-cwrs.onrender.com)
[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Powered by Supabase](https://img.shields.io/badge/Powered%20by-Supabase-3ECF8E?style=flat-square&logo=supabase)](https://supabase.com)

## 🚀 Live Demo
**[Try the app here](https://expense-tracker-cwrs.onrender.com)** - Sign up with any email to test all features!

## ✨ Key Features

### 🔐 **Multi-User Authentication**
- **Supabase Auth** with email/password authentication
- **Row Level Security (RLS)** ensuring users only see their own data
- Secure session management with refresh tokens

### 📸 **Smart Receipt Scanning**
- **OCR Processing** using Tesseract and Google Cloud Vision
- **Confidence scoring** for extracted data (date, merchant, amount)
- **Multi-language support** (English, Japanese, Chinese)
- Automatic receipt categorization and processing

### 🔄 **Recurring Expenses & Averaging**
- **Intelligent recurring expense management** (weekly, monthly, quarterly, yearly)
- **Expense averaging system** - converts large periodic payments into smooth monthly budgets
- **Automatic expense generation** from recurring templates
- **Smart suggestions** based on spending patterns

### 📊 **Rich Analytics & Visualization**
- **Interactive dashboards** with 12+ chart types using Plotly
- **Budget tracking** with customizable category limits
- **Spending forecasting** using machine learning
- **Dual view modes**: Actual payments vs. Averaged budgeting
- **Weekly/Monthly comparisons** and trend analysis

### 🛠️ **Advanced Data Management**
- **CSV import/export** with duplicate detection
- **Automated backups** with restore functionality
- **Smart categorization** with ML-based merchant matching
- **Custom categories** with visual color coding

### 💳 **Payment Integration**
- **Stripe integration** for coffee donations (test & live API)
- Secure checkout with automatic redirect handling
- Ready for subscription payment implementation

## 🏗️ Technical Architecture

### **Backend & Database**
- **Supabase** - PostgreSQL database with real-time capabilities
- **Row Level Security** - User data isolation at database level
- **Efficient caching** with LRU cache for performance
- **Comprehensive error handling** with custom exceptions

### **Frontend & UI**
- **Streamlit** - Modern, responsive web interface
- **Custom CSS** with light teal theme [[memory:3328597]]
- **Component-based architecture** with reusable utilities
- **Mobile-responsive design** with optimized layouts

### **Data Processing**
- **OCR Pipeline**: Tesseract + Google Cloud Vision
- **Smart categorization**: Fuzzy matching + keyword analysis
- **Averaging algorithms**: Complex financial calculations
- **ML-based suggestions**: Pattern recognition for recurring expenses

### **Security & Performance**
- **Environment-based configuration** for API keys
- **Input validation** and sanitization
- **Database connection pooling** and query optimization
- **Secure file handling** for receipt uploads

## 🔧 Installation & Setup

### Prerequisites
```bash
Python 3.8+
Tesseract OCR
```

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/expense-tracker.git
cd expense-tracker
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file with:
```env
# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key

# Stripe
STRIPE_SECRET_KEY=your_stripe_secret_key
STRIPE_PUBLISHABLE_KEY=your_stripe_publishable_key

# Optional OCR APIs
OPENAI_API_KEY=your_openai_key
GOOGLE_CLOUD_VISION_KEY=your_gcp_key
```

### 4. Database Setup
Run the provided SQL scripts to set up tables:
```sql
-- See create_recurring_expenses_table.sql
-- Includes RLS policies and indexes
```

### 5. Launch Application
```bash
streamlit run app.py
```

## 🎯 Usage Guide

### **Getting Started**
1. **Sign up** with any email address
2. **Add your first expense** manually or via receipt scan
3. **Set up recurring expenses** for regular payments
4. **Enable averaging** for large periodic expenses
5. **Explore analytics** across different time periods

### **Pro Tips**
- 📱 **Mobile-friendly**: Works great on phones for quick expense entry
- 🔄 **Recurring setup**: Set up rent, insurance, and subscriptions for automated tracking
- 📊 **View modes**: Switch between "Actual" and "Averaged" views for better budgeting
- 🎯 **Categories**: Customize categories to match your spending patterns

## 🛠️ Technical Highlights

### **Database Design**
```sql
-- Multi-tenant architecture with RLS
CREATE POLICY "Users can view their own expenses" ON expenses
  FOR SELECT USING (auth.uid() = user_id);

-- Optimized indexes for performance
CREATE INDEX idx_expenses_user_date ON expenses(user_id, date DESC);
```

### **Smart Categorization Algorithm**
```python
def categorize_expense(self, merchant: str, amount: float = None) -> Tuple[str, float]:
    # 1. Exact store matching (Japanese + English)
    # 2. Fuzzy string matching with confidence scoring
    # 3. Keyword analysis with NLP
    # 4. Fallback to default category
```

### **Averaging System**
```python
def calculate_monthly_equivalent(amount: float, frequency: str) -> float:
    # Converts periodic payments to monthly budget amounts
    # Handles quarterly (÷3) and yearly (÷12) frequencies
```

## 📈 Performance Optimizations

- **Database connection pooling** for concurrent users
- **LRU caching** for frequently accessed data
- **Lazy loading** of charts and analytics
- **Efficient pagination** for large datasets
- **Optimized SQL queries** with proper indexing

## 🔐 Security Features

- **Row Level Security** - Database-level user isolation
- **Input sanitization** - Prevents injection attacks
- **Secure file uploads** - Validates file types and sizes
- **Session management** - Proper token handling
- **Environment variables** - Secure API key management

## 🌟 Future Enhancements

- [ ] **Subscription payments** via Stripe
- [ ] **Mobile app** with React Native
- [ ] **Bank API integration** for automatic transaction import
- [ ] **Advanced AI** for better expense categorization
- [ ] **Multi-currency support**
- [ ] **Shared family accounts**

## 💻 Development

### **Tech Stack**
- **Backend**: Python, Supabase (PostgreSQL)
- **Frontend**: Streamlit, Plotly, Custom CSS
- **OCR**: Tesseract, Google Cloud Vision
- **Payments**: Stripe API
- **Deployment**: Render, Streamlit Cloud

### **Key Libraries**
```python
streamlit>=1.28.0      # Web framework
supabase>=2.16.0       # Database & auth
plotly>=5.17.0         # Interactive charts
pandas>=2.1.0          # Data manipulation
stripe>=10.1.0         # Payment processing
pytesseract>=0.3.10    # OCR processing
```

### **Architecture Patterns**
- **Repository pattern** for database operations
- **Service layer** for business logic
- **Component-based UI** with reusable modules
- **Configuration management** with environment variables

## 📊 Analytics Examples

The app generates sophisticated analytics including:
- **Monthly spending trends** with forecasting
- **Category breakdowns** with budget comparisons
- **Merchant analysis** and frequency patterns
- **Payment method distributions**
- **Weekly spending heatmaps**

## 💡 Why This Project?

This expense tracker demonstrates:
- **Full-stack development** skills
- **Database design** and optimization
- **API integration** (Stripe, OCR services)
- **User experience** design
- **Security best practices**
- **Performance optimization**
- **Modern deployment** strategies

## 🤝 Contributing

Feel free to open issues or submit pull requests. This project showcases modern Python web development with real-world applications.

## 📄 License

MIT License - feel free to use this code for learning or your own projects!

---

**Built with ❤️ by TomCat415** | [LinkedIn](https://linkedin.com/in/yourprofile) | [Portfolio](https://yourwebsite.com)

*If you find this useful, consider [buying me a coffee](https://expense-tracker-cwrs.onrender.com) through the app! ☕*
