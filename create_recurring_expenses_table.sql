-- Create recurring_expenses table
CREATE TABLE IF NOT EXISTS recurring_expenses (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL,
    name TEXT NOT NULL,
    merchant TEXT NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    category TEXT NOT NULL,
    description TEXT,
    payment_method TEXT NOT NULL,
    frequency TEXT NOT NULL CHECK (frequency IN ('weekly', 'monthly', 'quarterly', 'yearly')),
    start_date DATE NOT NULL,
    end_date DATE,
    next_due_date DATE NOT NULL,
    last_generated_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    averaging_type TEXT DEFAULT 'none' CHECK (averaging_type IN ('none', 'monthly')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_recurring_expenses_user_id ON recurring_expenses(user_id);
CREATE INDEX IF NOT EXISTS idx_recurring_expenses_next_due_date ON recurring_expenses(next_due_date);
CREATE INDEX IF NOT EXISTS idx_recurring_expenses_is_active ON recurring_expenses(is_active);

-- Add foreign key constraint if auth.users table exists
ALTER TABLE recurring_expenses ADD CONSTRAINT fk_recurring_expenses_user_id 
FOREIGN KEY (user_id) REFERENCES auth.users(id) ON DELETE CASCADE;

-- Add RLS (Row Level Security) policies
ALTER TABLE recurring_expenses ENABLE ROW LEVEL SECURITY;

-- Policy to allow users to see only their own recurring expenses
CREATE POLICY "Users can view their own recurring expenses" ON recurring_expenses
    FOR SELECT USING (auth.uid() = user_id);

-- Policy to allow users to insert their own recurring expenses
CREATE POLICY "Users can insert their own recurring expenses" ON recurring_expenses
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Policy to allow users to update their own recurring expenses
CREATE POLICY "Users can update their own recurring expenses" ON recurring_expenses
    FOR UPDATE USING (auth.uid() = user_id);

-- Policy to allow users to delete their own recurring expenses
CREATE POLICY "Users can delete their own recurring expenses" ON recurring_expenses
    FOR DELETE USING (auth.uid() = user_id);

-- Add trigger to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_recurring_expenses_updated_at BEFORE UPDATE
    ON recurring_expenses FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Add recurring_id column to expenses table to link generated expenses
ALTER TABLE expenses ADD COLUMN IF NOT EXISTS recurring_id UUID;
CREATE INDEX IF NOT EXISTS idx_expenses_recurring_id ON expenses(recurring_id);

-- Add foreign key constraint for recurring_id
ALTER TABLE expenses ADD CONSTRAINT fk_expenses_recurring_id 
FOREIGN KEY (recurring_id) REFERENCES recurring_expenses(id) ON DELETE SET NULL; 