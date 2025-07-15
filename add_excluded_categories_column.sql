-- Add excluded_dashboard_categories column to user_settings table
-- This column will store an array of category names that should be excluded from the dashboard

ALTER TABLE user_settings ADD COLUMN IF NOT EXISTS excluded_dashboard_categories JSONB DEFAULT '[]'::jsonb;

-- Create an index for better performance when querying excluded categories
CREATE INDEX IF NOT EXISTS idx_user_settings_excluded_categories ON user_settings USING GIN (excluded_dashboard_categories);

-- Add a comment to document the column
COMMENT ON COLUMN user_settings.excluded_dashboard_categories IS 'Array of category names to exclude from dashboard view'; 