import pandas as pd

def analyze_csv(file_path):
    print(f"Analyzing {file_path}...")
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Basic information
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # First few rows
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Data types
        print("\nData types:")
        print(df.dtypes)
        
        # Check for missing values
        print("\nMissing values:")
        print(df.isnull().sum())
        
        # Basic statistics for numeric columns
        print("\nBasic statistics:")
        print(df.describe())
        
        return df
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

if __name__ == "__main__":
    # Analyze the movies dataset
    movies_df = analyze_csv("movies.csv") 