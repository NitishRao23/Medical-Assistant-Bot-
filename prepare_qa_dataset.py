import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split

def connect_to_postgres(user: str, password: str,db_name: str, host: str = "localhost", port: int = 5432 ) -> psycopg2.extensions.connection:
    """
    Connect to a PostgreSQL database
    """
    
    try:
            conn = psycopg2.connect(
                dbname=db_name,
                user=user,
                password=password,
                host=host,
                port=port
            )
            print(f"Connected to database '{db_name}' successfully!")
            return conn
    except Exception as e:
            print(f"Error connecting to database: {e}")
            return None

# --- 1. Connect to your database and load data
def load_data_from_postgres(conn, table_name):
    query = f"SELECT question, answer FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# --- 2. Format the dataset for training (no changes needed if question-answer only)
def format_for_qa(df):
    # If your model needs a prefix (e.g., for T5), add it
    df['input'] = "question: " + df['question'].astype(str)
    df['output'] = df['answer'].astype(str)
    return df[['input', 'output']]

# --- 3. Split dataset into train/val/test
def split_dataset(df, train_size=0.8, val_size=0.1, test_size=0.1, seed=42):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-5, "Splits must add up to 1"

    train_df, temp_df = train_test_split(df, test_size=(1 - train_size), random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=test_size / (test_size + val_size), random_state=seed)

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

# --- 4. Save splits to CSV (optional)
def save_splits(train_df, val_df, test_df):
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)
    print("Datasets saved: train.csv, val.csv, test.csv")

# --- MAIN: Run everything
if __name__ == "__main__":
    user = "postgres"
    password = "Nitish@23"
    db_name = "chatbot_db"
    table_name = "medical_qa"

    print("🔄 Loading data from database...")
    conn = connect_to_postgres(user, password, db_name)
    df = load_data_from_postgres(conn, table_name)

    print(f"✅ Loaded {len(df)} rows")

    print("🧹 Formatting data...")
    qa_df = format_for_qa(df)

    print("✂️ Splitting data...")
    train_df, val_df, test_df = split_dataset(qa_df)

    print(f"✅ Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    save_splits(train_df, val_df, test_df)
