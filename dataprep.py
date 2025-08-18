
import pandas as pd
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.extras import execute_values
import re

# Connect to a PostgreSQL database
def connect_to_postgres(user: str, password: str,db_name: str, host: str = "localhost", port: int = 5432 ) -> psycopg2.extensions.connection:
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
            try:
                conn = psycopg2.connect(
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
    
def normalize_question_marks(text: str) -> str:
    """
    Normalize question marks in a question:
    - Remove any internal question marks
    - Remove extra spaces before question mark
    - Keep exactly one question mark at the end
    """
    text = text.strip()
    # Remove all question marks inside the text
    text = re.sub(r'\?+', '', text)
    # Remove trailing spaces
    text = text.rstrip()
    # Add single question mark at the end
    return text + '?'

# Clean and normalize data
def clean_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean the medical Q&A dataset
    """
    df = pd.read_csv(file_path)

    # Drop missing values
    df = df.dropna()

    # Strip spaces and normalize text
    df['question'] = df['question'].str.strip().str.lower()
    df['answer'] = df['answer'].str.strip()

    # remove extra spaces, newlines, non-ASCII chars and normalize question marks
    df['question'] = df['question'].apply(lambda x: re.sub(r'\s+', ' ', x))
    df['answer'] = df['answer'].apply(lambda x: re.sub(r'\s+', ' ', x))
    df['question'] = df['question'].apply(lambda x: normalize_question_marks(x))
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Group by question and combine answers
    df = df.groupby('question', as_index=False).agg({'answer': lambda x: ' '.join(x)})
    print(df.count())
        
    return df

# Create Postgres database
def create_database(db_name: str, conn):
    """
    Create a PostgreSQL database if it doesn't exist
    """
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{db_name}'")
    exists = cur.fetchone()
    if not exists:
        cur.execute(f"CREATE DATABASE {db_name}")
        print(f"Database '{db_name}' created successfully!")
    else:
        print(f"Database '{db_name}' already exists.")

# Create Postgres table
def create_table(conn, table_name: str):
    """
    Create a table for storing Q&A in Postgres
    """
    cur = conn.cursor()

    # Create table 
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL
        );
    """)
    conn.commit()
    print(f"Table '{table_name}' is ready.")

# Function 4: Load data into Postgres
def load_data(df: pd.DataFrame, table_name: str, conn):
    """
    Load cleaned Q&A data into Postgres table
    """
    cur = conn.cursor()

    # Prepare data as list of tuples
    data = list(df[['question', 'answer']].itertuples(index=False, name=None))

    # Batch insert
    insert_query = f"INSERT INTO {table_name} (question, answer) VALUES %s"
    execute_values(cur, insert_query, data)

    conn.commit()
    cur.close()
    conn.close()
    print(f"Inserted {len(data)} records into '{table_name}'.")

if __name__ == "__main__":
    file_path = "mle_screening_dataset.csv"
    db_name = "chatbot_db"
    table_name = "medical_qa"
    user = "postgres"
    password = "Nitish@23"

    df_clean = clean_data(file_path)
    # print(df_clean.head())
    # print(df_clean.count())
    try:
        conn = connect_to_postgres(user, password, db_name)
        print("Connected to Postgres successfully.")
    except Exception as e:
        print(f"Failed to connect to Postgres: {e}")
    
    create_database(db_name, conn)
    create_table(conn, table_name)
    load_data(df_clean, table_name, conn)
    print("Data loading process completed successfully.")
