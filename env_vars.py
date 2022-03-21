import os

database_url = os.getenv('DATABASE_URL','sqlite:///data.db?check_same_thread=False').replace("postgres://","postgresql://")
blob_conn_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')