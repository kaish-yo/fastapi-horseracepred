from sqlmodel import Session, create_engine
import os
database_uri = os.environ.get('DATABASE_URL','sqlite:///data.db?check_same_thread=False').replace("postgres://","postgresql://")
engine = create_engine(database_uri,echo=False)
