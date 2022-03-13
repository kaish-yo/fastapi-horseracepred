from sqlmodel import Session, create_engine
import os
DATABASE_URI = os.environ.get('DATABASE_URL','sqlite:///data.db?check_same_thread=False').replace("postgres://","postgresql://")
engine = create_engine(DATABASE_URI,echo=False)
