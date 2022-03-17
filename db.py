from sqlmodel import Session, create_engine
import os
# database_url = os.environ.get('DATABASE_URL').replace("postgres://","postgresql://")
# database_url = os.getenv('DATABASE_URL','sqlite:///data.db?check_same_thread=False').replace("postgres://","postgresql://")
database_url = 'postgresql://ztgsuxglqxgqaf:87f6b30fe550286ea4d05cde3c4ff3408dbc2d0c1562119dbab6abd773535e05@ec2-54-236-137-173.compute-1.amazonaws.com:5432/dkk1q1rsknfg7'
engine = create_engine(database_url,echo=False)
