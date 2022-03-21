from sqlmodel import Session, create_engine
from env_vars import database_url
import os

engine = create_engine(database_url,echo=False)
