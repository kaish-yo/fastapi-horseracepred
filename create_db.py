from sqlmodel import SQLModel
from model.main import MainData
from db import engine

print('Creating database....')

SQLModel.metadata.create_all(engine)