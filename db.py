from sqlmodel import Session, create_engine

engine = create_engine('sqlite:///data.db',echo=False)
