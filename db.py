from sqlmodel import Session, create_engine

engine = create_engine('sqlite:///data.db?check_same_thread=False',echo=False)
