from typing import Optional
from pydantic import BaseModel

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    race_id: Optional[int] = 202206010609 #適当に選んだサンプルID
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str