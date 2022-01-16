import uvicorn
import os
from fastapi import Depends, FastAPI, BackgroundTasks, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
import json
from datetime import datetime, timedelta
from typing import Optional
from model.main import MainData
from user.user import Token, TokenData, User, UserInDB

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.get("/") #ホーム画面
def home():
    return {"message":"Welcome to Horse race prediction app!"}


'''#####################Security Purpose#####################'''
# to get a string like this run:
# openssl rand -hex 32
with open("./user/security.json") as f: #keyなどはjsonファイルに格納
    d = json.load(f)
    SECRET_KEY = d["secret_key"]
    users = d["user_data"] #user_db

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)


def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(users, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user
'''##########################################################'''


##予測結果についてはログインしていないと取得できないようにする
@app.get("/prediction") #モデルを用いて予測を実行する
def pred(req: User = Depends(get_current_active_user)):
    race_id = req.race_id
    pred = MainData.predict(race_id)
    return pred


@app.post("/data/update") #Firebase上のデータベースをアップデートする #非同期処理
async def update_db(background_tasks:BackgroundTasks):
    background_tasks.add_task(MainData.save_to_db)
    # return MainData.save_to_db()
    return {"Message":"Update of the database has started!"}


@app.post("/model/update") #学習モデルをアップデートする
async def update_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(MainData.create_model)
    return {"Success":"Model update has started in the background!"}


if __name__ == '__main__':
    port = int(os.environ.get('PORT',8000))
    uvicorn.run(app, host="0.0.0.0",port=port,debug=True)
