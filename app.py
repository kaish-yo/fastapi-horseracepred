from fastapi import FastAPI
import uvicorn
import os
from model.main import MainData

app = FastAPI()

@app.get("/") #ホーム画面
def home():
    return {"message":"Welcome to Horse race prediction app!"}

@app.get("/prediction/{race_id}") #モデルを用いて予測を実行する
def pred(race_id: int):
    pred = MainData.predict(race_id)
    return pred

@app.post("/data/update") #Firebase上のデータベースをアップデートする
def update_db():
    return MainData.save_to_db()

@app.post("/model/update") #学習モデルをアップデートする
def update_model():  
    _, cm, cr = MainData.create_model()
    print(cm,cr)
    return {"Success":"Model updated"}

if __name__ == '__main__':
    port = int(os.environ.get('PORT',8000))
    uvicorn.run(app, host="0.0.0.0",port=port,debug=True)
