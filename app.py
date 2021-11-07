from fastapi import FastAPI, BackgroundTasks
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
