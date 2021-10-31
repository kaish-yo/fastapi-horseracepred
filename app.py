from fastapi import FastAPI
import uvicorn
import os
from model.main import MainData

app = FastAPI()

@app.get("/")
def home():
    return {"message":"Welcome to Horse race prediction app!"}

@app.get("/prediction/{race_id}")
def pred(race_id: int):
    pred = MainData.predict(race_id)
    return pred

@app.post("/model/update")
def update_model():
    _, cm, cr = MainData.create_model()
    print(cm,cr)
    return {"Success":"Model updated"}

if __name__ == '__main__':
    port = int(os.environ.get('PORT',8000))
    uvicorn.run(app, host="0.0.0.0",port=port,debug=True)
