import os, sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pytest
from models.main import MainData
from app import app
from db import db

db.init_app(app)

def test_prediction():
    with app.app_context():
        race_id = 202106040905
        MainData.create_model()
        pred = MainData.predict(race_id)
        assert len(pred) > 1
