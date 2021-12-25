import os, sys
from datetime import datetime
import pickle

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pytest
from model.main import MainData


def test_scrape_pred():
        race_id = 202106040905
        df = MainData.scrape_pred(202106040905)
        assert df.empty == False 


def test_scrape_train():
        test_result = MainData.save_to_db(test_mode=True)
        assert test_result == {'Result': 'Update testing has successfully finished!!'}


def test_db_connect():
        db = MainData.database_connect()
        assert db != None


def test_get_data():
        data = MainData.get_data()
        assert len(data) > 0


# def test_create_model():
#         assert MainData.create_model() != None


# def test_prediction():
#         race_id = 202106040905
#         pred = MainData.predict(race_id)
#         assert len(pred) > 1
