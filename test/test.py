import os, sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pytest
from model.main import MainData

def test_prediction():
        race_id = 202106040905
        MainData.create_model()
        pred = MainData.predict(race_id)
        assert len(pred) > 1
