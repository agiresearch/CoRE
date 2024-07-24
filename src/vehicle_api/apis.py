import pandas as pd
from pandas import DataFrame
from typing import Optional
from utils.flow_utils import extract_before_parenthesis
import json

class VehicleInfo:

    def __init__(self, path="../vehicle_data/info.json"):
        self.path = path
        self.data = None

        with open(self.path, 'r') as file:
            self.data = json.load(file)


    def load_task_idx(self, task_idx):
        self.task_idx = task_idx

    def run(self) -> str:
        info = self.data[self.task_idx]
        res = f"The weather is {info['weather']}, the time of day is {info['timeofday']}, the scene is {info['scene']}, the longitude is {info['longitude']}, the latitude is {info['latitude']}, "
        if info['permission to share']:
            res += 'apps have permission to use information.'
        else:
            res += 'apps do no have permission to use any information.'

        return res