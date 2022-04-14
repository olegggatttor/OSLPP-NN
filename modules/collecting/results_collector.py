import pandas as pd


class Collector:
    def collect(self, logs: dict):
        pass

    def get_result(self):
        pass


class DataFrameCollector(Collector):
    def __init__(self, schema: dict):
        self.results = pd.DataFrame(schema)

    def collect(self, logs: dict):
        self.results = self.results.append(logs, ignore_index=True)

    def get_result(self):
        return self.results
