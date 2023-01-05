import abc
from enum import Enum


class ModelTypeEnum(Enum):
    scikit = 0
    pyspark = 0


class Model(metaclass=abc.ABCMeta):
    def __init__(self, config):
        self.config = config
        self.model_type = None

    @abc.abstractmethod
    def load(self):
        pass

    @abc.abstractmethod
    def transform(self, df):
        pass


class PySparkModel(Model):
    def load(self):
        pass

    def transform(self, df):
        pass