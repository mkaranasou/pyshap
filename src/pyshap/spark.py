from pyspark import SparkConf
from pyspark.sql import SparkSession

from pyshap.helpers import SPARK_DRIVER_MEMORY, CPU_COUNT


def get_spark_session():
    """
    With an effort to optimize memory and partitions
    """
    conf = SparkConf()
    print(SPARK_DRIVER_MEMORY)
    conf.set('spark.driver.memory', SPARK_DRIVER_MEMORY)
    conf.set('spark.sql.shuffle.partitions', str(CPU_COUNT))
    return SparkSession \
        .builder \
        .config(conf=conf) \
        .appName("IForest feature importance") \
        .getOrCreate()
