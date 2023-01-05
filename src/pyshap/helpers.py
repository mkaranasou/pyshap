import os
import random

import pyspark
from psutil import virtual_memory
import numpy as np
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import functions as F, SparkSession, types as T, Window

mem = virtual_memory()
MEMORY_GB = int(round((mem.total / 2) / 1024 / 1024 / 1024, 0))
SPARK_DRIVER_MEMORY = f'{MEMORY_GB}G'
CPU_COUNT = os.cpu_count() * 2


def class_from_str(str_class):
    """
    Imports the module that contains the class requested and returns
    the class
    :param str_class: the full path to class, e.g. pyspark.sql.DataFrame
    :return: the class requested
    """
    import importlib

    split_str = str_class.split('.')
    class_name = split_str[-1]
    module_name = '.'.join(split_str[:-1])
    module = importlib.import_module(module_name)

    return getattr(module, class_name)


def get_model_metadata(model_path):
    """
    Loads model metadata from saved model path
    """
    import json
    with open(os.path.join(model_path, 'metadata', 'part-00000')) as mp:
        return json.load(mp)


def load_model(model_path, metadata):
    """
    Returns the model instance based on the metadata class. It is assumed that
    the spark session has been instantiated with the proper spark.jars
    """
    return class_from_str(
        metadata['class']
    ).load(model_path)


def load_df_with_expanded_features(
        df_path, feature_names, features_col='features'
):
    spark = get_spark_session()
    df = spark.read.json(df_path).cache()
    if features_col in df.columns:
        df = expand_features(df, feature_names, features_col)
    return df


def load_df_from_json(df_path):
    """
    Load the spark dataframe - json format
    """
    return get_spark_session().read.json(df_path).cache()


def expand_features(df, feature_names, features_col='features'):
    """
    Return a dataframe with as many extra columns as the feature_names
    """
    for feature in feature_names:
        column = f'{features_col}.{feature}'
        df = df.withColumn(
            column, F.col(column).cast('double').alias(feature)
        )
    return df

def set_selected_row(df, row_filter, col='id'):
    """
    Mark row as selected
    """
    if not isinstance(row_filter, (F.column,)):
        row_filter = F.col(col) == row_filter
    return df.withColumn(
        'is_selected',
        F.when(row_filter, F.lit(True)).otherwise(F.lit(False))
    )

def get_row_of_interest(df, row_value=True, col='is_selected', cols_to_select=('id', 'features')):
    return df.select(*cols_to_select).where(
        F.col(col) == row_value  # noqa
    ).first()


def get_features_schema(feature_names):
    return T.StructType([
        T.StructField(
            name=feature,
            dataType=T.FloatType(),
            nullable=True) for feature in feature_names
    ])


def generate_df_for_features(spark, feature_names, rows=5000):
    data = np.zeros([rows, len(feature_names)])
    df = spark.createDataFrame(
        data.tolist(),
        schema=get_features_schema(feature_names)
    )
    df = df.rdd.zipWithIndex().toDF()
    df = df.withColumnRenamed('_2', 'id')
    for feat in feature_names:
        df = df.withColumn(feat, F.col(f'_1.{feat}')).withColumn(
            feat, F.round(F.rand(random.randint(0, 10)), 3)
        )
    df = df.drop('_1')
    df.show(10, False)
    return df


def add_labels(df: pyspark.sql.DataFrame, labels=(0, 1), split=(.8, .2)):
    if len(labels) > 2:
        raise NotImplementedError
    df_first_label = df.sample(
        fraction=split[0], seed=42
    ).withColumn('label', F.lit(labels[0]))

    return df.join(
        df_first_label.select('id', 'label').alias('df_first_label'),
        on=['id'],
        how='left'
    ).fillna(labels[1], subset=['label'])


def test_df(feature_names, n=5000):
    df = generate_df_for_features(feature_names, n)
    assembler = VectorAssembler(
        inputCols=feature_names,
        outputCol="features")
    df = assembler.transform(df)
    df.show(10, False)
    df = add_labels(df)
    # (trainingData, testData) = df.randomSplit([0.8, 0.2])
    #

    return df

def test_model(df):
    training_df, test_df = df.randomSplit([0.8, 0.2])
    estimator = DecisionTreeClassifier(
        labelCol="label", featuresCol="features",
    )
    model = estimator.fit(training_df)
    return model, model.transform(test_df)
