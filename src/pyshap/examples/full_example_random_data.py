import random

import numpy as np
import pyspark
from shapley_spark_calculation import \
    calculate_shapley_values, select_row
from pyspark.ml.classification import RandomForestClassifier, LinearSVC, \
    DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession, types as T, functions as F


def get_spark_session():
    import os
    from psutil import virtual_memory
    from pyspark import SparkConf

    mem = virtual_memory()
    conf = SparkConf()
    memory = f'{int(round((mem.total / 2) / 1024 / 1024 / 1024, 0))}G'
    print(memory)
    conf.set('spark.driver.memory', memory)
    conf.set('spark.sql.shuffle.partitions', str(os.cpu_count()*2))
    return SparkSession \
        .builder \
        .config(conf=conf) \
        .appName("IForest feature importance") \
        .getOrCreate()


def get_features_schema(feature_names):
    return T.StructType([
        T.StructField(
            name=feature,
            dataType=T.FloatType(),
            nullable=True) for feature in feature_names
    ])


def generate_df_for_features(feature_names, rows=5000):
    data = np.zeros([rows, len(feature_names)])
    spark = get_spark_session()
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


if __name__ == '__main__':
    feature_names = [f'f{i}' for i in range(6)]
    spark = get_spark_session()
    df = generate_df_for_features(feature_names, 5000)
    assembler = VectorAssembler(
        inputCols=feature_names,
        outputCol="features")
    df = assembler.transform(df)
    df.show(10, False)
    df = add_labels(df)
    (trainingData, testData) = df.randomSplit([0.8, 0.2])

    estimator = DecisionTreeClassifier(
        labelCol="label", featuresCol="features",
    )
    # estimator = RandomForestClassifier(
    #     labelCol="label", featuresCol="features", numTrees=10
    # )
    model = estimator.fit(trainingData)
    predictions = model.transform(testData)
    column_to_examine = 'prediction'
    predictions.select(column_to_examine, "label", "features").show(5)

    evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol=column_to_examine
    )
    accuracy = evaluator.evaluate(predictions)
    print(f'Test Error = %{(1.0 - accuracy)}')

    testData = select_row(testData, testData.select('id').take(1)[0].id)
    row_of_interest = testData.select('id', 'features').where(
        F.col('is_selected') == True  # noqa
    ).first()
    print('Row: ', row_of_interest)
    testData = testData.select('*').where(F.col('is_selected') != True)
    df = df.drop(
        column_to_examine
    )
    shap_values = calculate_shapley_values(
        testData,
        model,
        row_of_interest,
        feature_names,
        features_col='features',
        column_to_examine='prediction',
    )
    print('Feature ranking by Shapley values:')
    print('-' * 20)


    top_three = []
    last_three = []
    for i, (feat, value) in enumerate(shap_values):
        print(f'#{i}. {feat} = {value}')
        if i < 3:
            top_three.append(feat)
        if i >= 3:
            last_three.append(feat)
    # re-test wiht top 3 features
    trainingData_top_3 = trainingData.select('id', *top_three, 'label')
    testData_top_3= trainingData.select('id', *top_three, 'label')
    assembler_new = VectorAssembler(
        inputCols=top_three,
        outputCol="features")
    trainingData_top_3 = assembler_new.transform(trainingData_top_3)
    testData_top_3 = assembler_new.transform(testData_top_3)
    model = estimator.fit(trainingData_top_3)
    predictions_top_3 = model.transform(testData_top_3)
    accuracy_top_3 = evaluator.evaluate(predictions_top_3)
    print(f'Test Error = %{(1.0 - accuracy_top_3)}')

    print('Top3: Improved by', accuracy_top_3 - accuracy)
    assert accuracy_top_3 > accuracy

    # re-test wiht last 3 features
    trainingData_last_3 = trainingData.select('id', *last_three, 'label')
    testData_last_3 = trainingData.select('id', *last_three, 'label')
    assembler_new = VectorAssembler(
        inputCols=last_three,
        outputCol="features")
    trainingData_last_3 = assembler_new.transform(trainingData_last_3)
    testData_last_3 = assembler_new.transform(testData_last_3)
    model = estimator.fit(trainingData_last_3)
    predictions_last_3 = model.transform(testData_last_3)
    accuracy_last_3 = evaluator.evaluate(predictions_last_3)
    print(f'Test Error = %{(1.0 - accuracy_last_3)}')

    print('Last 3: Worse by', accuracy_last_3 - accuracy_top_3)
    assert accuracy_last_3 < accuracy_top_3