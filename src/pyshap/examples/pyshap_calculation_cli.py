import operator
import os

import time
import warnings

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import functions as F, SparkSession, types as T, Window


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


def select_row(df, row_id):
    """
    Mark row as selected
    """
    return df.withColumn(
        'is_selected',
        F.when(F.col('id') == row_id, F.lit(True)).otherwise(F.lit(False))
    )


def get_features_permutations(
        df,
        feature_names,
        output_col='features_permutations'
):
    """
    Creates a column for the ordered features and then shuffles it.
    The result is a dataframe with a column `output_col` that contains:
    [feat2, feat4, feat3, feat1],
    [feat3, feat4, feat2, feat1],
    [feat1, feat2, feat4, feat3],
    ...
    """
    return df.withColumn(
        output_col,
        F.shuffle(
            F.array(*[F.lit(f) for f in feature_names])
        )
    )


def calculate_shapley_values(
        df,
        model,
        row_of_interest,
        feature_names,
        features_col='features',
        column_to_examine='anomalyScore'
):
    """
    # Based on the algorithm described here:
    # https://christophm.github.io/interpretable-ml-book/shapley.html#estimating-the-shapley-value
    # And on Baskerville's implementation for IForest/ AnomalyModel here:
    # https://github.com/equalitie/baskerville/blob/develop/src/baskerville/util/model_interpretation/helpers.py#L235
    """
    results = {}
    features_perm_col = 'features_permutations'
    spark = get_spark_session()
    marginal_contribution_filter = F.avg('marginal_contribution').alias(
        'shap_value'
    )
    # broadcast the row of interest and ordered feature names
    ROW_OF_INTEREST_BROADCAST = spark.sparkContext.broadcast(
        row_of_interest[features_col]
    )
    ORDERED_FEATURE_NAMES = spark.sparkContext.broadcast(feature_names)

    # persist before continuing with calculations
    if not df.is_cached:
        df = df.persist()

    # get permutations
    features_df = get_features_permutations(
        df,
        feature_names,
        output_col=features_perm_col
    )

    # set up the udf - x-j and x+j need to be calculated for every row
    def calculate_x(
            feature_j, z_features, curr_feature_perm
    ):
        """
        The instance  x+j is the instance of interest,
        but all values in the order before feature j are
        replaced by feature values from the sample z
        The instance  x−j is the same as  x+j, but in addition
        has feature j replaced by the value for feature j from the sample z
        """
        x_interest = ROW_OF_INTEREST_BROADCAST.value
        ordered_features = ORDERED_FEATURE_NAMES.value
        x_minus_j = list(z_features).copy()
        x_plus_j = list(z_features).copy()
        f_i = curr_feature_perm.index(feature_j)
        after_j = False
        for f in curr_feature_perm[f_i:]:
            # replace z feature values with x of interest feature values
            # iterate features in current permutation until one before j
            # x-j = [z1, z2, ... zj-1, xj, xj+1, ..., xN]
            # we already have zs because we go row by row with the udf,
            # so replace z_features with x of interest
            f_index = ordered_features.index(f)
            new_value = x_interest[f_index]
            x_plus_j[f_index] = new_value
            if after_j:
                x_minus_j[f_index] = new_value
            after_j = True

        # minus must be first because of lag
        return Vectors.dense(x_minus_j), Vectors.dense(x_plus_j)

    udf_calculate_x = F.udf(calculate_x, T.ArrayType(VectorUDT()))

    # persist before processing
    features_df = features_df.persist()

    for f in feature_names:
        # x column contains x-j and x+j in this order.
        # Because lag is calculated this way:
        # F.col('anomalyScore') - (F.col('anomalyScore') one row before)
        # x-j needs to be first in `x` column array so we should have:
        # id1, [x-j row i,  x+j row i]
        # ...
        # that with explode becomes:
        # id1, x-j row i
        # id1, x+j row i
        # ...
        # to give us (x+j - x-j) when we calculate marginal contribution
        # Note that with explode, x-j and x+j for the same row have the same id
        # This gives us the opportunity to use lag with
        # a window partitioned by id
        x_df = features_df.withColumn('x', udf_calculate_x(
            F.lit(f), features_col, features_perm_col
        )).persist()
        print(f'Calculating SHAP values for "{f}"...')
        x_df.show(10, False)
        x_df = x_df.selectExpr(
            'id', f'explode(x) as {features_col}'
        ).cache()
        print('Exploded df:')
        x_df.show(10, False)
        x_df = model.transform(x_df)

        # marginal contribution is calculated using a window and a lag of 1.
        # the window is partitioned by id because x+j and x-j for the same row
        # will have the same id
        x_df = x_df.withColumn(
            'marginal_contribution',
            (
                    F.col(column_to_examine) - F.lag(
                        F.col(column_to_examine), 1
                    ).over(Window.partitionBy('id').orderBy('id')
            )
            )
        )
        x_df.show(5, False)
        # calculate the average
        x_df = x_df.filter(
            x_df.marginal_contribution.isNotNull()
        )
        tdf = x_df.select(
            marginal_contribution_filter
        )
        tdf.show()
        results[f] = x_df.select(
            marginal_contribution_filter
        ).first().shap_value
        x_df.unpersist()
        del x_df
        print(f'Marginal Contribution for feature: {f} = {results[f]} ')

    ordered_results = sorted(
        results.items(),
        key=operator.itemgetter(1),
        reverse=True
    )
    return ordered_results


def shapley_values_for_model(
        model_path,
        feature_names,
        row_id,
        data_path=None,
        column_to_examine=None
):
    from baskerville.spark import get_spark_session
    _ = get_spark_session()
    metadata = get_model_metadata(model_path)
    model = load_model(model_path, metadata)
    features_col = metadata['paramMap'].get('featuresCol', 'features')
    # get sample dataset
    df = load_df_from_json(data_path)
    if 'id' not in df.columns:
        # not a good idea
        warnings.warn('Missing column "id", using monotonically_increasing_id')
        df = df.withColumn('id', F.monotonically_increasing_id())
    df.show(10, False)

    # predict on the data if the df does not contain the results
    if column_to_examine not in df.columns:
        pred_df = model.predict(df)
        df = pred_df.select('id', features_col, column_to_examine)

    # select the row to be examined
    df = select_row(df, row_id)
    row_of_interest = df.select('id', 'features').where(
        F.col('is_selected') == True  # noqa
    ).first()
    print('Row: ', row_of_interest)

    # remove x and drop column_to_examine column
    df = df.select('*').where(
        F.col('is_selected') == False  # noqa
    ).drop(column_to_examine)

    shap_values = calculate_shapley_values(
        df,
        model,
        row_of_interest,
        feature_names,
        features_col=features_col,
        column_to_examine=column_to_examine,
    )
    print('Feature ranking by Shapley values:')
    print('-' * 20)
    print(*[f'#{i}. {feat} = {value}' for i, (feat, value) in
            enumerate(shap_values)])
    return shap_values


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'modelpath', help='The path to the model to be examined'
    )
    parser.add_argument(
        'feature_names',
        help='The feature names. E.g. feature1,feature2,feature3',
        nargs='+', default=[]
    )

    parser.add_argument(
        'datapath', help='The path to the data'
    )
    parser.add_argument(
        'id',
        help='The row id to be examined',
        type=str
    )
    parser.add_argument(
        '-c', '--column',
        help='Column to examine, e.g. prediction. '
             'Default is "anomalyScore"',
        default='anomalyScore',
        type=str
    )
    args = parser.parse_args()
    model_path = args.modelpath
    feature_names = args.feature_names
    data_path = args.datapath
    column_to_examine = args.column
    row_id = args.id

    if not os.path.isdir(model_path):
        raise ValueError('Model path does not exist')
    if not os.path.isdir(data_path):
        raise ValueError('Data path does not exist')

    start = time.time()
    print('Start:', start)
    shapley_values_for_model(
        model_path,
        feature_names,
        row_id,
        data_path,
        column_to_examine
    )
    end = time.time()
    print('End:', end)
    print(f'Calculation took {int(end - start)} seconds')