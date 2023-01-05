import operator
import os
from psutil import virtual_memory
from pyspark import SparkConf
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import functions as F, SparkSession, types as T, Window


def get_spark_session():
    """
    With an effort to optimize memory and partitions
    """
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
    ROW_OF_INTEREST_BROADCAST = spark.sparkContext.broadcast(row_of_interest)
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

        x_df = x_df.selectExpr(
            'id', f'explode(x) as {features_col}'
        ).cache()
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
        # calculate the average
        x_df = x_df.filter(
            x_df.marginal_contribution.isNotNull()
        )
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