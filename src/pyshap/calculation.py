import operator
import warnings

from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.ml.util import MLReader
from pyspark.sql import functions as F, Window, types as T

from pyshap.base import ShapCalculatorBase
from pyshap.helpers import get_model_metadata, load_model, set_selected_row, get_row_of_interest, \
    generate_df_for_features, test_df, test_model
from pyshap.spark import get_spark_session


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


class ShapleySparkCalculator(ShapCalculatorBase):

    def __init__(self, config, spark, df=None):
        super().__init__(config, spark, df)
        self.metadata = None
        self.model_reader = None
        self.initialize()

    def load_df(self):
        #
        if self.df == None:
            if self.config.test_data:
                self.feature_names = [str(i) for i in range(10)]
                self.df = test_df(self.feature_names)
            else:
                self.df = self.spark.read.load(
                    self.config.feature_vectors_dir, format=self.config.format
                ).cache()
        # # persist before continuing with calculations
        # if not self.df.is_cached:
        #     self.df = self.df.persist()

    def load_model(self):

        if self.config.test_data:
            self.model, self.df = test_model(self.df)
        else:
            # this needs to be flexible...might be a pipeline?
            self.metadata = get_model_metadata(self.config.model_dir)
            self.model = load_model(self.config.model_dir, self.metadata)
            self.features_col = self.metadata['paramMap'].get('featuresCol', 'features')
            # self.model_reader = MLReader()
            # self.model_reader.session(self.spark)
            # self.model = self.model_reader.load(self.config.model_dir)

    def calculate(self):
        """
        # Based on the algorithm described here:
        # https://christophm.github.io/interpretable-ml-book/shapley.html#estimating-the-shapley-value
        # And on Baskerville's implementation for IForest/ AnomalyModel here:
        # https://github.com/equalitie/baskerville/blob/develop/src/baskerville/util/model_interpretation/helpers.py#L235
        """
        results = {}
        features_perm_col = 'features_permutations'
        # broadcast the row of interest and ordered feature names
        ROW_OF_INTEREST_BROADCAST = self.spark.sparkContext.broadcast(self.row_of_interest)
        ORDERED_FEATURE_NAMES = self.spark.sparkContext.broadcast(self.feature_names)

        # get permutations
        features_df = get_features_permutations(
            self.df,
            self.feature_names,
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

        for f in self.feature_names:
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
                F.lit(f), self.features_col, features_perm_col
            )).persist()
            print(f'Calculating SHAP values for "{f}"...')

            x_df = x_df.selectExpr(
                'id', f'explode(x) as {self.features_col}'
            ).cache()
            x_df = self.model.transform(x_df)

            # marginal contribution is calculated using a window and a lag of 1.
            # the window is partitioned by id because x+j and x-j for the same row
            # will have the same id
            x_df = x_df.withColumn(
                'marginal_contribution',
                (
                        F.col(self.col_to_examine) - F.lag(
                            F.col(self.col_to_examine), 1
                        ).over(Window.partitionBy('id').orderBy('id')
                )
                )
            )
            # calculate the average
            x_df = x_df.filter(
                x_df.marginal_contribution.isNotNull()
            )
            results[f] = x_df.select(
                self.marginal_contribution_filter
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