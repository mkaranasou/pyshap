import abc
import warnings

from pyspark.sql import functions as F

from pyshap.helpers import set_selected_row, get_row_of_interest
from pyshap.spark import get_spark_session


class ShapCalculatorBase(metaclass=abc.ABCMeta):
    def __init__(self, config, spark, df=None):
        self.config = config
        self.spark = spark
        self.df = df
        self.model = None
        self.feature_names = []
        self.features_col = 'features'
        self.row_of_interest = None
        self.marginal_contribution_filter = None
        self.output_col = self.config.column_to_examine
        self.row_id_col = self.config.col_of_interest
        self.row_of_interest_filter = None
        self.col_to_examine = self.config.column_to_examine
        self.cols_to_select = (self.row_id_col, self.features_col)

    def initialize(self):
        if not self.spark:
            self.spark = get_spark_session()
        self.marginal_contribution_filter = F.avg('marginal_contribution').alias(
            'shap_value'
        )
        self.load_df()
        self.load_model()
        self.get_row_of_interest_filter()
        self.set_up_df()

    def get_row_of_interest_filter(self):
        # yikes...
        self.row_of_interest_filter = eval(self.config.row_of_interest_selector)

    def set_up_df(self):
        # if self.config.col_of_interest not in self.df.columns:
        #     # not a good idea
        #     warnings.warn(f'Missing column "{self.config.col_of_interest}", using monotonically_increasing_id')
        #     self.df = self.df.withColumn(self.config.col_of_interest, F.monotonically_increasing_id())
        # self.df.show(10, False)

        # predict on the data if the df does not contain the results
        if self.col_to_examine not in self.df.columns:
            self.df = self.model.predict(self.df).select(
                self.config.col_of_interest,
                self.features_col,
                self.config.column_to_examine
            ).cache()

        self.get_row_of_interest()
        self.get_df_without_row_of_interest()

    def get_row_of_interest(self):
        # select the row to be examined
        self.df = set_selected_row(self.df, self.row_of_interest_filter, col=self.row_id_col)
        self.row_of_interest = get_row_of_interest(self.df, cols_to_select=self.cols_to_select)
        print('Row: ', self.row_of_interest)

    def get_df_without_row_of_interest(self):
        # remove x and drop column_to_examine column
        self.df = self.df.select('*').where(
            F.col('is_selected') == False  # noqa
        ).drop(self.config.column_to_examine)


    @abc.abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load_df(self):
        raise NotImplementedError

    @abc.abstractmethod
    def calculate(self):
        raise NotImplementedError