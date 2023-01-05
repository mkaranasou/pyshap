import os
import time

from pyaml_env import parse_config

from pyshap import SRC_DIR
from pyshap.calculation import ShapleySparkCalculator
from pyshap.spark import get_spark_session

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--conf", action="store", dest="conf_file",
        default=os.path.join(SRC_DIR, '..', 'conf', 'conf.yaml'),
        help="Path to config file, defaults to conf/conf.yaml"
    )

    args = parser.parse_args()
    conf_file = args.conf_file

    if not os.path.isfile(conf_file):
        raise ValueError('Config file does not exist')

    start = time.time()
    print('Start:', start)
    config = parse_config(args.conf_file)
    spark = get_spark_session()
    calculator = ShapleySparkCalculator(config, spark)
    results = calculator.calculate()
    end = time.time()
    print('End:', end)
    print(f'Calculation took {int(end - start)} seconds')
    print(results)