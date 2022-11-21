import logging
import statistics
import sys
from typing import Tuple

import pandas
import seaborn as sns
import random

from leaker.api import RelationalDatabase
from leaker.attack.relational_estimators.estimator import NaruRelationalEstimator, KDERelationalEstimator, \
    NaiveRelationalEstimator, SamplingRelationalEstimator
from leaker.extension import IdentityExtension
from leaker.sql_interface import SQLBackend

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')
console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)
file = logging.FileHandler('eval_dmv_red.log', 'w', 'utf-8')
file.setFormatter(f)
logging.basicConfig(handlers=[console, file], level=logging.DEBUG)
log = logging.getLogger(__name__)

# Import
backend = SQLBackend()
log.info(f"has dbs {backend.data_sets()}")
mimic_db: RelationalDatabase = backend.load("dmv_full")
log.info(
    f"Loaded {mimic_db.name()} data. {len(mimic_db)} documents with {len(mimic_db.keywords())} words. {mimic_db.has_extension(IdentityExtension)}")


def evaluate_actual_rlen(keywords):
    mimic_db.open()
    rlen_list = []
    for q in keywords:
        rlen_list.append(sum(1 for _ in mimic_db(q)))

    median = statistics.median(rlen_list)
    point95 = statistics.quantiles(data=rlen_list, n=100)[94]
    point99 = statistics.quantiles(data=rlen_list, n=100)[98]
    maximum = max(rlen_list)

    mimic_db.close()
    return median, point95, point99, maximum


def evaluate_estimator_rlen(estimator, keywords) -> Tuple[float, float, float, float]:
    mimic_db.open()
    error_value_list = []
    for q in keywords:
        est_value = estimator.estimate(q) + 1
        actual_value = sum(1 for _ in mimic_db(q)) + 1
        current_error = max(est_value, actual_value) / min(est_value, actual_value)  # q-error, naru paper, p. 8
        error_value_list.append(current_error)

    median = statistics.median(error_value_list)
    point95 = statistics.quantiles(data=error_value_list, n=100)[94]
    point99 = statistics.quantiles(data=error_value_list, n=100)[98]
    maximum = max(error_value_list)

    '''
    log.info('MEDIAN: ' + str(median))
    log.info('.95: ' + str(point95))
    log.info('.99: ' + str(point99))
    log.info('MAX: ' + str(maximum))
    '''
    mimic_db.close()
    return median, point95, point99, maximum


def run_rlen_eval(nr_of_evals=1):
    results_list = []
    for _ in range(0, nr_of_evals):
        for known_data_rate in [i / 10 for i in range(2, 11, 2)]:
            sampled_db = mimic_db.sample(known_data_rate)
            kw_sample = random.sample(mimic_db.keywords(), 100)  # nr of queries for evaluation

            # SAMPLING
            sampling_est = SamplingRelationalEstimator(sample=sampled_db, full=mimic_db)
            log.info('SAMPLING')
            results_list.append(
                ('SAMPLING-' + str(known_data_rate),) + evaluate_estimator_rlen(sampling_est, kw_sample))

            # NAIVE
            naive_est = NaiveRelationalEstimator(sample=sampled_db, full=mimic_db)
            log.info('NAIVE')
            results_list.append(('NAIVE-' + str(known_data_rate),) + evaluate_estimator_rlen(naive_est, kw_sample))

            # KDE Estimator
            kde_est = KDERelationalEstimator(sample=sampled_db, full=mimic_db)
            log.info('KDE')
            results_list.append(('KDE-' + str(known_data_rate),) + evaluate_estimator_rlen(kde_est, kw_sample))

            # NARU Estimator
            # naru_est = NaruRelationalEstimator(sample=sampled_db, full=mimic_db)
            # log.info('NARU')
            # df['NARU-' + str(known_data_rate)] = evaluate_estimator_rlen(naru_est, kw_sample)

    df = pandas.DataFrame(data=results_list, columns=['method', 'median', '.95', '.99', 'max'])
    df = df.groupby(['method']).mean()
    df = df.sort_index()
    sns_plot = sns.heatmap(df, annot=True, cmap='RdYlGn_r', fmt=".1f")
    sns_plot.figure.savefig("estimators.png", bbox_inches='tight')


run_rlen_eval(5)
# print(evaluate_actual_rlen(mimic_db.keywords()))
