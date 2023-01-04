import logging
import statistics
import sys
from random import sample

import numpy as np
from typing import Tuple, List, Union

import pandas
import seaborn as sns
import matplotlib.pyplot as plt

from leaker.api import RelationalDatabase, RelationalQuery, Selectivity
from leaker.attack.relational_estimators.estimator import NaruRelationalEstimator, KDERelationalEstimator, \
    SamplingRelationalEstimator, RelationalEstimator
from leaker.extension import IdentityExtension, CoOccurrenceExtension
from leaker.sql_interface import SQLBackend

# SPECIFY TO BE USED DATASET HERE
dataset = 'dmv_10k'

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')
console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)
file = logging.FileHandler('eval_' + dataset + '.log', 'w', 'utf-8')
file.setFormatter(f)
logging.basicConfig(handlers=[console, file], level=logging.DEBUG)
log = logging.getLogger(__name__)

# Import
backend = SQLBackend()
log.info(f"has dbs {backend.data_sets()}")
mimic_db: RelationalDatabase = backend.load(dataset)
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


def ErrorMetric(est_card, card):
    est_card = round(est_card)  # especially KDE returns floats
    if card == 0 and est_card != 0:
        return est_card
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)


def evaluate_estimator(estimator: RelationalEstimator, keywords: Union[List[RelationalQuery], List[Tuple[RelationalQuery, RelationalQuery]]],
                       use_cooc=False, ignore_zero_one=False) -> Tuple[float, float, float, float]:
    """
    Run list of (co-occ) queries on estimator and calculate statistic properties.

    Other Parameters
    ----------------
    estimator : RelationalEstimator
        actual estimator
    keywords : Union[List[RelationalQuery], List[Tuple[RelationalQuery, RelationalQuery]]]
        list or queries or list of query tuples (in case of cooc)
    use_cooc : bool
        calculate selectivity based on two keywords
    ignore_zero_one : bool
        ignore cases where estimation is zero and true selectivity is one
    """
    error_value_list = []
    error_value_list_zero_one_ignored = []
    if not use_cooc:
        _full_identity = mimic_db.get_extension(IdentityExtension)
        for q in keywords:
            est_value = estimator.estimate(q)
            actual_value = len(_full_identity.doc_ids(q))
            current_error = ErrorMetric(est_value, actual_value)
            error_value_list.append(current_error)
            if ignore_zero_one and not (est_value == 0 and actual_value == 1):
                error_value_list_zero_one_ignored.append(current_error)
    else:
        _full_coocc = mimic_db.get_extension(CoOccurrenceExtension)
        for q1, q2 in keywords:
            est_value = estimator.estimate(q1, q2)
            actual_value = _full_coocc.co_occurrence(q1, q2)
            current_error = ErrorMetric(est_value, actual_value)
            error_value_list.append(current_error)
            if ignore_zero_one and not (est_value == 0 and actual_value == 1):
                error_value_list_zero_one_ignored.append(current_error)

    median = np.median(error_value_list)
    point95 = np.quantile(error_value_list, 0.95)
    point99 = np.quantile(error_value_list, 0.99)
    maximum = np.max(error_value_list)

    if ignore_zero_one:
        median_ignored = np.median(error_value_list_zero_one_ignored)
        point95_ignored = np.quantile(error_value_list_zero_one_ignored, 0.95)
        point99_ignored = np.quantile(error_value_list_zero_one_ignored, 0.99)
        maximum_ignored = np.max(error_value_list_zero_one_ignored)
        return median, point95, point99, maximum, median_ignored, point95_ignored, point99_ignored, maximum_ignored

    '''
    log.info('MEDIAN: ' + str(median))
    log.info('.95: ' + str(point95))
    log.info('.99: ' + str(point99))
    log.info('MAX: ' + str(maximum))
    '''
    return median, point95, point99, maximum


def run_rlen_eval(nr_of_evals=1, nr_of_queries=100, sel=Selectivity.Independent, use_cooc=False, ignore_zero_one=False):
    """ selectivity only for rlen, not for cooc (uses independent query tuples with known co-occ != 0 """
    if not mimic_db.has_extension(IdentityExtension):
        mimic_db.extend_with(IdentityExtension)

    if use_cooc:
        if not mimic_db.has_extension(CoOccurrenceExtension):
            mimic_db.extend_with(CoOccurrenceExtension)

    results_list = []
    for i in range(0, nr_of_evals):
        log.info('######################################')
        log.info('RUNNING ITERATION: ' + str(i+1))
        log.info('######################################')
        for known_data_rate in [i / 10 for i in range(2, 10, 2)]:
            mimic_db.open()
            sampled_db = mimic_db.sample(known_data_rate)

            if not use_cooc:
                kw_sample = mimic_db.queries(max_queries=nr_of_queries, sel=sel)
            else:
                # only use tuples of queries with a known cooc-selectivity != 0
                # (idea: cooc of 0 sampling is best, therefore we focus on the other ones)
                kw_sample = []
                all_sampled_queries = sampled_db.queries()
                _sampled_coocc = sampled_db.get_extension(CoOccurrenceExtension)
                for q1 in range(0, len(all_sampled_queries)):
                    for q2 in range(i+1, len(all_sampled_queries)):
                        if _sampled_coocc.co_occurrence(all_sampled_queries[q1], all_sampled_queries[q2]) > 0:
                            kw_sample.append((all_sampled_queries[q1], all_sampled_queries[q2]))
                if len(kw_sample) > nr_of_queries:
                    kw_sample = list(sample(kw_sample, nr_of_queries))
                else:
                    log.info(f"Only {len(kw_sample)} suitable query tuples. No restriction of number of queries applied.")

            # SAMPLING
            sampling_est = SamplingRelationalEstimator(sample=sampled_db, full=mimic_db)
            log.info('SAMPLING')
            results_list.append(
                ('SAMPLING-' + str(known_data_rate),) + evaluate_estimator(sampling_est, kw_sample, use_cooc, ignore_zero_one))

            # NAIVE
            # naive_est = NaiveRelationalEstimator(sample=sampled_db, full=mimic_db)
            # log.info('NAIVE')
            # results_list.append(('NAIVE-' + str(known_data_rate),) + evaluate_estimator(naive_est, kw_sample, use_cooc, ignore_zero_one))

            # KDE Estimator
            kde_est = KDERelationalEstimator(sample=sampled_db, full=mimic_db)
            log.info('KDE')
            results_list.append(('KDE-' + str(known_data_rate),) + evaluate_estimator(kde_est, kw_sample, use_cooc, ignore_zero_one))

            # UAE Estimator
            # uae_est = UaeRelationalEstimator(sample=sampled_db, full=mimic_db, nr_train_queries=200)
            # 20000 training queries in UAE paper
            # log.info('UAE')
            # results_list.append(
            #    ('UAE-' + str(known_data_rate) + '-200',) + evaluate_estimator(uae_est, kw_sample, use_cooc, ignore_zero_one))

            # Neurocard Estimator
            # neurocard_est = NeurocardRelationalEstimator(sample=sampled_db, full=mimic_db)
            # log.info('Neurocard')
            # results_list.append(
            #    ('Neurocard-' + str(known_data_rate),) + evaluate_estimator(neurocard_est, kw_sample, use_cooc, ignore_zero_one))

            # NARU Estimator
            naru_est = NaruRelationalEstimator(sample=sampled_db, full=mimic_db)
            log.info('NARU')
            results_list.append(('NARU-' + str(known_data_rate),) + evaluate_estimator(naru_est, kw_sample, use_cooc, ignore_zero_one))

    if ignore_zero_one:
        df = pandas.DataFrame(data=[el[:5] for el in results_list], columns=['method', 'median', '.95', '.99', 'max'])
        df_ignored = pandas.DataFrame(data=[(el[0],) + el[5:] for el in results_list], columns=['method', 'median', '.95', '.99', 'max'])
    else:
        df = pandas.DataFrame(data=results_list, columns=['method', 'median', '.95', '.99', 'max'])

    df = df.groupby(['method']).mean()
    df = df.sort_index()
    sns_plot = sns.heatmap(df, annot=True, cmap='RdYlGn_r', fmt=".1f")
    plt.title('dataset=' + str(mimic_db.name()) + ', cooc=' + str(use_cooc) + ', nr_of_evals=' + str(
        nr_of_evals) + ', nr_of_queries=' + str(nr_of_queries) +
              ', ignore_zero_one=False')
    sns_plot.figure.savefig("estimators.png", bbox_inches='tight')
    plt.figure()

    if ignore_zero_one:
        df_ignored = df_ignored.groupby(['method']).mean()
        df_ignored = df_ignored.sort_index()
        sns_plot_ignored = sns.heatmap(df_ignored, annot=True, cmap='RdYlGn_r', fmt=".1f")
        plt.title('dataset=' + str(mimic_db.name()) + ', cooc=' + str(use_cooc) + ', nr_of_evals=' + str(
            nr_of_evals) + ', nr_of_queries=' + str(nr_of_queries) +
                  ', ignore_zero_one=' + str(ignore_zero_one))
        sns_plot_ignored.figure.savefig("estimators_zero_one.png", bbox_inches='tight')

    mimic_db.close()


run_rlen_eval(nr_of_evals=5, nr_of_queries=1000, use_cooc=False, ignore_zero_one=False)
# print(evaluate_actual_rlen(mimic_db.keywords()))
