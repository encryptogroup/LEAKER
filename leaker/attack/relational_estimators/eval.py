import logging
import statistics
import sys

import pandas

from leaker.api import Dataset, RelationalDatabase, RelationalQuery
from leaker.attack.relational_estimators.estimator import NaruRelationalEstimator, KDERelationalEstimator, \
    NaiveRelationalEstimator
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
log.info(f"Loaded {mimic_db.name()} data. {len(mimic_db)} documents with {len(mimic_db.keywords())} words. {mimic_db.has_extension(IdentityExtension)}")

sampled_db = mimic_db.sample(0.1)


def evaluate_estimator_rlen(estimator):
    mimic_db.open()
    error_value_list = []
    for q in mimic_db.keywords()[:1000]:
        est_value = max(1, estimator.estimate(q))  # lower bound at 1 to prevent 0 division
        actual_value = max(1, sum(1 for _ in mimic_db(q)))
        current_error = max(est_value, actual_value) / min(est_value, actual_value)  # q-error, naru paper, p. 8
        error_value_list.append(current_error)

    log.info('MEDIAN: ' + str(statistics.median(error_value_list)))
    log.info('.95: ' + str(statistics.quantiles(data=error_value_list, n=100)[94]))
    log.info('.99: ' + str(statistics.quantiles(data=error_value_list, n=100)[98]))
    log.info('MAX: ' + str(max(error_value_list)))
    mimic_db.close()

# NAIVE
naive_est = NaiveRelationalEstimator(sample=sampled_db, full=mimic_db, use_full=True)
log.info('NAIVE')
evaluate_estimator_rlen(naive_est)

# NARU Estimator
naru_est = NaruRelationalEstimator(sample=sampled_db, full=mimic_db)
log.info('NARU')
evaluate_estimator_rlen(naru_est)

# KDE Estimator
kde_est = KDERelationalEstimator(sample=sampled_db, full=mimic_db)
log.info('KDE')
evaluate_estimator_rlen(kde_est)




