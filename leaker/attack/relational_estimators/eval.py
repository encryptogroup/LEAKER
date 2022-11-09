import logging
import statistics
import sys

import pandas

from leaker.api import Dataset, RelationalDatabase, RelationalQuery
from leaker.attack.relational_estimators.estimator import NaruRelationalEstimator, KDERelationalEstimator
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

# NAIVE
errors = []
ratio = len(mimic_db.documents()) / len(sampled_db.documents())
mimic_db.open()
for query in mimic_db.keywords():
    est = max(1, ratio * sum(1 for _ in sampled_db(query)))  # lower bound at 1 to prevent 0 division
    actual = max(1, sum(1 for _ in mimic_db(query)))
    error = max(est, actual) / min(est, actual)  # q-error, naru paper, p. 8
    errors.append(error)

print('NAIVE:')
print('MEDIAN: ' + str(statistics.median(errors)))
print('.95: ' + str(statistics.quantiles(data=errors, n=100)[94]))
print('.99: ' + str(statistics.quantiles(data=errors, n=100)[98]))
print('MAX: ' + str(max(errors)))
mimic_db.close()

# NARU Estimator
naru_est = NaruRelationalEstimator(sample=sampled_db, full=mimic_db)
mimic_db.open()
errors = []
for query in mimic_db.keywords()[:10]:
    est = max(1, naru_est.estimate(query))  # lower bound at 1 to prevent 0 division
    actual = max(1, sum(1 for _ in mimic_db(query)))
    error = max(est, actual) / min(est, actual)  # q-error, naru paper, p. 8
    errors.append(error)

print('NARU:')
print('MEDIAN: ' + str(statistics.median(errors)))
print('.95: ' + str(statistics.quantiles(data=errors, n=100)[94]))
print('.99: ' + str(statistics.quantiles(data=errors, n=100)[98]))
print('MAX: ' + str(max(errors)))
mimic_db.close()

# KDE Estimator
kde_est = KDERelationalEstimator(sample=sampled_db, full=mimic_db)
mimic_db.open()
errors = []
for query in mimic_db.keywords()[:1000]:
    est = max(1, kde_est.estimate(query))  # lower bound at 1 to prevent 0 division
    actual = max(1, sum(1 for _ in mimic_db(query)))
    error = max(est, actual) / min(est, actual)  # q-error, naru paper, p. 8
    errors.append(error)

print('KDE:')
print('MEDIAN: ' + str(statistics.median(errors)))
print('.95: ' + str(statistics.quantiles(data=errors, n=100)[94]))
print('.99: ' + str(statistics.quantiles(data=errors, n=100)[98]))
print('MAX: ' + str(max(errors)))
mimic_db.close()
