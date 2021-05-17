"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import logging
import sys
import numpy as np

from leaker.stats import Statistics, StatisticsCase, QueryDistribution, QuerySelectivityDistribution, \
    SelectivityDistribution
from leaker.whoosh_interface import WhooshBackend

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('google_stats.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.INFO)

log = logging.getLogger(__name__)

backend = WhooshBackend()
file_description = "test"

for data in ["gmail", "drive"]:
    try:
        q_log = backend.load_querylog(f"{data}_log", pickle_description=file_description)
        log.info(f"Loaded {data} Log. {len(q_log.doc_ids())} searches performed for {len(q_log.keywords())} words.")
    except FileNotFoundError:
        log.info(f"No {data} Log found.")
        q_log = None

    try:
        db = backend.load_dataset(f"{data}_data", pickle_description=file_description)

        log.info(f"Loaded {data} data. {len(db.doc_ids())} documents with {len(db.keywords())} words.")
    except FileNotFoundError:
        log.info(f"No {data} data found.")
        db = None

    distributions = []
    if q_log is not None:
        distributions.append(QueryDistribution(file_description=file_description))
    if q_log is not None and db is not None:
        log.info(f"Pre-computing selectivities of queries...")
        sels = np.array([db.selectivity(kw) for kw in q_log.keywords() if db.selectivity(kw) > 0])

        log.info(f"{len(q_log.keywords()) - len(sels)} searched keywords were not found in the data. "
                 f"Mean selectivity of the queries is {np.nanmean(sels)}.")
        distributions.append(QuerySelectivityDistribution(file_description=file_description))
    if db is not None:
        distributions.append(SelectivityDistribution(file_description=file_description))

    stat = Statistics(StatisticsCase(distributions, query_data=q_log, dataset=db), file_description=file_description)

    stat.compute()
