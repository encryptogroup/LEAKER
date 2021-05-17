"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import logging
import sys
import numpy as np

from leaker.stats import Statistics, StatisticsCase, QueryDistribution, QuerySelectivityDistribution, \
    SelectivityDistribution, QueryDistributionResults, QuerySelectivityDistributionResults
from leaker.whoosh_interface import WhooshBackend
from leaker.api import KeywordQueryLog, Dataset

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('keyword_stats.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.INFO)

log = logging.getLogger(__name__)

backend = WhooshBackend()

file_description = "stats"  # to identify different evaluations


for ql_str, db_str in [("aol", "wikipedia"), ("tair_ql", "tair_db")]:

    if ql_str == "aol":
        ql: KeywordQueryLog = backend.load_querylog(ql_str, pickle_description=file_description, min_user_count=1000,
                                                    max_user_count=26000)
    else:
        ql: KeywordQueryLog = backend.load_querylog(ql_str, pickle_description=file_description)

    db: Dataset = backend.load_dataset(db_str, pickle_description=file_description)

    sls = []
    kwrds = []
    for user_id in ql.user_ids():
        sel = []
        for kw in ql.keywords_list(user_id):
            if db.selectivity(kw) > 0:
                sel.append(db.selectivity(kw))
                if kw not in kwrds:
                    kwrds.append(kw)
        sls.append(np.nanmean(sel))

    log.info(f"QL has {len(ql.doc_ids())} entries. Mean Selectivity is {np.nanmean(sls)} over "
             f"{len([a for a in sls if not np.isnan(a)])} out of {len(ql.user_ids())} users.")

    stat = Statistics(StatisticsCase([QueryDistribution(file_description=file_description),
                                      QuerySelectivityDistribution(file_description=file_description),
                                      SelectivityDistribution(file_description=file_description)],
                                     query_data=ql, dataset=db), file_description=file_description)

    results = stat.compute()

    qdistro = []
    qsel = []
    qselcorr = []
    r = 0
    corr = 0
    for res in results:
        if isinstance(res, QueryDistributionResults):
            for user_res in res.user_exponents:
                if isinstance(user_res.R, float):
                    qdistro.append(user_res.R)
        elif isinstance(res, QuerySelectivityDistributionResults):
            r = res.overall_exponent.R
            corr = res.overall_coefficient
            for user_res in res.user_exponents:
                if isinstance(user_res.R, float):
                    qsel.append(user_res.R)
            for user_res in res.user_coefficients:
                if isinstance(user_res, float):
                    qselcorr.append(user_res)

    log.info(f"Aggregated R: {r} aggregated corr: {corr}")
    log.info(f"Mean R: {np.mean(qsel)} and mean corr: {np.mean(qselcorr)} over all users.")
