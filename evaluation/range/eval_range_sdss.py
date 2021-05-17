"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import logging
import sys

from leaker.api import BTRangeDatabase, ABTRangeDatabase
from leaker.api.backend import RangeBackend
from leaker.attack import QueryLogRangeQuerySpace, Apa, LMPrank, LMPrid, LMPappRec, GeneralizedKKNO, ApproxValue, Arr, \
    Arrorder, GLMP18, GJWbasic, GJWmissing
from leaker.evaluation import RangeAttackEvaluator, EvaluationCase, OrderedMAError, MAError, CountAError
from leaker.plotting import RangeMatPlotLibSink
from leaker.stats import Statistics, StatisticsCase, RangeQueryDistribution

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('sdss.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.INFO)

log = logging.getLogger(__name__)

backend = RangeBackend()

db = backend.load_range_database("sdss_photoobjall.dec")

for ql_str in ["sdss_s_photoobjall.dec", "sdss_m_photoobjall.dec", "sdss_l_photoobjall.dec"]:

    ql = backend.load_range_querylog(ql_str)
    log.info(f"Created QL {ql.name()} with users {ql.user_ids()} and unique queries: {len(ql.queries())}")

    qsp = QueryLogRangeQuerySpace(db, qlog=ql)
    values = db.get_numerical_values()
    log.info(f"QLogspace total queries: {len([b for a in qsp.select() for b in a])}")
    log.debug(f"Created range db {db.name()} of {len(db)} entries with density {db.get_density()}, min,max "
              f"{(db.get_original_min(), db.get_original_max())}.")

    # Query distribution statistics
    stat = Statistics(StatisticsCase([RangeQueryDistribution()], query_data=qsp, dataset=db))
    results = stat.compute()

    # Evaluate data reconstruction attacks
    eval = RangeAttackEvaluator(EvaluationCase([LMPrank, LMPrid, LMPappRec, GeneralizedKKNO, ApproxValue,
                                                Arr, Arrorder], db, 1, error=MAError,
                                               base_restrictions_repetitions=25,
                                               base_restriction_rates=[0.0019076]),  # n=10000
                                range_queries=qsp, query_counts=[len([b for a in qsp.select() for b in a])],
                                sinks=RangeMatPlotLibSink(out_file=f"{ql_str}_{db.name()}_data_rec.png"),
                                normalize=True, parallelism=5)

    eval.run()

    # Evaluate count reconstruction attacks
    eval = RangeAttackEvaluator(EvaluationCase([GLMP18, GJWbasic, GJWmissing], db, 1, error=CountAError,
                                               base_restrictions_repetitions=25,
                                               base_restriction_rates=[0.0019076]),  # n=10000
                                range_queries=qsp, query_counts=[len([b for a in qsp.select() for b in a])],
                                sinks=RangeMatPlotLibSink(out_file=f"{ql_str}_{db.name()}_count_rec.png"),
                                normalize=True, parallelism=5)

    eval.run()

    # Evaluate APA
    db_str = db.name()
    for db in [BTRangeDatabase(f"{db_str}_bt", values=values), ABTRangeDatabase(f"{db_str}_abt", values=values)]:
        eval = RangeAttackEvaluator(EvaluationCase([Apa], db, 1, error=OrderedMAError,
                                                   base_restrictions_repetitions=25,
                                                   base_restriction_rates=[0.0019076 * .1]),  # n=1000
                                    range_queries=qsp, query_counts=[len([b for a in qsp.select() for b in a])],
                                    sinks=RangeMatPlotLibSink(out_file=f"{ql_str}_{db.name()}_apa.png"),
                                    normalize=True, parallelism=5)

        eval.run()
