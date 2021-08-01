"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import logging
import sys

from leaker.api import BTRangeDatabase, ABTRangeDatabase
from leaker.api.backend import RangeBackend
from leaker.attack import Apa, LMPrank, LMPrid, LMPappRec, GeneralizedKKNO, ApproxValue, Arr, \
    Arrorder, GLMP18, GJWbasic, GJWmissing, ZipfRangeQuerySpace
from leaker.evaluation import RangeAttackEvaluator, EvaluationCase, OrderedMAError, MAError, CountAError
from leaker.plotting import RangeMatPlotLibSink
from leaker.stats import Statistics, StatisticsCase, SelectivityDistribution

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('range_attacks.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.INFO)

log = logging.getLogger(__name__)

backend = RangeBackend()

for db_str in ["mimic_t4", "mimic_protein_creatine", "mimic_cea", "salaries", "sales", "insurance"]:

    db = backend.load_range_database(db_str)
    log.info(f"Created range db {db.name()} of {len(db)} entries with density {db.get_density()}, min,max "
             f"{(db.get_original_min(), db.get_original_max())}, new max {db.get_max()}.")

    # Data distribution statistics
    stat = Statistics(StatisticsCase([SelectivityDistribution()], dataset=db))
    results = stat.compute()

    if db_str == "mimic_t4":
        num = 20
        frac = .2
        bounds = [10, 73]
    elif db_str == "mimic_protein_creatine" or db_str == "mimic_cea" or db_str == "sales":
        num = 20
        frac = .02
        bounds = [100, 1000]
    elif db_str == "salaries":
        num = 20
        frac = .2
        bounds = [100, 395]
    else:
        num = 10
        frac = .002
        bounds = [1000, 25425]

    for bound in bounds:
        for qsp, qsp_str in [(ZipfRangeQuerySpace(db, 10 ** 7, allow_repetition=True, allow_empty=True, s=5,
                                                  width=bound, restrict_frac=frac), f"zipf_{frac}_{bound}")]:

            # Evaluate data reconstruction attacks
            eval = RangeAttackEvaluator(EvaluationCase([LMPrank, LMPrid, LMPappRec, GeneralizedKKNO, ApproxValue,
                                                        Arr, Arrorder], db, num, error=MAError),
                                        range_queries=qsp, query_counts=[100, 500, 1000, 5000, 10000, 100000],
                                        sinks=RangeMatPlotLibSink(out_file=f"{db.name()}_{qsp_str}_data_rec.png"),
                                        normalize=True, parallelism=5)

            eval.run()

            # Evaluate count reconstruction attacks
            eval = RangeAttackEvaluator(EvaluationCase([GLMP18, GJWbasic.definition(bound=bound),
                                                        GJWmissing.definition(bound=bound, k=bound - 2)], db, num,
                                                       error=CountAError),
                                        range_queries=qsp, query_counts=[100, 500, 1000, 5000, 10000, 100000],
                                        sinks=RangeMatPlotLibSink(out_file=f"{db.name()}_{qsp_str}_count_rec.png"),
                                        normalize=True, parallelism=5)

            eval.run()

            # Evaluate APA
            db_str = db.name()
            values = db.get_numerical_values()
            for db in [BTRangeDatabase(f"{db_str}_bt", values=values),
                       ABTRangeDatabase(f"{db_str}_abt", values=values)]:
                eval = RangeAttackEvaluator(EvaluationCase([Apa], db, num, error=OrderedMAError),
                                            range_queries=qsp, query_counts=[100, 500, 1000, 5000, 10000, 100000],
                                            sinks=RangeMatPlotLibSink(out_file=f"{db.name()}_{qsp_str}_apa.png"),
                                            normalize=True, parallelism=5)

                eval.run()
