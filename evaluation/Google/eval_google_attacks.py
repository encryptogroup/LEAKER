"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import logging
import sys

from leaker.api import Selectivity
from leaker.attack import SubgraphVL, VolAn, SelVolAn, SubgraphID, Countv2, FullQuerySpace, FullQueryLogSpace
from leaker.evaluation import KeywordAttackEvaluator, EvaluationCase, DatasetSampler, QuerySelector
from leaker.plotting import KeywordMatPlotLibSink
from leaker.whoosh_interface import WhooshBackend

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('google_attack.log', 'w', 'utf-8')
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

    if db is not None:
        found = 0
        if q_log is not None:
            found = len([1 for kw in q_log.keywords() if db.selectivity(kw) > 0])

        eval_selectivities = [(Selectivity.Independent, "ind")]

        if found < 15:
            log.info(f"Too little query data. Running classic attack eval.")
            q_log = None
            qsp = FullQuerySpace
            qsp_size = 500
            queries = 150
            eval_selectivities.append((Selectivity.High, "high"))
            eval_selectivities.append((Selectivity.Low, "low"))
        else:
            qsp = FullQueryLogSpace
            qsp_size = found
            queries = min(found, 150)
            if queries + 25 < qsp_size:
                eval_selectivities.append((Selectivity.High, "high"))
                eval_selectivities.append((Selectivity.Low, "low"))
            log.info(f"Query space size is {qsp_size}, of which {queries} queries will be sampled")

        samples = [.05, .35, .75, 1]
        attacks = [VolAn, SelVolAn, SubgraphID.definition(epsilon=13), SubgraphVL.definition(epsilon=7)]
        runs = 3
        reuse = True
        parallelism = 8
        if "--low-memory" in sys.argv[1:]:
            runs = 5
            reuse = False
            parallelism = 1
        elif "--high-memory" in sys.argv[1:]:
            samples = [.05, .2, .35, .6, .75, .9, 1]

        if "--countv2" in sys.argv[1:]:
            attacks.append(Countv2)

        for sel, sel_str in eval_selectivities:
            eva = KeywordAttackEvaluator(evaluation_case=EvaluationCase(attacks=attacks,
                                                                        dataset=db, runs=runs),
                                         dataset_sampler=DatasetSampler(kdr_samples=samples, reuse=reuse,
                                                                        monotonic=False),
                                         query_selector=QuerySelector(query_space=qsp, query_log=q_log,
                                                                      selectivity=sel,
                                                                      query_space_size=qsp_size, queries=queries,
                                                                      allow_repetition=False),
                                         sinks=KeywordMatPlotLibSink(out_file=
                                                              f"{data}_{sys.argv[1:]}_q{queries}_{sel_str}.png"),
                                         parallelism=parallelism)

            eva.run()
