"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import logging
import sys

import numpy as np

from leaker.api import Dataset, Selectivity
from leaker.attack import PartialUserQueryLogSpace, FullUserQueryLogSpace, Ikk, Countv2, VolAn, SelVolAn, SubgraphID, \
    SubgraphVL, Ikkoptimized, PartialQueryLogSpace, FullQueryLogSpace
from leaker.evaluation import KeywordAttackEvaluator, EvaluationCase, DatasetSampler, QuerySelector
from leaker.plotting import KeywordMatPlotLibSink
from leaker.whoosh_interface import WhooshBackend

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('eval_attacks.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.INFO)

log = logging.getLogger(__name__)

backend = WhooshBackend()
file_description = "eval"

allow_repetition = False  # set to True for evaluation with repeating queries

for ql_str, db_str in [("aol", "wikipedia"), ("tair_ql", "tair_db")]:

    db: Dataset = backend.load_dataset(db_str, pickle_description=file_description)

    if ql_str == "aol":
        min_user_count = 1000
        max_user_count = 1005
    else:
        min_user_count = 0
        max_user_count = 5

    # evaluate on all users, frequent users, infrequent users
    for ql, setting in [
        (backend.load_querylog(ql_str, pickle_description=file_description, min_user_count=min_user_count), "all"),
        (backend.load_querylog(ql_str, pickle_description=file_description, min_user_count=min_user_count,
                               max_user_count=max_user_count), "freq"),
        (backend.load_querylog(ql_str, pickle_description=file_description, max_user_count=5, reverse=True), "infreq")]:

        # Print selectivities
        sls = []
        kwrds = []
        sels = []
        for user_id in ql.user_ids():
            sel = []
            for kw in ql.keywords_list(user_id):
                if db.selectivity(kw) > 0:
                    sel.append(db.selectivity(kw))
                    if kw not in kwrds:
                        kwrds.append(kw)
            sls.append(np.nanmean(sel))
            sels.extend(sel)

        sls = np.array(sls)
        sels = np.array(sels)
        log.info(
            f"QL {ql_str}_{setting} has {len(ql.doc_ids())} entries. Mean selectivity is {np.nanmean(sls)} over "
            f"{len([a for a in sls if not np.isnan(a)])} out of {len(ql.user_ids())} users. "
            f"Lowest and max mean selectivity are {(np.min(sls[np.nonzero(sls)]), np.max(sls))}."
            f"Lowest and max selectivity are {(np.min(sels[np.nonzero(sels)]), np.max(sels))}.")

        if setting == "all":
            qsp = PartialQueryLogSpace  # set to FullQueryLogSpace for evaluation of full sampling setting
        else:
            qsp = PartialUserQueryLogSpace  # set to FullUserQueryLogSpace for evaluation of full sampling setting

        # Evaluate attacks
        for sel, sel_str in [(Selectivity.Low, "low"), (Selectivity.High, "high")]:
            eval = KeywordAttackEvaluator(evaluation_case=EvaluationCase(attacks=[Ikk, Ikkoptimized.definition(
                                                                                    deterministic=True),
                                                                                  Countv2, VolAn, SelVolAn,
                                                                                  SubgraphID.definition(epsilon=13),
                                                                                  SubgraphVL.definition(epsilon=7)],
                                                                         dataset=db, runs=5),
                                          dataset_sampler=DatasetSampler(
                                              kdr_samples=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], reuse=True,
                                              monotonic=False),
                                          query_selector=QuerySelector(query_space=qsp,
                                                                       query_log=ql, selectivity=sel,
                                                                       query_space_size=500, queries=150,
                                                                       allow_repetition=allow_repetition),
                                          sinks=KeywordMatPlotLibSink(out_file=f"{ql_str}_{db_str}_{setting}_{sel_str}."
                                                                               f"png"),
                                          parallelism=8)
            eval.run()
