"""
For License information see the LICENSE file.

Authors: Amos Treiber, Patrick Ehrler

"""
import logging
import sys

from leaker.api import Dataset, Selectivity
from leaker.attack import PartialQuerySpace, Ihop, Score, RefinedScore, Countv2, Ikk, SubgraphID, Ikkoptimized
from leaker.attack.ihop import PerfectRelationalIhop, RelationalIhop
from leaker.attack.sap import RelationalSap, PerfectRelationalSap
from leaker.attack.scoring import RelationalScoring, RelationalRefinedScoring, ScoringAttackTen, \
    RefinedScoringAttack, ScoringAttack
from leaker.evaluation import EvaluationCase, QuerySelector, KnownDatasetSampler
from leaker.evaluation.evaluator import KeywordAttackEvaluator, RelationalAttackEvaluator
from leaker.extension import SelectivityExtension, IdentityExtension
from leaker.plotting import KeywordMatPlotLibSink
from leaker.sql_interface import SQLBackend

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('eval_relational_attacks.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.DEBUG)

log = logging.getLogger(__name__)

backend = SQLBackend()
log.info(f"has dbs {backend.data_sets()}")
rel_db: Dataset = backend.load("dmv_10k_11cols")
#rel_db.extend_with(SelectivityExtension)

log.info(
    f"Loaded {rel_db.name()} data. {len(rel_db)} documents with {len(rel_db.keywords())} words. {rel_db.has_extension(SelectivityExtension)}")

attacks = [SubgraphID, Countv2, RelationalSap, ScoringAttack, RefinedScoringAttack, Ihop]  # the attacks to evaluate
runs = 3  # Amount of evaluations

# From this, we can construct a simple EvaluationCase:
evaluation_case = EvaluationCase(attacks=attacks, dataset=rel_db, runs=runs)

kdr = [0.01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]  # known data rates
reuse = True  # If we reuse sampled datasets a number of times (=> we will have a 5x5 evaluation here)
# From this, we can construct a DatasetSampler:
dataset_sampler = KnownDatasetSampler(kdr_samples=kdr, reuse=reuse)

query_space = PartialQuerySpace  # The query space to populate. Here, we use partial sampling from
# the data collection. With a query log, a QueryLogSpace is used.
sel = Selectivity.High  # When sampling queries, we use high selectivity keywords
qsp_size = 500  # Size of the query space
sample_size = 150  # Amount of queries attacked at a time (sampled from the query space)
allow_repetition = False  # If queries can repeat
# From this, we can construct a QuerySelector:
query_selector = QuerySelector(query_space=query_space, selectivity=sel, query_space_size=qsp_size, queries=sample_size,
                               allow_repetition=allow_repetition)

out_file = "dmv_10k_11cols_1x1_High.png"  # Output file (if desired), will be stored in data/figures

# With these parameters, we can set up the Evaluator:
eva = RelationalAttackEvaluator(evaluation_case=evaluation_case, dataset_sampler=dataset_sampler,
                             query_selector=query_selector,
                             sinks=KeywordMatPlotLibSink(out_file=out_file),
                             parallelism=2)

# And then run it:
eva.run()
