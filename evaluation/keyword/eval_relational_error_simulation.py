"""
For License information see the LICENSE file.

Authors: Patrick Ehrler

"""
import logging
import sys

from leaker.api import Dataset, Selectivity
from leaker.attack import PartialQuerySpace
from leaker.attack.scoring import ErrorSimulationRelationalScoring, ErrorSimulationRelationalRefinedScoring
from leaker.evaluation import EvaluationCase, QuerySelector, KnownDatasetSampler
from leaker.evaluation.evaluator import ErrorSimulationRelationalAttackEvaluator
from leaker.extension import SelectivityExtension
from leaker.plotting.matplotlib import ErrorSimulationRelationalMatPlotLibSink
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
rel_db: Dataset = backend.load("dmv_100k_random_11cols")

log.info(
    f"Loaded {rel_db.name()} data. {len(rel_db)} documents with {len(rel_db.keywords())} words. {rel_db.has_extension(SelectivityExtension)}")

attacks = [ErrorSimulationRelationalScoring, ErrorSimulationRelationalRefinedScoring]  # the attacks to evaluate
runs = 3  # Amount of evaluations

# From this, we can construct a simple EvaluationCase:
evaluation_case = EvaluationCase(attacks=attacks, dataset=rel_db, runs=runs)

kdr = [.25]  # known data rates
reuse = True  # If we reuse sampled datasets a number of times
# From this, we can construct a DatasetSampler:
dataset_sampler = KnownDatasetSampler(kdr_samples=kdr, reuse=reuse)

query_space = PartialQuerySpace  # The query space to populate. Here, we use partial sampling from
sel = Selectivity.High  # When sampling queries, we use high selectivity keywords
qsp_size = 500  # Size of the query space
sample_size = 150  # Amount of queries attacked at a time (sampled from the query space)
allow_repetition = False  # If queries can repeat
# From this, we can construct a QuerySelector:
query_selector = QuerySelector(query_space=query_space, selectivity=sel, query_space_size=qsp_size, queries=sample_size,
                               allow_repetition=allow_repetition)

out_file = "error_simulation_dmv_100k_random_11cols_25kdr_3x3_High.png"  # Output file (if desired), will be stored in data/figures

# With these parameters, we can set up the Evaluator:
error_rates = [1, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5,
               1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2]
eva = ErrorSimulationRelationalAttackEvaluator(evaluation_case=evaluation_case, dataset_sampler=dataset_sampler,
                             query_selector=query_selector,
                             sinks=ErrorSimulationRelationalMatPlotLibSink(out_file=out_file),
                             parallelism=1,
                             error_rates=error_rates)

# And then run it:
eva.run()