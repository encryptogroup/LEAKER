"""
For License information see the LICENSE file.

Authors: Amos Treiber, Patrick Ehrler

"""
import logging
import sys

from leaker.api import Dataset, Selectivity
from leaker.attack import PartialQuerySpace
from leaker.attack.sap import RelationalSap
from leaker.evaluation import EvaluationCase, QuerySelector, KnownDatasetSampler
from leaker.evaluation.evaluator import KeywordAttackEvaluator
from leaker.extension import SelectivityExtension
from leaker.plotting import KeywordMatPlotLibSink
from leaker.sql_interface import SQLBackend

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('eval_dmv_red.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.DEBUG)

log = logging.getLogger(__name__)

backend = SQLBackend()
log.info(f"has dbs {backend.data_sets()}")
dmv_db: Dataset = backend.load("dmv_1000k_11cols")
dmv_db.extend_with(SelectivityExtension)
# dmv_db.extend_with(CoOccurrenceExtension)

log.info(
    f"Loaded {dmv_db.name()} data. {len(dmv_db)} documents with {len(dmv_db.keywords())} words. {dmv_db.has_extension(SelectivityExtension)}")

attacks = [RelationalSap]  # the attacks to evaluate
runs = 5  # Amount of evaluations

# From this, we can construct a simple EvaluationCase:
evaluation_case = EvaluationCase(attacks=attacks, dataset=dmv_db, runs=runs)

kdr = [0.01, 0.25, .5, .75, 1]  # known data rates
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

out_file = "sap_dmv_1000k_11cols_500_150(3xHigh).png"  # Output file (if desired), will be stored in data/figures

# With these parameters, we can set up the Evaluator:
eva = KeywordAttackEvaluator(evaluation_case=evaluation_case, dataset_sampler=dataset_sampler,
                             query_selector=query_selector,
                             sinks=KeywordMatPlotLibSink(out_file=out_file),
                             parallelism=8)

# And then run it:
eva.run()
