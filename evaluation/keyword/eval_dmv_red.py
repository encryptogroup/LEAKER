"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import logging
import sys

from leaker.api import Dataset, Selectivity
from leaker.attack import PartialQuerySpace, Countv2, NaruCount
from leaker.attack.dummy import DummyRelationalAttack
from leaker.attack.sap import Sap, RelationalSap, NaruRelationalSap
from leaker.evaluation import DatasetSampler, EvaluationCase, QuerySelector, RelationalAttackEvaluator
from leaker.extension import SelectivityExtension, IdentityExtension, CoOccurrenceExtension
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

#import ctypes
#libgcc_s = ctypes.CDLL('libgcc_s.so.1')

backend = SQLBackend()
log.info(f"has dbs {backend.data_sets()}")
mimic_db: Dataset = backend.load("dmv_10k")
mimic_db.extend_with(SelectivityExtension)
#mimic_db.extend_with(CoOccurrenceExtension)

log.info(f"Loaded {mimic_db.name()} data. {len(mimic_db)} documents with {len(mimic_db.keywords())} words. {mimic_db.has_extension(SelectivityExtension)}")

attacks = [RelationalSap, NaruRelationalSap]  # the attacks to evaluate
runs = 3  # Amount of evaluations
max_keywords = 1000  # restrict base dataset to 1000 queries
base_restrictions_repetitions = 3  # repeat restriction
selectivity = Selectivity.Independent  # choose restriction independent

# From this, we can construct a simple EvaluationCase:
evaluation_case = EvaluationCase(attacks=attacks, dataset=mimic_db, runs=runs, max_keywords=max_keywords,
                                 base_restrictions_repetitions=base_restrictions_repetitions, selectivity=selectivity)

kdr = [.2, .4, .6]  # known data rates
reuse = True  # If we reuse sampled datasets a number of times (=> we will have a 5x5 evaluation here)
# From this, we can construct a DatasetSampler:
dataset_sampler = DatasetSampler(kdr_samples=kdr, reuse=reuse)

query_space = PartialQuerySpace  # The query space to populate. Here, we use partial sampling from
# the data collection. With a query log, a QueryLogSpace is used.
sel = Selectivity.PseudoLowTwo  # When sampling queries, we use high selectivity keywords
qsp_size = 500  # Size of the query space
sample_size = 150  # Amount of queries attacked at a time (sampled from the query space)
allow_repetition = False  # If queries can repeat
# From this, we can construct a QuerySelector:
query_selector = QuerySelector(query_space=query_space, selectivity=sel, query_space_size=qsp_size, queries=sample_size,
                               allow_repetition=allow_repetition)

out_file = "dmv_test.png"  # Output file (if desired), will be stored in data/figures

# With these parameters, we can set up the Evaluator:
eva = RelationalAttackEvaluator(evaluation_case=evaluation_case, dataset_sampler=dataset_sampler,
                                query_selector=query_selector,
                                sinks=KeywordMatPlotLibSink(out_file=out_file), parallelism=1)

# And then run it:
eva.run()

