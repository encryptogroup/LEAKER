"""
For License information see the LICENSE file.

Authors: Dominique Dittert

"""
import logging
import sys
import pickle as pkl

from typing import List, Iterable, Tuple

from leaker.api import InputDocument, Dataset, DummyKeywordQueryLogFromTrends, Selectivity, RandomRangeDatabase, RangeAttack, LeakagePattern, \
    RangeDatabase
from leaker.attack import Countv2, Sap, PartialQuerySpace, PartialQueryLogSpace, GeneralizedKKNO, UniformRangeQuerySpace, Ihop, ScoringAttack
from leaker.attack.query_space import AuxiliaryKnowledgeQuerySpace
from leaker.evaluation import KnownDatasetSampler, SampledDatasetSampler, EvaluationCase, QuerySelector, KeywordAttackEvaluator, MAError, \
    RangeAttackEvaluator
from leaker.extension.selectivity import SelectivityExtension
from leaker.extension.volume import VolumeExtension
from leaker.pattern.cooccurrence import CoOccurrence
from leaker.plotting import KeywordMatPlotLibSink, RangeMatPlotLibSink, SampledMatPlotLibSink
from leaker.preprocessing import Filter, Sink, Preprocessor
from leaker.preprocessing.data import DirectoryEnumerator, RelativeFile, FileLoader, EMailParser, FileToDocument, \
    RelativeContainsFilter, UbuntuMailParser, DebianMailParser
from leaker.whoosh_interface import WhooshWriter, WhooshBackend
from leaker.stats import Statistics, StatisticsCase, QueryDistribution, QuerySelectivityDistribution, StatisticalCloseness,\
    SelectivityDistribution, QueryDistributionResults, QuerySelectivityDistributionResults

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('examples.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.INFO)

log = logging.getLogger(__name__)

"""
This file is intended to give you a brief overview and starting guide to LEAKER. It is not extensive, as many attacks
and settings are not covered (see the evaluations and tests for a more comprehensive overview).
"""

###### PRE-PROCESSING ######
debian_list = DirectoryEnumerator("../Debian")
debian_filter: Filter[RelativeFile, InputDocument] = FileLoader(DebianMailParser()) | FileToDocument()
debian_sink: Sink[InputDocument] = WhooshWriter("debian_data")
preprocessor = Preprocessor(debian_list, [debian_filter > debian_sink])
preprocessor.run()

ubuntu_list = DirectoryEnumerator("../Ubuntu")
ubuntu_filter: Filter[RelativeFile, InputDocument] = FileLoader(UbuntuMailParser()) | FileToDocument()
ubuntu_sink: Sink[InputDocument] = WhooshWriter("ubuntu_data")
preprocessor = Preprocessor(ubuntu_list, [ubuntu_filter > ubuntu_sink])
preprocessor.run()

enron_dir = DirectoryEnumerator("data_sources/Enron/maildir")
enron_sent_filter: Filter[RelativeFile, InputDocument] = RelativeContainsFilter("_sent_mail/") | FileLoader(EMailParser()) | FileToDocument()
enron_sent_sink: Sink[InputDocument] = WhooshWriter("enron_sent")
preprocessor = Preprocessor(enron_dir, [enron_sent_filter > enron_sent_sink])
preprocessor.run()

log.info("Pre-processing done.")


###### LOADING ######
backend = WhooshBackend()
debian_db: Dataset = backend.load_dataset("debian_data")
log.info(f"Loaded {debian_db.name()} data. {len(debian_db)} documents with {len(debian_db.keywords())} words.")
debian_db_restricted = debian_db.restrict_keyword_size(5000,Selectivity.High)

ubuntu_db: Dataset = backend.load_dataset("ubuntu_data")
log.info(f"Loaded {ubuntu_db.name()} data. {len(ubuntu_db)} documents with {len(ubuntu_db.keywords())} words.")
ubuntu_db_restricted = ubuntu_db.restrict_keyword_size(5000,Selectivity.High)

enron_db: Dataset = backend.load_dataset("enron_sent")
log.info(f"Loaded {enron_db.name()} data. {len(enron_db)} documents with {len(enron_db.keywords())} words.")
enron_db_restricted = enron_db.restrict_keyword_size(1000,Selectivity.High)


###### EVALUATION ######
stat = StatisticalCloseness(out_file='stat-close_demo.png')
metric = stat.compute_metric([(enron_db, [.5,.05,.005]),(ubuntu_db,debian_db)])
print(metric)
keyword_trends: dict = None
with open("/home/user/Documents/LEAKER/enron_db.pkl",'rb') as f:
    _, keyword_trends = pkl.load(f)

query_log = DummyKeywordQueryLogFromTrends("trends_querylog", keyword_trends,100,(210,260),5,5,Selectivity.High)
query_space = AuxiliaryKnowledgeQuerySpace

attacks = [ScoringAttack.definition(known_query_size=0.05), Sap.definition(known_frequencies=query_log.frequencies(), chosen_keywords=query_log.chosen_keywords(),alpha=0.5), Ihop.definition(known_frequencies=query_log.frequencies(),chosen_keywords=query_log.chosen_keywords(),alpha=0.5)]
runs = 5  

# ----- 1: Keyword Attack Evaluation for Sampled Data -----#
metric_sampled = dict(list(metric.items())[:3])
evaluation_case = EvaluationCase(attacks=attacks, dataset=enron_db_restricted, runs=runs)

kdr = [.5,.05,.005]  # sample rates
reuse = False  # If we reuse sampled datasets a number of times (=> we will have a 5x5 evaluation here)
# From this, we can construct a DatasetSampler:
dataset_sampler = SampledDatasetSampler(kdr_samples=kdr, reuse=reuse)
sel = Selectivity.High  # When sampling queries, we use high selectivity keywords
qsp_size = 500  # Size of the query space
sample_size = 100  # Amount of queries attacked at a time (sampled from the query space)
allow_repetition = True  # If queries can repeat
# From this, we can construct a QuerySelector:
query_selector = QuerySelector(query_space=query_space, selectivity=sel, query_space_size=qsp_size, queries=sample_size,
                               allow_repetition=allow_repetition, query_log=query_log)

out_file = "sampled-data_attack-demo.png"  # Output file (if desired), will be stored in data/figures

# With these parameters, we can set up the Evaluator:
eva = KeywordAttackEvaluator(evaluation_case=evaluation_case, dataset_sampler=dataset_sampler,
                             query_selector=query_selector,
                             sinks=SampledMatPlotLibSink(out_file=out_file, metric=metric_sampled), parallelism=8)

# And then run it:
eva.run()


# ----- 1: Keyword Attack Evaluation for Different Data -----#
metric_different = {0:list(metric.values())[3]}
evaluation_case = EvaluationCase(attacks=attacks, dataset=debian_db, runs=runs)

reuse = False  # If we reuse sampled datasets a number of times (=> we will have a 5x5 evaluation here)
# From this, we can construct a DatasetSampler:
dataset_sampler = SampledDatasetSampler(training_set=ubuntu_db)
sel = Selectivity.High  # When sampling queries, we use high selectivity keywords
qsp_size = 500  # Size of the query space
sample_size = 100  # Amount of queries attacked at a time (sampled from the query space)
allow_repetition = True  # If queries can repeat
# From this, we can construct a QuerySelector:
query_selector = QuerySelector(query_space=query_space, selectivity=sel, query_space_size=qsp_size, queries=sample_size,
                               allow_repetition=allow_repetition, query_log=query_log)

out_file = "different-data_attack-demo.png"  # Output file (if desired), will be stored in data/figures

# With these parameters, we can set up the Evaluator:
eva = KeywordAttackEvaluator(evaluation_case=evaluation_case, dataset_sampler=dataset_sampler,
                             query_selector=query_selector,
                             sinks=SampledMatPlotLibSink(out_file=out_file, metric=metric_different), parallelism=8)

# And then run it:
eva.run()
