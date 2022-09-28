"""
For License information see the LICENSE file.

Authors: Patrick Ehrler

"""

import logging
import sys

from leaker.api import Dataset, Selectivity, InputDocument
from leaker.attack.leap import Leap
from leaker.evaluation import EvaluationCase, DatasetSampler
from leaker.evaluation.evaluator import L2KeywordDocumentAttackEvaluator
from leaker.plotting.matplotlib import KeywordDocumentMatPlotLibSink
from leaker.preprocessing import Filter, Sink, Preprocessor
from leaker.preprocessing.data import DirectoryEnumerator, RelativeFile, FileLoader, FileToDocument, EMailParser, \
    RelativeContainsFilter

from leaker.whoosh_interface import WhooshBackend, WhooshWriter

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')
console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)
file = logging.FileHandler('leap_experiment.log', 'w', 'utf-8')
file.setFormatter(f)
logging.basicConfig(handlers=[console, file], level=logging.INFO)
log = logging.getLogger(__name__)

"""
This file reproduces the LEAP attack. It is inspired by the examples.py file where further descriptions can be found.

You need to download/extract the Enron email dataset (https://www.cs.cmu.edu/~enron/) to the 'data_sources' folder.
"""

# IMPORT AND PREPROCESSING (filter to sent mails only)
enron_dir = DirectoryEnumerator("data_sources/Enron/maildir")
enron_sent_filter: Filter[RelativeFile, InputDocument] = RelativeContainsFilter("_sent_mail/") | \
                                                         FileLoader(EMailParser()) | FileToDocument()
enron_sent_sink: Sink[InputDocument] = WhooshWriter("enron_sent")
preprocessor = Preprocessor(enron_dir, [enron_sent_filter > enron_sent_sink])
preprocessor.run()
log.info("Pre-processing done.")

# LOADING OF DATA
backend = WhooshBackend()
enron_db: Dataset = backend.load_dataset("enron_sent")

# RESTRICTION TO 5000 KEYWORDS
log.info(f"Loaded {enron_db.name()} data. {len(enron_db)} documents with {len(enron_db.keywords())} words.")
enron_db_restricted = enron_db.restrict_keyword_size(5000, Selectivity.High)

# EVALUATION
attacks = [Leap]
base_restriction_rates = [0.05]  # restrict dataset to 5% of the documents
base_restriction_repetitions = 3  # repeat experiment with newly chosen base dataset
runs = 3  # number of sampling of one known data rate
evaluation_case = EvaluationCase(attacks=attacks, dataset=enron_db_restricted, runs=runs,
                                 base_restriction_rates=base_restriction_rates,
                                 base_restrictions_repetitions=base_restriction_repetitions)

known_data_rates = [.01, .05, .1, .2, .3, .4, .5, 1]  # without 0.001 and 0.005 because not really visible in graph
reuse = False
dataset_sampler = DatasetSampler(kdr_samples=known_data_rates, reuse=reuse)

out_file = "leap_enron_sent_high_partial.png"
eva = L2KeywordDocumentAttackEvaluator(evaluation_case=evaluation_case, dataset_sampler=dataset_sampler,
                                       sinks=KeywordDocumentMatPlotLibSink(out_file=out_file),
                                       parallelism=4)
eva.run()
