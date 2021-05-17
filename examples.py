"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import logging
import sys

from typing import List, Iterable, Tuple

from leaker.api import InputDocument, Dataset, Selectivity, RandomRangeDatabase, RangeAttack, LeakagePattern, \
    RangeDatabase
from leaker.attack import Countv2, PartialQuerySpace, GeneralizedKKNO, UniformRangeQuerySpace
from leaker.evaluation import DatasetSampler, EvaluationCase, QuerySelector, KeywordAttackEvaluator, MAError, \
    RangeAttackEvaluator
from leaker.plotting import KeywordMatPlotLibSink, RangeMatPlotLibSink
from leaker.preprocessing import Filter, Sink, Preprocessor
from leaker.preprocessing.data import DirectoryEnumerator, RelativeFile, FileLoader, EMailParser, FileToDocument, \
    RelativeContainsFilter
from leaker.whoosh_interface import WhooshWriter, WhooshBackend

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

# ----- 1: Keyword Attack Evaluation -----#

###### PRE-PROCESSING ######
# To use LEAKER, the data source you want to use needs to be pre-processed and indexed *once* for later use.
# Let's start with the Enron keyword dataset, available at https://www.cs.cmu.edu/~./enron/.
# Extract it into the data_sources directory and select the folder to index:
enron_dir = DirectoryEnumerator("data_sources/Enron/maildir")

# We can filter certain files and sub-directories by calling instances of Filter (found in leaker.preprocessing.data).
# Let's replicate the experiments of [CGPR15] for Count.v2 on the "sent" directories of Enron emails by calling
# a RelativeContainsFilter("_sent_mail/") that filters out all files not in a "_sent_mail/" directory.
# Then, the files need to be loaded via a FileLoader that parses the file according to a specified FileParser.
# Here, we use the EMailParser() to parse email text files. Alternatively, just calling FileLoader() will convert
# different file types to text files. Then, FileToDocument() converts the extracted text into an internal document.
# These filters and loaders represent a pipeline arriving at InputDocuments.
enron_sent_filter: Filter[RelativeFile, InputDocument] = RelativeContainsFilter("_sent_mail/") | \
                                                         FileLoader(EMailParser()) | FileToDocument()

# These InputDocuments need to be indexed and stored via a Writer, using a WhooshWriter here:
enron_sent_sink: Sink[InputDocument] = WhooshWriter("enron_sent")

# The Preprocessor executes the above specified pre-processing (this can take multiple minutes):

preprocessor = Preprocessor(enron_dir, [enron_sent_filter > enron_sent_sink])
preprocessor.run()

log.info("Pre-processing done.")

# The above process works similarly for query logs and range data, see e.g. the index_range.py in evaluation/range.

###### LOADING ######
# Now, we can load the "enron_sent" data collection using the WhooshBackend. This can take multiple minutes due to
# pre-computation, but you will notice a decrease in runtime in the next loading due to caching:
backend = WhooshBackend()
enron_db: Dataset = backend.load_dataset("enron_sent")

log.info(f"Loaded {enron_db.name()} data. {len(enron_db)} documents with {len(enron_db.keywords())} words.")

# In [CGPR15], the collection was restricted to the most frequent 500 keywords. In this example, we do the same,
# by restricting the size and specifying the Selectivity:
enron_db_restricted = enron_db.restrict_keyword_size(500, Selectivity.High)
log.info(f"{enron_db_restricted.name()} now contains {len(enron_db_restricted)} documents with "
         f"{len(enron_db_restricted.keywords())} words.")

# For the range case, you can similarly use a RangeBackend.

###### EVALUATION ######
# We can evaluate according to many criteria:
attacks = [Countv2]  # the attacks to evaluate
runs = 5  # Amount of evaluations

# From this, we can construct a simple EvaluationCase:
evaluation_case = EvaluationCase(attacks=attacks, dataset=enron_db_restricted, runs=runs)

kdr = [.5, .6, .75, .85, .95]  # known data rates
reuse = True  # If we reuse sampled datasets a number of times (=> we will have a 5x5 evaluation here)
# From this, we can construct a DatasetSampler:
dataset_sampler = DatasetSampler(kdr_samples=kdr, reuse=reuse)

query_space = PartialQuerySpace  # The query space to populate. Here, we use partial sampling from
# the data collection. With a query log, a QueryLogSpace is used.
sel = Selectivity.High  # When sampling queries, we use high selectivity keywords
qsp_size = 500  # Size of the query space
sample_size = 150  # Amount of queries attacked at a time (sampled from the query space)
allow_repetition = False  # If queries can repeat
# From this, we can construct a QuerySelector:
query_selector = QuerySelector(query_space=query_space, selectivity=sel, query_space_size=qsp_size, queries=sample_size,
                               allow_repetition=allow_repetition)

out_file = "countv2_enron_sent_high_partial.png"  # Output file (if desired), will be stored in data/figures

# With these parameters, we can set up the Evaluator:
eva = KeywordAttackEvaluator(evaluation_case=evaluation_case, dataset_sampler=dataset_sampler,
                             query_selector=query_selector,
                             sinks=KeywordMatPlotLibSink(out_file=out_file), parallelism=8)

# And then run it:
eva.run()

# ----- 2: Range Attack Evaluation -----#
# As mentioned, pre-processing and loading range data works similarly to the keyword case.
# Let's therefore consider a randomly generated data collection with values in 1,...,500, n=1200 and repeated values:
example_db = RandomRangeDatabase("test_db", 1, 500, length=1200, allow_repetition=True)

log.info(f"Created range db {example_db.name()} of {len(example_db)} entries with density {example_db.get_density()},"
         f" N={example_db.get_max()}.")

###### EVALUATION ######
# We can evaluate according to many criteria:
attacks = [GeneralizedKKNO]  # the attacks to evaluate
runs = 10  # Amount of evaluations
error = MAError  # Type of error (see leaker.evaluation.errors)
# From this, we can construct a simple EvaluationCase:
evaluation_case = EvaluationCase(attacks=attacks, dataset=example_db, runs=runs, error=error) #

query_space = UniformRangeQuerySpace(example_db, 10 ** 6, allow_repetition=True, allow_empty=True) # Query distribution.
# Here, we use the uniform query space with 10**6 queries being created initially. With a query log of real-world
# queries, a QueryLogRangeQuerySpace is used instead (see leaker.attack.query_space for more spaces).
query_counts = [100, 500, 1000, 10 ** 4]  # How many queries to attack, each amount being sampled from the query space.
normalize = True  # Whether to display absolute error results or normalize them.

out_file = "genkkno_uniform.png"  # Output file (if desired), will be stored in data/figures

# With these parameters, we can set up the Evaluator:
eva = RangeAttackEvaluator(evaluation_case=evaluation_case, range_queries=query_space, query_counts=query_counts,
                           normalize=True, sinks=RangeMatPlotLibSink(out_file=out_file), parallelism=5)
# And then run it:
eva.run()

# ----- 3: New Attack -----#
# To implement a new attack, you just need to create a new class fulfilling the Attack interface, i.e.,
# implementing the name(), required_leakage(), and recover(queries) methods.


class ExampleAttack(RangeAttack):
    """
    Implements a baseline range attack that just outputs the min value, ignoring any leakage
    """
    __min_val: int
    __n: int

    def __init__(self, db: RangeDatabase):
        super().__init__(db)
        self.__min_val = db.get_min()
        self.__n = len(db)

    @classmethod
    def name(cls) -> str:
        return "Example"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[int]]:
        return []

    def recover(self, queries: Iterable[Tuple[int, int]]) -> List[int]:
        """Recover has to provide the adversary's goal. In this case, we want to return the database values.
        A better attack would use leakage information obtained by calling the patterns of self.required_leakage() on
        queries."""
        res = [self.__min_val for _ in range(len(self.db()))]

        return res


# Let's test it in the previous setting:
log.info("Testing new attack.")
eva = RangeAttackEvaluator(evaluation_case=EvaluationCase(attacks=[ExampleAttack], dataset=example_db, runs=runs,
                                                          error=error),
                           range_queries=query_space, query_counts=query_counts,
                           normalize=True, sinks=RangeMatPlotLibSink(out_file="example_attack.png"), parallelism=5)
eva.run()


# ----- 4: New Countermeasure -----#
# To implement a new countermeasure that uses, e.g., false positives, you just need to create a new class fulfilling the
# Data interface, i.e., re-implementing the query() method of a keyword or range database.

class ExampleCountermeasureRangeDatabase(RangeDatabase):
    """
    Dummy class that adds all possible responses as false positives to the query results. This is not practical and
    just serves to show how countermasures can be incorporated into LEAKER, as this will diminish attack results.
    """

    def query(self, *query):
        """Perform a query on (min, max) to give all possible responses to any query, ignoring the actual query."""
        q = (self.get_min(), self.get_max())
        return super().query(q)


# Build a new RangeDatabase out of the previous one:
countermeasure_db = ExampleCountermeasureRangeDatabase("countermeasure", values=example_db.get_numerical_values())

# Let's test it in the previous setting and see that it diminishes accuracy:
log.info("Testing new countermeasure.")
eva = RangeAttackEvaluator(evaluation_case=EvaluationCase(attacks=[GeneralizedKKNO], dataset=countermeasure_db,
                                                          runs=runs, error=error),
                           range_queries=query_space, query_counts=query_counts,
                           normalize=True, sinks=RangeMatPlotLibSink(out_file="example_countermeasure.png"),
                           parallelism=5)
eva.run()
