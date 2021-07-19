"""
For License information see the LICENSE file.

Authors: Abdelkarim Kati, Amos Treiber, Michael Yonli

"""
import logging
import math
import random
import sys
from typing import Callable, Set

import numpy as np
import pytest

from leaker.api import DataSink, RandomRangeDatabase, InputDocument, RangeDatabase, BaseRangeDatabase, \
    PermutedBetaRandomRangeDatabase, BTRangeDatabase, ABTRangeDatabase, Selectivity
from leaker.attack import SubgraphVL, VolAn, SelVolAn, SubgraphID, Countv2, LMPrid, LMPrank, ApproxValue, \
    LMPaux, ApproxOrder, GJWbasic, GJWspurious, GJWmissing, GLMP18, LMPappRec, Arrorder, GJWpartial, \
    RangeCountBaselineAttack, Apa
from leaker.attack.kkno import GeneralizedKKNO
from leaker.attack.query_space import MissingBoundedRangeQuerySpace, ShortRangeQuerySpace, \
    ValueCenteredRangeQuerySpace, PermutedBetaRangeQuerySpace, PartialQuerySpace, UniformRangeQuerySpace, \
    BoundedRangeQuerySpace
from leaker.evaluation import KeywordAttackEvaluator, RangeAttackEvaluator, EvaluationCase, DatasetSampler, \
    QuerySelector
from leaker.evaluation.errors import MAError, MaxASymError, MaxABucketError, CountSError, CountAError, \
    CountPartialVolume, MSError, SetCountAError, OrderedMAError
from leaker.extension import VolumeExtension
from leaker.preprocessing import Preprocessor, Filter, Sink
from leaker.preprocessing.data import DirectoryEnumerator, RelativeFile, FileLoader, FileToDocument, \
    PlainFileParser
from leaker.whoosh_interface import WhooshBackend, WhooshWriter

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('test_laa_eval.log', 'w', 'utf-8')
file.setFormatter(f)

log = logging.getLogger(__name__)

logging.basicConfig(handlers=[console, file], level=logging.INFO)


def init_rngs(seed):
    random.seed(seed)
    np.random.seed(seed)


class EvaluatorTestSink(DataSink):
    __n: int
    __cb: Callable[[str, int, float, float, int], None]

    def __init__(self, callback: Callable[[str, int, float, float, int], None]):
        self.__n = 0
        self.__cb = callback

    def register_series(self, series_id: str, user_ids: int = 1) -> None:
        pass

    def offer_data(self, series_id: str, user_id: int, kdr: float, rr: float) -> None:
        self.__cb(series_id, kdr, rr, self.__n)
        self.__n += 1

    def flush(self) -> None:
        pass


def test_indexing():
    random_words = DirectoryEnumerator("data_sources/random_words")

    rw_filter: Filter[RelativeFile, InputDocument] = FileLoader(PlainFileParser()) | FileToDocument()
    rw_sink: Sink[InputDocument] = WhooshWriter("random_words")

    preprocessor = Preprocessor(random_words, [rw_filter > rw_sink])
    preprocessor.run()

    backend = WhooshBackend()
    data = backend.load_dataset('random_words')

    keywords: Set[str] = set()
    with open(f"random_words_processed.txt", "r") as f:
        for line in f:
            for word in line.split():
                keywords.add(word)

    assert keywords == data.keywords()


def test_keyword_attack():
    init_rngs(1)

    backend = WhooshBackend()
    if not backend.has('random_words'):
        test_indexing()
    random_words = backend.load_dataset('random_words').extend_with(VolumeExtension)

    query_space = PartialQuerySpace
    space_size = 500
    query_size = 150
    sel = Selectivity.High

    def verif_cb(series_id: str, kdr: float, rr: float, n: int) -> None:
        golden_dict = {0.25: 0.01, 0.5: 0.1, 0.75: 0.2, 1: 0.5}
        if series_id != "Countv2" or kdr == 1:
            assert (rr >= golden_dict[kdr])

    verifier = EvaluatorTestSink(verif_cb)

    run = KeywordAttackEvaluator(evaluation_case=EvaluationCase(attacks=[VolAn, SelVolAn,
                                                                         SubgraphID.definition(epsilon=13),
                                                                         SubgraphVL.definition(epsilon=7), Countv2],
                                                                dataset=random_words, runs=1),
                                 dataset_sampler=DatasetSampler(kdr_samples=[0.25, 0.5, 0.75, 1.0], reuse=True,
                                                                monotonic=False),
                                 query_selector=QuerySelector(query_space=query_space,
                                                              selectivity=sel,
                                                              query_space_size=space_size, queries=query_size,
                                                              allow_repetition=False),
                                 sinks=verifier,
                                 parallelism=8)

    run.run()


def test_full_range_attacks():
    init_rngs(1)

    db = RandomRangeDatabase("test", 1, 16, density=1)

    def verif_cb(series_id: str, kdr: float, rr: float, n: int) -> None:
        assert rr == 0

    verifier = EvaluatorTestSink(verif_cb)

    run = RangeAttackEvaluator(evaluation_case=EvaluationCase(attacks=[LMPrank, LMPrid,
                                                                       LMPappRec.definition(return_mid_point=False,
                                                                                            error=.0625)],
                                                              dataset=db,
                                                              runs=1,
                                                              error=MAError),
                               range_queries=UniformRangeQuerySpace(db, 10 ** 4, allow_repetition=True,
                                                                    allow_empty=True),
                               query_counts=[10 ** 4],
                               sinks=verifier,
                               parallelism=8)
    run.run()


def test_approx_range_attacks():
    init_rngs(1)

    db_kkno = RandomRangeDatabase("test", 0, 1000, density=0.4, allow_repetition=True)

    golden_dict_kkno = {100: (0.05, 0.08), 1000: (0.01, 0.02), 10000: (0.009, 0.02)}
    golden_dict_avalue = {100: (0.05, 0.4), 500: (0.01, 0.12), 1000: (0.0009, 0.09)}
    golden_dict_lmpapprox = {100: (0.0000001, 0.1), 1000: (0, 0.01), 10000: (0, 0.001)}

    def verif_cb(series_id: str, kdr: float, rr: float, n: int) -> None:
        if series_id == "GeneralizedKKNO":
            assert golden_dict_kkno[kdr][0] <= rr <= golden_dict_kkno[kdr][1]
        if series_id == "ApproxValue":
            assert golden_dict_avalue[kdr][0] <= rr <= golden_dict_avalue[kdr][1]
        if series_id == "LMP-aux":
            assert golden_dict_lmpapprox[kdr][0] <= rr <= golden_dict_lmpapprox[kdr][1]

    verifier = EvaluatorTestSink(verif_cb)

    run = RangeAttackEvaluator(evaluation_case=EvaluationCase(attacks=[GeneralizedKKNO],
                                                              dataset=db_kkno,
                                                              runs=1,
                                                              error=MAError),
                               range_queries=UniformRangeQuerySpace(db_kkno, 10 ** 6, allow_repetition=True,
                                                                    allow_empty=True),
                               query_counts=[100, 1000, 10 ** 4], normalize=True,
                               sinks=verifier,
                               parallelism=8)
    run.run()

    db_avalue = RandomRangeDatabase("test", min_val=1, max_val=1000, length=1000, allow_repetition=True)

    eval = RangeAttackEvaluator(EvaluationCase(ApproxValue, db_avalue, 1, error=MaxASymError),
                                UniformRangeQuerySpace(db_avalue, 10 ** 5, allow_repetition=True, allow_empty=True),
                                [100, 500, 1000],
                                verifier, normalize=True, parallelism=8)

    eval.run()

    db_lmpapprox = RandomRangeDatabase("test", 1, 100, length=80, allow_repetition=True)

    run = RangeAttackEvaluator(evaluation_case=EvaluationCase(attacks=[LMPaux],
                                                              dataset=db_lmpapprox,
                                                              runs=1,
                                                              error=MAError),
                               range_queries=UniformRangeQuerySpace(db_lmpapprox, 10 ** 6, allow_repetition=True,
                                                                    allow_empty=True),
                               query_counts=[100, 1000, 10 ** 4], normalize=True,
                               sinks=verifier,
                               parallelism=8)
    run.run()


def test_approx_order_attack():
    init_rngs(1)

    db = RandomRangeDatabase("test", min_val=1, max_val=1000, length=1000, allow_repetition=True)

    golden_dict = {100: (0.01, 0.35), 500: (0, 0.02), 1000: (0, 0.01)}  # Fig. 3 of GLMP19

    def verif_cb(series_id: str, kdr: float, rr: float, n: int) -> None:
        assert golden_dict[kdr][0] <= rr <= golden_dict[kdr][1]

    verifier = EvaluatorTestSink(verif_cb)

    eval = RangeAttackEvaluator(EvaluationCase(ApproxOrder.definition(bucket_error_rec=True), db, 1,
                                               error=MaxABucketError),
                                UniformRangeQuerySpace(db, 10 ** 5, allow_repetition=True, allow_empty=True),
                                [100, 500, 1000],
                                verifier, normalize=True, parallelism=8)

    eval.run()


def test_range_attack_arr_uniform():
    init_rngs(1)

    db = RandomRangeDatabase("test", 1, 10 ** 3, density=0.4, allow_repetition=False)

    def verif_cb(series_id: str, kdr: float, rr: float, n: int) -> None:
        assert 1 < rr < 25

    verifier = EvaluatorTestSink(verif_cb)

    run = RangeAttackEvaluator(evaluation_case=EvaluationCase(attacks=[Arrorder],
                                                              dataset=db,
                                                              runs=1,
                                                              error=MAError),
                               range_queries=UniformRangeQuerySpace(db, 10 ** 4, allow_repetition=False,
                                                                    allow_empty=False),
                               query_counts=[10 ** 4],
                               sinks=verifier,
                               parallelism=8, normalize=False)
    run.run()


def test_range_attack_arr_shortranges():
    init_rngs(1)

    db = RandomRangeDatabase("test", 1, 10 ** 3, density=0.8, allow_repetition=False)

    def verif_cb(series_id: str, kdr: float, rr: float, n: int) -> None:
        assert 1 < rr < 125

    verifier = EvaluatorTestSink(verif_cb)

    run = RangeAttackEvaluator(evaluation_case=EvaluationCase(attacks=[Arrorder],
                                                              dataset=db,
                                                              runs=1,
                                                              error=MSError),
                               range_queries=ShortRangeQuerySpace(db, 10 ** 4, allow_repetition=False,
                                                                    allow_empty=False),
                               query_counts=[10 ** 4],
                               sinks=verifier,
                               parallelism=8, normalize=False)
    run.run()


def test_range_attack_arr_valuecentered():
    init_rngs(1)

    db = RandomRangeDatabase("test", 1, 10 ** 3, density=0.8, allow_repetition=False)

    def verif_cb(series_id: str, kdr: float, rr: float, n: int) -> None:
        assert 1 < rr < 5000

    verifier = EvaluatorTestSink(verif_cb)

    run = RangeAttackEvaluator(evaluation_case=EvaluationCase(attacks=[Arrorder],
                                                              dataset=db,
                                                              runs=1,
                                                              error=MSError),
                               range_queries=ValueCenteredRangeQuerySpace(db, 5 * 10 ** 4, allow_repetition=False,
                                                                    allow_empty=False),
                               query_counts=[5 * 10 ** 4],
                               sinks=verifier,
                               parallelism=8, normalize=False)
    run.run()


# Takes about 20 seconds
def test_basic_range_counts():
    init_rngs(1)

    v = [5060, 13300, 7080, 4360, 3310, 2280, 1870, 1750, 1570, 1320, 1400, 1350, 1410, 1140, 1400, 1020, 1310, 1440,
         1220]
    db = RangeDatabase("test", [i + 1 for i, val in enumerate(v) for _ in range(val)])

    def verif_cb(series_id: str, kdr: float, rr: float, n: int) -> None:
        assert abs(rr) == pytest.approx(0)

    verifier = EvaluatorTestSink(verif_cb)

    run = RangeAttackEvaluator(evaluation_case=EvaluationCase(attacks=[GJWbasic, GJWspurious],
                                                              dataset=db,
                                                              runs=1,
                                                              error=CountSError),
                               range_queries=BoundedRangeQuerySpace(db, allow_empty=False, allow_repetition=False),
                               query_counts=[-1],
                               sinks=verifier,
                               parallelism=8, normalize=False)
    run.run()

    run = RangeAttackEvaluator(evaluation_case=EvaluationCase(attacks=[GLMP18],
                                                              dataset=db,
                                                              runs=1,
                                                              error=CountSError),
                               range_queries=UniformRangeQuerySpace(db, allow_empty=False, allow_repetition=False),
                               query_counts=[-1],
                               sinks=verifier,
                               parallelism=8, normalize=False)
    run.run()

    v = [1344, 9635, 13377, 17011, 17731, 19053, 21016]
    db = RangeDatabase("test", [i + 1 for i, val in enumerate(v) for _ in range(val)])

    bound = 6
    k = 2
    run = RangeAttackEvaluator(evaluation_case=EvaluationCase(attacks=[GJWmissing.definition(bound=bound, k=k)],
                                                              dataset=db,
                                                              runs=1,
                                                              error=CountSError),
                               range_queries=MissingBoundedRangeQuerySpace(db, allow_empty=False,
                                                                           allow_repetition=False, bound=bound, k=k),
                               query_counts=[-1],
                               sinks=verifier,
                               parallelism=8, normalize=False)
    run.run()


# Takes about 3 minutes and tests the case when preprocessing does not find a solution and clique-finding is employed
def test_glmp18_cliques():
    init_rngs(1)

    v_networkx = [3, 6, 2, 3, 3, 4, 2, 3, 2]
    v_graphtool = [20, 4040, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 780, 10, 10, 20, 30, 10, 10, 20, 10, 10,
                   10, 20, 10, 30, 20, 10, 20]

    def verif_cb_networkx(series_id: str, kdr: float, rr: float, n: int) -> None:
        assert 0 <= rr < .25

    def verif_cb_graphtool(series_id: str, kdr: float, rr: float, n: int) -> None:
        assert 0 <= rr < 60

    for v, verif_cb in [(v_networkx, verif_cb_networkx), (v_graphtool, verif_cb_graphtool)]:
        db = RangeDatabase("test", [i + 1 for i, val in enumerate(v) for _ in range(val)])

        verifier = EvaluatorTestSink(verif_cb)

        run = RangeAttackEvaluator(evaluation_case=EvaluationCase(attacks=[GLMP18],
                                                                  dataset=db,
                                                                  runs=1,
                                                                  error=SetCountAError),
                                   range_queries=UniformRangeQuerySpace(db, allow_empty=False, allow_repetition=False),
                                   query_counts=[-1],
                                   sinks=verifier,
                                   parallelism=8, normalize=False)
        run.run()


@pytest.mark.skip()
def test_gjwpartial():
    init_rngs(1)

    db = RandomRangeDatabase("test", 1, 30, density=10, allow_repetition=True)

    def verif_cb(series_id: str, kdr: float, rr: float, n: int) -> None:
        print(rr)

    verifier = EvaluatorTestSink(verif_cb)

    run = RangeAttackEvaluator(evaluation_case=EvaluationCase(attacks=[GJWpartial],
                                                              dataset=db,
                                                              runs=1,
                                                              error=CountPartialVolume),
                               range_queries=BoundedRangeQuerySpace(db, allow_repetition=False,
                                                                    allow_empty=False),
                               query_counts=[-1],
                               sinks=verifier,
                               parallelism=1, normalize=False)
    run.run()


def test_regular_schemes():
    big_n = 2**10
    init_rngs(1)

    vals = RandomRangeDatabase("test", 1, big_n, density=.5, allow_repetition=True).get_numerical_values()
    db1 = BaseRangeDatabase("test", values=vals)
    assert db1.num_canonical_queries() == sum(big_n - 2 ** i + 1 for i in range(int(math.log2(big_n)) + 1))

    db2 = ABTRangeDatabase("test", values=vals)
    assert db2.num_canonical_queries() == 2*(2*big_n - 1) - math.log2(big_n) - big_n

    db3 = BTRangeDatabase("test", values=vals)
    assert db3.num_canonical_queries() == 2*big_n - 1

    for db in [db1, db2, db3]:

        def verif_cb(series_id: str, kdr: float, rr: float, n: int) -> None:
            pass

        verifier = EvaluatorTestSink(verif_cb)

        run = RangeAttackEvaluator(evaluation_case=EvaluationCase(attacks=[RangeCountBaselineAttack],
                                                                  dataset=db,
                                                                  runs=1,
                                                                  error=CountAError),
                                   range_queries=UniformRangeQuerySpace(db, allow_empty=True, allow_repetition=True),
                                   query_counts=[-1],
                                   sinks=verifier,
                                   parallelism=1, normalize=False)
        run.run()


def test_range_attack_apa():
    init_rngs(1)

    db = PermutedBetaRandomRangeDatabase("test", 1, 2**10, .1)
    values = db.get_numerical_values()

    for db in [ABTRangeDatabase("test", values=values), BTRangeDatabase("test", values=values)]:

        def verif_cb(series_id: str, kdr: float, rr: float, n: int) -> None:
            print(f"rr: {rr}")
            assert 1 < rr < 40

        verifier = EvaluatorTestSink(verif_cb)

        run = RangeAttackEvaluator(evaluation_case=EvaluationCase(attacks=[Apa.definition(m=3)],
                                                                  dataset=db,
                                                                  runs=1,
                                                                  error=OrderedMAError),
                                   range_queries=PermutedBetaRangeQuerySpace(db, 10 ** 4, allow_repetition=True,
                                                                             allow_empty=True, alpha=1, beta=5),
                                   query_counts=[3072],
                                   sinks=verifier,
                                   parallelism=1, normalize=False)
        run.run()


def test_big_q_calculation():

    db = PermutedBetaRandomRangeDatabase("test", 1, 2**10, .05)
    values = db.get_numerical_values()

    abt = ABTRangeDatabase("test0", values=values)
    bt = BTRangeDatabase("test1", values=values)
    base = BaseRangeDatabase("test2", values=values)

    assert abt.num_canonical_queries() == 3060
    assert bt.num_canonical_queries() == 2047
    assert base.num_canonical_queries() == 9228

