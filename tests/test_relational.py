"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import logging
import sys
from typing import List

import pytest

from leaker.api import Selectivity, QueryInputDocument, RelationalQuery
from leaker.attack import PartialQuerySpace, Countv2
from leaker.attack.dummy import DummyRelationalAttack
from leaker.evaluation import EvaluationCase, RelationalAttackEvaluator, DatasetSampler, QuerySelector
from leaker.extension import IdentityExtension, SelectivityExtension, CoOccurrenceExtension
from leaker.pattern import ResponseIdentity, ResponseLength, CoOccurrence
from leaker.preprocessing import Preprocessor, Filter, Sink
from leaker.preprocessing.data import DirectoryEnumerator, RelativeFile, FileLoader, RelationalCsvParser, \
    FileToRelationalInputDocument
from leaker.sql_interface import SQLRelationalDatabaseWriter, SQLBackend
from .test_laa_eval import EvaluatorTestSink, init_rngs

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

log = logging.getLogger(__name__)

logging.basicConfig(handlers=[console], level=logging.DEBUG)


def test_indexing():
    rel_data = DirectoryEnumerator("data_sources/random_relational_tables")
    rel_filter: Filter[RelativeFile, QueryInputDocument] = FileLoader(RelationalCsvParser(delimiter=',')) | \
                                                           FileToRelationalInputDocument()
    rel_sink: Sink[List] = SQLRelationalDatabaseWriter("random_csv2")

    preprocessor = Preprocessor(rel_data, [rel_filter > rel_sink])
    preprocessor.run()


def test_backend():
    backend = SQLBackend()

    if not backend.has("random_csv2"):
        test_indexing()

    rdb = backend.load("random_csv2")

    with rdb:

        assert len(rdb.queries()) == 4450
        assert len(rdb.row_ids()) == 1298

        for q in rdb.queries(max_queries=100, sel=Selectivity.High):
            qp = RelationalQuery(None, q.table, q.attr, q.value)
            assert len([a for a in rdb.query(q)]) == rdb.selectivity(q)
            assert len([a for a in rdb.query(q)]) == rdb.selectivity(qp)
            assert len([a for a in rdb.query(qp)]) == rdb.selectivity(q)
            assert len([a for a in rdb.query(qp)]) == rdb.selectivity(qp)

            assert set(a for a in rdb.query(q)) == set(a for a in rdb.query(qp))
            assert rdb.selectivity(q) == rdb.selectivity(qp)


def test_pattern_and_extension():
    backend = SQLBackend()

    if not backend.has("random_csv"):
        test_indexing()

    rdb = backend.load("random_csv")

    with rdb:
        queries = rdb.queries(max_queries=100, sel=Selectivity.High)
        results = [set(rdb.query(q)) for q in queries]
        rlens = [len(r) for r in results]
        cooccs = [[len([i for i in rdb.query(q) if i in rdb.query(qp)]) for q in queries] for qp in queries]

        for i in range(2):
            if i == 1:
                rdb.extend_with(IdentityExtension)
                rdb.extend_with(SelectivityExtension)
                rdb.extend_with(CoOccurrenceExtension)

            qid_pattern = ResponseIdentity().leak(rdb, queries)
            rlen_pattern = ResponseLength().leak(rdb, queries)
            cocc_pattern = CoOccurrence().leak(rdb, queries)

            assert qid_pattern == results
            assert rlen_pattern == rlens
            assert cocc_pattern == cooccs


def test_sampling():
    backend = SQLBackend()

    if not backend.has("random_csv"):
        test_indexing()
    rdb = backend.load("random_csv")
    with rdb:
        with rdb.sample(.2, [0]) as rdb_sample:
            assert rdb_sample.row_ids().issubset(rdb.row_ids())
            assert len(rdb_sample.row_ids()) < len(rdb.row_ids())
            queries = rdb_sample.queries(max_queries=100, sel=Selectivity.High)
            for i, q in enumerate(queries):
                assert set(rdb_sample.query(q)).issubset(rdb_sample.row_ids())
                if q.table == 0:
                    assert set(rdb_sample.query(q)) == set(rdb.query(q))
                else:
                    assert set(rdb_sample.query(q)).issubset(rdb.query(q))

            results = [set(rdb_sample.query(q)) for q in queries]
            rlens = [len(r) for r in results]
            cooccs = [[len([i for i in rdb_sample.query(q) if i in rdb_sample.query(qp)]) for q in queries]
                      for qp in queries]

            for i in range(2):
                if i == 1:
                    rdb_sample.extend_with(IdentityExtension)
                    rdb_sample.extend_with(SelectivityExtension)
                    rdb_sample.extend_with(CoOccurrenceExtension)

                qid_pattern = ResponseIdentity().leak(rdb_sample, queries)
                rlen_pattern = ResponseLength().leak(rdb_sample, queries)
                cocc_pattern = CoOccurrence().leak(rdb_sample, queries)

                assert qid_pattern == results
                assert rlen_pattern == rlens
                assert cocc_pattern == cooccs


def test_evaluation():
    init_rngs(1)
    backend = SQLBackend()

    if not backend.has("random_csv"):
        test_indexing()

    rdb = backend.load("random_csv")

    def verif_cb(series_id: str, kdr: float, rr: float, n: int) -> None:
        if series_id != "Countv2" or kdr < 1:
            assert (rr >= 0.00)
        else:
            assert rr >= 0.01

    verifier = EvaluatorTestSink(verif_cb)

    query_space = PartialQuerySpace
    space_size = 500
    query_size = 150
    sel = Selectivity.High

    for par, tab in [(1, None), (8, [1])]:
        run = RelationalAttackEvaluator(evaluation_case=EvaluationCase(attacks=[DummyRelationalAttack, Countv2],
                                                                       dataset=rdb, runs=2),
                                        dataset_sampler=DatasetSampler(kdr_samples=[0.25, 0.5, 0.75, 1.0], reuse=True,
                                                                       monotonic=False, table_samples=tab),
                                        query_selector=QuerySelector(query_space=query_space,
                                                                     selectivity=sel,
                                                                     query_space_size=space_size, queries=query_size,
                                                                     allow_repetition=False),
                                        sinks=verifier,
                                        parallelism=par)

        run.run()
