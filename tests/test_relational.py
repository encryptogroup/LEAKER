"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import logging
import sys
from typing import List

import pytest

from leaker.api import Selectivity, QueryInputDocument, RelationalQuery
from leaker.extension import IdentityExtension, SelectivityExtension, CoOccurrenceExtension
from leaker.pattern import ResponseIdentity, ResponseLength, CoOccurrence
from leaker.preprocessing import Preprocessor, Filter, Sink
from leaker.preprocessing.data import DirectoryEnumerator, RelativeFile, FileLoader, RelationalCsvParser, \
    FileToRelationalInputDocument
from leaker.sql_interface import SQLRelationalDatabaseWriter, SQLBackend

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('test_relational.log', 'w', 'utf-8')
file.setFormatter(f)

log = logging.getLogger(__name__)

logging.basicConfig(handlers=[console, file], level=logging.INFO)


def test_indexing():
    rel_data = DirectoryEnumerator("data_sources/random_relational_tables")
    rel_filter: Filter[RelativeFile, QueryInputDocument] = FileLoader(RelationalCsvParser(delimiter=',')) | \
                                                           FileToRelationalInputDocument()
    rel_sink: Sink[List] = SQLRelationalDatabaseWriter("random_csv")

    preprocessor = Preprocessor(rel_data, [rel_filter > rel_sink])
    preprocessor.run()


def test_backend():
    backend = SQLBackend()

    if not backend.has("random_csv"):
        test_indexing()

    rdb = backend.load("random_csv")

    with rdb:

        assert len(rdb.queries()) == 4412
        assert len(rdb.row_ids()) == 1286

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
            for q in rdb_sample.queries(max_queries=100, sel=Selectivity.High):
                assert set(rdb_sample.query(q)).issubset(rdb_sample.row_ids())
                assert rdb_sample.row_ids().issubset(rdb.row_ids())
                if q.table == 0:
                    assert set(rdb_sample.query(q)) == set(rdb.query(q))
                else:
                    assert set(rdb_sample.query(q)).issubset(rdb.query(q))
