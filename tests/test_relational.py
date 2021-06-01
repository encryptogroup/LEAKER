"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import logging
import sys
from typing import List

from leaker.api import Selectivity, QueryInputDocument, RelationalQuery
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
