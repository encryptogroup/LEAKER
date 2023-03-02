"""
For License information see the LICENSE file.

Authors: Patrick Ehrler

"""
import logging
import sys
from typing import List

from leaker.api import QueryInputDocument
from leaker.preprocessing import Filter, Sink, Preprocessor
from leaker.preprocessing.data import DirectoryEnumerator, RelativeFile, FileLoader, FileToRelationalInputDocument, \
    RelationalCsvParser
from leaker.sql_interface import SQLRelationalDatabaseWriter

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('relational_mimiciv_indexing.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.DEBUG)

log = logging.getLogger(__name__)


rel_data = DirectoryEnumerator("../../data_sources/mimiciv/admissions/")

rel_filter: Filter[RelativeFile, QueryInputDocument] = FileLoader(RelationalCsvParser([],
                                                                                      delimiter=',')) | \
                                                                     FileToRelationalInputDocument()
rel_sink: Sink[List] = SQLRelationalDatabaseWriter(f"mimiciv_admissions")

preprocessor = Preprocessor(rel_data, [rel_filter > rel_sink])
preprocessor.run()