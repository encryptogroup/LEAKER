"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber

"""
import os
from logging import getLogger

from whoosh import index
from whoosh.analysis import Analyzer
from whoosh.fields import SchemaClass
from whoosh.index import FileIndex
from whoosh.writing import SegmentWriter

from ..api.constants import WRITING_INTERVAL, WHOOSH_INDEX_DIRECTORY
from .__setup import DataSetIndexSchema, QueryLogIndexSchema, keyword_analyzer
from ..api import InputDocument, QueryInputDocument
from ..preprocessing.writer import DatasetWriter

log = getLogger(__name__)


class WhooshWriter(DatasetWriter):
    """
    A `DatasetWriter` for writing (Document) Datasets to Whoosh indices.

    Parameters
    ----------
    name: str
        the name of the resulting Whoosh index
    """

    _name: str

    _keyword_analyzer: Analyzer
    _writer: SegmentWriter
    _written_count: int
    _ix: FileIndex

    def __init__(self, name: str):
        dirname = WHOOSH_INDEX_DIRECTORY + name
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        self._name = name

        schema = self._schema()
        self._keyword_analyzer = keyword_analyzer()

        log.info(f'Creating index for dataset {name} in {dirname}')
        self._ix = index.create_in(dirname, schema)

        self._open()

    def _open(self) -> None:
        self._writer = self._ix.writer()
        self._written_count = 0

    def write(self, document: InputDocument) -> None:
        if self._writer.is_closed:
            self._open()
        log.debug(f'Writing dataset document {document.id()} with length {document.length()}')
        keywords = set(map(lambda token: token.text, self._keyword_analyzer(document.content(), mode='index')))
        self._writer.add_document(doc_id=document.id(), content=document.content(), keywords=keywords,
                                  length=document.length())

        self._written_count += 1
        if self._written_count >= WRITING_INTERVAL:
            log.debug("Writing interval reached. Committing files...")
            self._writer.commit()
            self._open()

    def flush(self) -> None:
        log.info(f'Dataset {self._name} created. Committing files...')
        self._writer.commit()

    def cancel(self) -> None:
        log.info(f'Creation of dataset {self._name} was cancelled')
        self._writer.cancel()

    def __del__(self) -> None:
        if not self._writer.is_closed:
            self.cancel()

    @classmethod
    def _schema(cls) -> SchemaClass:
        return DataSetIndexSchema()


class WhooshQueryLogWriter(WhooshWriter):
    """
    A `QueryLogWriter` for writing Query Logs to Whoosh indices. Extends WhooshWriter to store user ids.
    """

    def write(self, document: QueryInputDocument) -> None:
        if self._writer.is_closed:
            self._open()
        log.debug(f'Writing query log document {document.id()} with length {document.length()}')
        self._writer.add_document(doc_id=document.id(), content=document.content(), user_id=document.user_id())

        self._written_count += 1
        if self._written_count >= WRITING_INTERVAL:
            log.debug("Writing interval reached. Committing files...")
            self._writer.commit()
            self._open()

    def _schema(self):
        return QueryLogIndexSchema()
