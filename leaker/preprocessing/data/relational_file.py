"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import csv
from logging import getLogger
from typing import List, Iterator, Union, TextIO, Tuple, Set

from .keyword_file import LoadedFile, CsvParser
from ..pipeline import Filter

log = getLogger(__name__)


class RelationalCsvParser(CsvParser):
    """
    A parser for the CSV files that contain relational data. Can take a list of columns that shall not be included.
    """

    _excluded_columns: Union[None, Set[str]]

    def __init__(self, excluded_columns: List[str] = None, delimiter: str = '\t'):
        self._excluded_columns = set(s.lower() for s in excluded_columns)
        super(RelationalCsvParser, self).__init__(delimiter=delimiter)

    def parse(self, f: TextIO) -> Iterator[Tuple[str, str, str]]:
        csv_reader = csv.reader(f, delimiter=self._delimiter)
        line = next(csv_reader, None)  # first line
        excluded_column_ids = set([i for i, value in enumerate(line) if value.strip() in self._excluded_columns])

        for line in csv_reader:
            content = []
            for i in set(range(len(line))).difference(excluded_column_ids):
                content.append(line[i].strip())

            yield ("", content, "")


class FileToRelationalInputDocument(Filter[LoadedFile, List[Union[float, int]]]):
    """
    A `Filter` converting `LoadedFile` to relational rows (table_name, list of values).
    """

    def filter(self, source: Iterator[LoadedFile]) -> Iterator[Tuple[str, List[str]]]:
        for relative_name, content, payload in source:
            res: List[str] = content

            if len(res) == 0:
                log.warning(f"Encountered entry {relative_name} without any content. Skipping that...")
                continue
            yield (relative_name, res)
