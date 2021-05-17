"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import collections
import csv
import re
from abc import ABC
from logging import getLogger
from typing import List, Iterator, Union, TextIO, Tuple

import sqlparse

from .keyword_file import LoadedFile, CsvParser, FileParser
from ..pipeline import Filter

log = getLogger(__name__)


class RangeCsvParser(CsvParser):
    """
    A parser for the CSV files that contain numerical data. Can take rows that only match a certain row via a filter
    [(filter_pos, filter_val)], where each entry at position filter_pos has to match string filter_val
    to include the record.
    """

    _filters: Union[None, List[Tuple[int, str]]]

    def __init__(self, column: int, delimiter: str = '\t',
                 filters: Union[None, Tuple[int, str], List[Tuple[int, str]]] = None,
                 payload_attribute_pos: Union[None, int] = None):
        if isinstance(filters, Tuple):
            filters = [filters]
        self._filters = filters
        super(RangeCsvParser, self).__init__(id_attribute_pos=None, content_attribute_pos=column, delimiter=delimiter,
                                             payload_attribute_pos=payload_attribute_pos)

    def parse(self, f: TextIO) -> Iterator[Tuple[str, str, str]]:
        csv_reader = csv.reader(f, delimiter=self._delimiter)
        next(csv_reader, None)  # skip first line

        content = ""
        for line in csv_reader:
            if self._filters is None or all(filter_val == line[filter_pos].strip()
                                            for filter_pos, filter_val in self._filters):
                content += line[self._content_attribute].strip().lstrip('0') + ","

        yield ("", content[:-1], "")  # Get rid of trailing ','


class NYCInsuranceParser(RangeCsvParser):

    def __init__(self, company_name: Union[None, str] = None):
        super(NYCInsuranceParser, self).__init__(14, ',', (13, company_name))
        self.__company_name = company_name


class HMDALoanParser(RangeCsvParser):

    def __init__(self, county_code: Union[None, str] = None):
        super(HMDALoanParser, self).__init__(7, ',', (12, county_code))


class QueryLogRangeParser(FileParser, ABC):
    _db: Union[str, None]

    def __init__(self, db: Union[str, None] = None):
        super().__init__()
        self._db = db


class QueryLogRangeCsvParser(RangeCsvParser):
    _db: Union[str, None]

    def __init__(self, column: int, db: Union[str, None], delimiter: str = '\t',
                 payload_attribute_pos: Union[None, int] = None):
        super().__init__(column, delimiter, payload_attribute_pos=payload_attribute_pos)
        self._db = db


class SDSSParser(QueryLogRangeCsvParser):

    def __init__(self, db: Union[str, None] = None):
        super().__init__(17, db=db, delimiter=',', payload_attribute_pos=9)

    def parse(self, f: TextIO) -> Iterator[Tuple[str, str, str]]:
        csv_reader = csv.reader(f, delimiter=self._delimiter)
        next(csv_reader, None)  # skip first line

        for i, line in enumerate(csv_reader):
            stmt = line[self._content_attribute].lower()
            payload = line[self._payload_attribute].strip()
            if ">" in stmt or "<" in stmt or 'between' in stmt:
                parsed_stmt = sqlparse.parse(stmt)[0]
                where_results = []  # [(DB, OP, (NUM, NUM))]
                select_results = []
                for token in parsed_stmt.tokens:
                    if isinstance(token, sqlparse.sql.Identifier):
                        p = re.compile("([^ ]+)[ ]+a?s?[ ]?([^ ])")
                        finds = re.findall(p, str(token))
                        if finds is None:
                            continue
                        else:
                            select_results.extend(finds)
                    if isinstance(token, sqlparse.sql.Where):
                        """Match comparisons of the form [column] CMP float and then float CMP [column]"""
                        results = []
                        p = re.compile("([^( ]+)[ ]+between[ ]+\(?(\d+.?\d*[+-]?\d*.?\d*)\)?[ ]+"
                                       "and[ ]+\(?(\d+.?\d*[+-]?\d*.?\d*)")
                        finds = re.findall(p, str(token))
                        if finds is not None:
                            results.extend(finds)
                        else:
                            continue

                        """Group by res[0], the DB name to produce [a,b] range queries from <=b and >=a"""
                        grouped_results = collections.OrderedDict()

                        for res in results:
                            grouped_results.setdefault(res[0], []).append((res[1], res[2]))

                        for db, queries in grouped_results.items():
                            for q in queries:
                                try:
                                    lower_str = q[0].lstrip('(')
                                    upper_str = q[1].rstrip(')')

                                    if '+' in lower_str:
                                        lower = sum(float(t) for t in lower_str.split('+'))
                                    elif '-' in lower_str:
                                        terms = lower_str.split('-')
                                        lower = float(terms[0]) - sum(float(t) for t in terms[1:])
                                    else:
                                        lower = float(lower_str)

                                    if '+' in upper_str:
                                        upper = sum(float(t) for t in upper_str.split('+'))
                                    elif '-' in upper_str:
                                        terms = lower_str.split('-')
                                        upper = float(terms[0]) - sum(float(t) for t in terms[1:])
                                    else:
                                        upper = float(upper_str)


                                except ValueError:
                                    continue

                                where_results.append((db, lower, upper))
                        for sel in select_results:
                            db_dict = dict()
                            for where in where_results:
                                if sel[0] in where[0]:
                                    key = f"{sel[0]}.{where[0]}"
                                elif sel[1] in where[0]:
                                    key = f"{sel[0]}.{where[0][len(sel[1]) + 1:]}"
                                else:
                                    continue
                                if key not in db_dict.keys():
                                    db_dict[key] = []
                                db_dict[key].append((key, where[1], where[2]))

                            for key, values in db_dict.items():
                                if self._db is None or key == self._db:
                                    yield (key, "_".join([str((val[1], val[2])) for val in values]), payload)


class SQLShareParser(QueryLogRangeParser):
    def parse(self, f: TextIO) -> Iterator[Tuple[str, str, str]]:
        stmt = ""

        for l in f:
            if not re.match(r"________________________________________", l):
                stmt += l.lower()
            elif stmt != "":
                """New SQL Statement"""
                if ">" in stmt or "<" in stmt:
                    parsed_stmt = sqlparse.parse(stmt)[0]
                    where_results = []  # [(DB, OP, (NUM, NUM))]
                    select_results = []
                    for token in parsed_stmt.tokens:
                        if isinstance(token, sqlparse.sql.Identifier):
                            p = re.compile("(\[[^ ]+\].\[[^ ]+\])")
                            finds = re.findall(p, str(token))
                            if finds is None:
                                continue
                            else:
                                select_results.extend(finds)
                        if isinstance(token, sqlparse.sql.Where):
                            """Match comparisons of the form [column] CMP float and then float CMP [column]"""
                            results = []
                            for i, cmp_match in enumerate(["(\[.[^\]]*\]|[^ ><=]+) ?(<=|>=|<|>|=) ?(\d+.?\d*)",
                                                           "(\d+.?\d*) ?(<=|>=|<|>|=) ?(\[.[^\]]*\]|[^ ]+)"]):
                                p = re.compile(cmp_match)
                                finds = re.findall(p, str(token))
                                if finds is not None:
                                    if i == 0:
                                        results.extend([res for res in finds if "numeric" not in res[0]])
                                    else:
                                        results.extend([(res[2], res[1], res[0]) for res in finds
                                                        if "numeric" not in res[0]])

                            """Group by res[0], the DB name to produce [a,b] range queries from >=a and <=b"""
                            grouped_results = collections.OrderedDict()

                            for res in results:
                                grouped_results.setdefault(res[0], []).append((res[1], res[2]))

                            for db, queries in grouped_results.items():
                                lower = None
                                upper = None
                                for q in queries:
                                    try:
                                        cmp = q[0].strip('\n').rstrip(')')
                                        val = float(q[1].strip(')'))
                                    except ValueError:
                                        continue
                                    if cmp == ">" and lower is None:
                                        lower = max(0, val - 1)
                                    elif cmp == ">=" and lower is None:
                                        lower = val
                                    elif cmp == "<" and upper is None:
                                        upper = val + 1
                                    elif cmp == "<=" and upper is None:
                                        upper = val
                                    elif cmp == "=":
                                        if lower is None:
                                            lower = val
                                        if upper is None:
                                            upper = val

                                    where_results.append((db, lower, upper))

                            for sel in select_results:
                                for where in where_results:
                                    key = f"{sel}.{where[0]}"
                                    if self._db is None or key == self._db:
                                        yield (key, str((where[1], where[2])), '0')

                stmt = ""


class FileToRangeInputDocument(Filter[LoadedFile, List[Union[float, int]]]):
    """
    A `Filter` converting `LoadedFile` to a list of range values.
    Content of the loaded files have to be comma separated values (int or float)
    """

    def filter(self, source: Iterator[LoadedFile]) -> Iterator[List[Union[float, int]]]:
        for relative_name, content, payload in source:
            res: List[Union[float, int]] = []

            if '.' in content:
                """Float values"""
                for val in content.split(','):
                    try:
                        res.append(float(val))
                    except ValueError:
                        log.debug(f"Encountered non-numeric number, skipping entry...")
                        continue
            else:
                """Int values"""
                for val in content.split(','):
                    try:
                        res.append(int(val))
                    except ValueError:
                        log.debug(f"Encountered non-numeric number, skipping entry...")
                        continue

            if len(res) == 0:
                log.warning(f"Encountered entry {relative_name} without any content. Skipping that...")
                continue
            yield res


class FileToRangeQueryLogInputDocument(Filter[LoadedFile, List[Union[float, int]]]):
    """
    A `Filter` converting `LoadedFile` to a list of range queries.
    Content of the loaded files have to be '_' separated tuples (int or float). Payload has to be the user_id.
    """

    def filter(self, source: Iterator[LoadedFile]) -> Iterator[Tuple[List[Union[Tuple[Union[float, None]],
                                                                                Tuple[Union[int, None]]]], str]]:
        for relative_name, content, payload in source:
            res: List[Union[Tuple[Union[float, None]], Tuple[Union[int, None]]]] = []
            for query_str in content.split('_'):
                try:
                    lower, upper = eval(query_str)
                except ValueError:
                    log.debug(f"Encountered ill-formatted entry, skipping that...")
                    continue
                res.append((lower, upper))

            if len(res) == 0:
                log.warning(f"Encountered entry {relative_name} without any content. Skipping that...")
                continue
            yield res, payload
