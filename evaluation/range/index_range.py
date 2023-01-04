"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import logging
import sys
from typing import List

from leaker.api import QueryInputDocument
from leaker.preprocessing import Filter, Sink, Preprocessor
from leaker.preprocessing.data import DirectoryEnumerator, RelativeFile, FileLoader, RangeCsvParser, \
    FileToRangeInputDocument, NYCInsuranceParser, SDSSParser, FileToRangeQueryLogInputDocument
from leaker.preprocessing.data.range_file import SQLShareParser
from leaker.preprocessing.writer import RangeDatabaseWriter, RangeQueryLogWriter

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('range_indexing.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.INFO)

log = logging.getLogger(__name__)

sdss = DirectoryEnumerator("../../data_sources/sdss_qlog/S/")

sdss_filter: Filter[RelativeFile, QueryInputDocument] = FileLoader(SDSSParser('photoobjall.dec')) | \
                                                                 FileToRangeQueryLogInputDocument()
sdss_sink: Sink[List] = RangeQueryLogWriter("sdss_s_photoobjall.dec", scale_factor=100)

preprocessor = Preprocessor(sdss, [sdss_filter > sdss_sink])
preprocessor.run()


sdss = DirectoryEnumerator("../../data_sources/sdss_qlog/M/")

sdss_filter: Filter[RelativeFile, QueryInputDocument] = FileLoader(SDSSParser('photoobjall.dec')) | \
                                                                 FileToRangeQueryLogInputDocument()
sdss_sink: Sink[List] = RangeQueryLogWriter("sdss_m_photoobjall.dec", scale_factor=100)

preprocessor = Preprocessor(sdss, [sdss_filter > sdss_sink])
preprocessor.run()

sdss = DirectoryEnumerator("../../data_sources/sdss_qlog/L/")

sdss_filter: Filter[RelativeFile, QueryInputDocument] = FileLoader(SDSSParser('photoobjall.dec')) | \
                                                                 FileToRangeQueryLogInputDocument()
sdss_sink: Sink[List] = RangeQueryLogWriter("sdss_l_photoobjall.dec", scale_factor=100)

preprocessor = Preprocessor(sdss, [sdss_filter > sdss_sink])
preprocessor.run()

sqlshare = DirectoryEnumerator("../../data_sources/sqlshare_qlog")

sqlshare_filter: Filter[RelativeFile, QueryInputDocument] = FileLoader(
    SQLShareParser('[1123].[gill_done_2].percent_meth')) | FileToRangeQueryLogInputDocument()
sqlshare_sink: Sink[List] = RangeQueryLogWriter("[1123].[gill_done_2].percent_meth", scale_factor=1)

preprocessor = Preprocessor(sqlshare, [sqlshare_filter > sqlshare_sink])
preprocessor.run()

sqlshare = DirectoryEnumerator("../../data_sources/sqlshare_gill_percent_meth")

sqlshare_filter: Filter[RelativeFile, QueryInputDocument] = FileLoader(RangeCsvParser(column=3, delimiter=',')) | \
                                                                 FileToRangeInputDocument()
sqlshare_sink: Sink[List] = RangeDatabaseWriter("[1123].[gill_done_2].percent_meth", scale_factor=1)

preprocessor = Preprocessor(sqlshare, [sqlshare_filter > sqlshare_sink])
preprocessor.run()

sdss = DirectoryEnumerator("../../data_sources/sdss/photoobjall.dec")
sdss_filter: Filter[RelativeFile, QueryInputDocument] = FileLoader(
            RangeCsvParser(column=2, delimiter=',')) | FileToRangeInputDocument()
sdss_sink: Sink[List] = RangeDatabaseWriter("sdss_photoobjall.dec", scale_factor=100)

preprocessor = Preprocessor(sdss, [sdss_filter > sdss_sink])
preprocessor.run()


uk_junior_data = DirectoryEnumerator("../../data_sources/uk_gov_pay_junior")

uk_junior_filter: Filter[RelativeFile, QueryInputDocument] = FileLoader(RangeCsvParser(column=3, delimiter=',')) | \
                                                                 FileToRangeInputDocument()
uk_junior_sink: Sink[List] = RangeDatabaseWriter("salaries")

preprocessor = Preprocessor(uk_junior_data, [uk_junior_filter > uk_junior_sink])
preprocessor.run()

range_data = DirectoryEnumerator("../../data_sources/sales")

range_filter: Filter[RelativeFile, QueryInputDocument] = FileLoader(RangeCsvParser(column=3, delimiter=',',
                                                                                   filters=[(0, '36'), (1, '2')])) | \
                                                                 FileToRangeInputDocument()
range_sink: Sink[List] = RangeDatabaseWriter("sales")

preprocessor = Preprocessor(range_data, [range_filter > range_sink])
preprocessor.run()


range_data = DirectoryEnumerator("../../data_sources/nyc_insurance")

range_filter_allstate: Filter[RelativeFile, QueryInputDocument] = FileLoader(
    NYCInsuranceParser(company_name="ALLSTATE INS. CO.")) | FileToRangeInputDocument()
range_sink_allstate: Sink[List] = RangeDatabaseWriter("insurance")

preprocessor = Preprocessor(range_data, [range_filter_allstate > range_sink_allstate])
preprocessor.run()

for code, meaning, scale in [('50900', 'cea', 1), ('51099', 'protein_creatine', 10), ('50995', 't4', 10)]:

    range_data = DirectoryEnumerator("../../data_sources/mimic/lab")

    range_filter: Filter[RelativeFile, QueryInputDocument] = FileLoader(
        RangeCsvParser(column=5, delimiter=',', filters=[(3, code)])) | FileToRangeInputDocument()
    range_sink: Sink[List] = RangeDatabaseWriter(f"mimic_{meaning}", scale_factor=scale)

    preprocessor = Preprocessor(range_data, [range_filter > range_sink])
    preprocessor.run()
