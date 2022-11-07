"""
For License information see the LICENSE file.

Authors: Amos Treiber

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

file = logging.FileHandler('relational_mimic_indexing.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.DEBUG)

log = logging.getLogger(__name__)


rel_data = DirectoryEnumerator("../../data_sources/dmv/")

rel_filter: Filter[RelativeFile, QueryInputDocument] = FileLoader(RelationalCsvParser(["dob", "dod", "dod_ssn",
                                                                                       "CHARTTIME", "VALUE", "starttime", "endtime", "amount", "AMOUNTUOM", "RATE", "RATEUOM", "STORETIME", "PATIENTWEIGHT", "TOTALAMOUNT", "TOTALAMOUNTUOM",
                                                                                       "VALUENUM", "intime", "outtime", "COMMENTS_DATE", "ORIGINALAMOUNT", "ORIGINALRATE",
                                                                                       "los", "dod_hosp", "row_id", "STARTDATE", "ENDDATE", "GSN", "NDC", 
                                                                                       "HADM_ID", "SEQ_NUM", "VALUE", "VALUEUOM",
                                                                                       "charttime", "storetime",
                                                                                       "Reg Valid Date",
                                                                                       "Reg Expiration Date"],
                                                                                      delimiter=',')) | \
                                                                     FileToRelationalInputDocument()
rel_sink: Sink[List] = SQLRelationalDatabaseWriter(f"dmv_full")

preprocessor = Preprocessor(rel_data, [rel_filter > rel_sink])
preprocessor.run()

