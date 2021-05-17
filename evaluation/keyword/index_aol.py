"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import logging
import sys

from leaker.api import InputDocument, QueryInputDocument
from leaker.preprocessing import Filter, Sink, Preprocessor
from leaker.preprocessing.data import DirectoryEnumerator
from leaker.preprocessing.data import RelativeFile, FileLoader, CsvParser, FileToQueryInputDocument
from leaker.whoosh_interface import WhooshQueryLogWriter

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('aol_indexing.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.INFO)

aol = DirectoryEnumerator("../../data_sources/AOL")

aol_filter: Filter[RelativeFile, QueryInputDocument] = FileLoader(CsvParser(id_attribute_pos=None,
                                                                            content_attribute_pos=1,
                                                                            payload_attribute_pos=0)) | \
                                                       FileToQueryInputDocument()
aol_sink: Sink[InputDocument] = WhooshQueryLogWriter("aol")

preprocessor = Preprocessor(aol, [aol_filter > aol_sink])
preprocessor.run()
