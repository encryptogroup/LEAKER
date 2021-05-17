"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import logging
import sys

from leaker.api import InputDocument, QueryInputDocument
from leaker.preprocessing import Filter, Sink, Preprocessor
from leaker.preprocessing.data import DirectoryEnumerator
from leaker.preprocessing.data import RelativeFile, FileLoader, CsvParser, FileToQueryInputDocument, \
    FileToDocument, RelativePrefixFilter, XMLParser
from leaker.whoosh_interface import WhooshQueryLogWriter, WhooshWriter

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('tair_indexing.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.INFO)

tairql = DirectoryEnumerator("../../data_sources/TAIR/query_log/")

tairql_filter: Filter[RelativeFile, QueryInputDocument] = FileLoader(CsvParser(content_attribute_pos=4,
                                                                               payload_attribute_pos=0,
                                                                               delimiter=',')) | \
                                                          FileToQueryInputDocument()
tairql_sink: Sink[InputDocument] = WhooshQueryLogWriter("tair_ql")

tair = DirectoryEnumerator("../../data_sources/TAIR/collection/")

tair_filter: Filter[RelativeFile, QueryInputDocument] = RelativePrefixFilter("gene_description/") | \
                                                        FileLoader(CsvParser(id_attribute_pos=0,
                                                                             content_attribute_pos=[0, 1, 2, 3, 4, 5],
                                                                             payload_attribute_pos=None,
                                                                             delimiter='\t')) | \
                                                        FileToDocument()

tair_xml_filter: Filter[RelativeFile, QueryInputDocument] = RelativePrefixFilter("genes/") | \
                                                        FileLoader(XMLParser(id_attribute_name=None,
                                                                             doc_attribute_name="TU",
                                                                             content_attribute_name=["FEAT_NAME",
                                                                                                     "PUB_LOCUS",
                                                                                                     "COM_NAME",
                                                                                                     "PUB_COMMENT",
                                                                                                     "CDS_SEQUENCE",
                                                                                                     "PROTEIN_SEQUENCE",
                                                                                                     "TRANSCRIPT_"
                                                                                                     "SEQUENCE"])) | \
                                                        FileToDocument()

tair_sink: Sink[InputDocument] = WhooshWriter("tair_db")

preprocessor = Preprocessor(tairql, [tairql_filter > tairql_sink])
preprocessor.run()

preprocessor = Preprocessor(tair, [tair_filter > tair_sink, tair_xml_filter > tair_sink])
preprocessor.run()
