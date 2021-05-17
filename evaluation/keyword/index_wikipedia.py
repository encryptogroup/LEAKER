"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import logging
import sys

from leaker.api import InputDocument
from leaker.preprocessing import Filter, Sink, Preprocessor
from leaker.preprocessing.data import DirectoryEnumerator
from leaker.preprocessing.data import RelativeFile, FileLoader, JSONParser, FileToDocument
from leaker.whoosh_interface import WhooshWriter

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('wiki_indexing.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.INFO)

wiki = DirectoryEnumerator("../../data_sources/Wikipedia/")

wiki_filter: Filter[RelativeFile, InputDocument] = FileLoader(JSONParser(content_attribute_name="text")) | \
                                                   FileToDocument()
wiki_sink: Sink[InputDocument] = WhooshWriter("wikipedia")

preprocessor = Preprocessor(wiki, [wiki_filter > wiki_sink])
preprocessor.run()
