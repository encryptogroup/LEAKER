"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import logging
import sys

from leaker.api.document import InputDocument
from leaker.preprocessing import Preprocessor
from leaker.preprocessing.data import DirectoryEnumerator, RelativeFile, StripPrefix, EMailParser, \
    RelativePrefixFilter, FileLoader, FileToDocument, RelativeContainsFilter, PlainFileParser
from leaker.preprocessing.pipeline import Sink, Filter
from leaker.whoosh_interface.preprocessing import WhooshWriter

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('enron_indexing.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.INFO)

enron = DirectoryEnumerator("../../data_sources/Enron/maildir")


enron_full_filter: Filter[RelativeFile, InputDocument] = FileLoader(EMailParser()) | FileToDocument()
enron_full_sink: Sink[InputDocument] = WhooshWriter("enron_full")


enron_sent_filter: Filter[RelativeFile, InputDocument] = RelativeContainsFilter("_sent_mail/") | \
                                                         FileLoader(EMailParser()) | FileToDocument()
enron_sent_sink: Sink[InputDocument] = WhooshWriter("enron_sent")

enron_su_filter: Filter[RelativeFile, InputDocument] = RelativePrefixFilter("arnold-j") | FileLoader(
    PlainFileParser()) | StripPrefix("arnold-j/") | FileToDocument()
enron_su_sink: Sink[InputDocument] = WhooshWriter("enron_su_pfp")

enron_smu_filter: Filter[RelativeFile, InputDocument] = RelativePrefixFilter(
    {"baughman-d", "gay-r", "heard-m", "hendrickson-s"}) | FileLoader(PlainFileParser()) | FileToDocument()
enron_smu_sink: Sink[InputDocument] = WhooshWriter("enron_s_mu_pfp")

enron_mmu_filter: Filter[RelativeFile, InputDocument] = RelativePrefixFilter(
    {"allen-p", "baughman-d", "buy-r", "forney-j", "gay-r", "heard-m", "hendrickson-s", "hyvl-d",
     "keiser-k"}) | FileLoader(PlainFileParser()) | FileToDocument()
enron_mmu_sink: Sink[InputDocument] = WhooshWriter("enron_m_mu_pfp")

preprocessor = Preprocessor(enron, [enron_full_filter > enron_full_sink,
                                    enron_sent_filter > enron_sent_sink,
                                    enron_su_filter > enron_su_sink,
                                    enron_smu_filter > enron_smu_sink,
                                    enron_mmu_filter > enron_mmu_sink])
preprocessor.run()
