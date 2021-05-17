"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import logging
import sys

from leaker.api import InputDocument, QueryInputDocument
from leaker.preprocessing import Filter, Sink, Preprocessor
from leaker.preprocessing.data import DirectoryEnumerator
from leaker.preprocessing.data import RelativeFile, FileLoader, FileToQueryInputDocument, \
    GoogleLogParser, FileToDocument, GMailParser, PlainFileParser
from leaker.whoosh_interface import WhooshWriter, WhooshQueryLogWriter

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('google_indexing.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.INFO)

gmail_log = DirectoryEnumerator("../../data_sources/MailLog")

if len([1 for _ in gmail_log.elements()]) > 0:
    gmail_log_filter: Filter[RelativeFile, QueryInputDocument] = FileLoader(GoogleLogParser()) | \
                                                                 FileToQueryInputDocument()
    gmail_log_sink: Sink[InputDocument] = WhooshQueryLogWriter("gmail_log")

    preprocessor = Preprocessor(gmail_log, [gmail_log_filter > gmail_log_sink])
    preprocessor.run()

gmail = DirectoryEnumerator("../../data_sources/Mail")

if len([1 for _ in gmail.elements()]) > 0:
    gmail_filter: Filter[RelativeFile, InputDocument] = FileLoader(GMailParser()) | \
                                                        FileToDocument()
    gmail_sink: Sink[InputDocument] = WhooshWriter("gmail_data")

    preprocessor = Preprocessor(gmail, [gmail_filter > gmail_sink])
    preprocessor.run()

drive_log = DirectoryEnumerator("../../data_sources/DriveLog")

if len([1 for _ in drive_log.elements()]) > 0:
    drive_log_filter: Filter[RelativeFile, QueryInputDocument] = FileLoader(GoogleLogParser()) | \
                                                                 FileToQueryInputDocument()
    drive_log_sink: Sink[InputDocument] = WhooshQueryLogWriter("drive_log")

    preprocessor = Preprocessor(drive_log, [drive_log_filter > drive_log_sink])
    preprocessor.run()

drive = DirectoryEnumerator("../../data_sources/Drive")

if len([1 for _ in drive.elements()]) > 0:
    drive_filter: Filter[RelativeFile, InputDocument] = FileLoader(PlainFileParser(), parse_all=False) | \
                                                        FileToDocument()
    drive_sink: Sink[InputDocument] = WhooshWriter("drive_data")

    preprocessor = Preprocessor(drive, [drive_filter > drive_sink])
    preprocessor.run()
