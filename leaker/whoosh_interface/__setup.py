"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""
from nltk import download as nltk_download
from nltk.corpus import stopwords
from whoosh.analysis import StemmingAnalyzer, StandardAnalyzer, Analyzer
from whoosh.fields import SchemaClass, ID, TEXT, NUMERIC, STORED

nltk_download("stopwords")


class DataSetIndexSchema(SchemaClass):
    doc_id = ID(stored=True)
    content = TEXT(analyzer=StemmingAnalyzer(stoplist=stopwords.words()))
    keywords = STORED
    length = NUMERIC(stored=True)


class QueryLogIndexSchema(SchemaClass):
    doc_id = ID(stored=True)
    user_id = ID(stored=True)
    content = TEXT(stored=True, analyzer=StemmingAnalyzer(stoplist=stopwords.words()))


def keyword_analyzer() -> Analyzer:
    return StandardAnalyzer(stoplist=stopwords.words())
