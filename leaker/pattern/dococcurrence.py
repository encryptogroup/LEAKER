"""
For License information see the LICENSE file.

Authors: Patrick Ehrler

"""
from typing import Iterable, List, Dict, Tuple
from leaker.api import Dataset, LeakagePattern
from leaker.extension import IdentityExtension
from leaker.extension.dococcurrence import DocOccurrenceExtension


class DocOccurrence(LeakagePattern[List[int]]):
    """
    The d-occurrence leakage pattern leaking the document occurrence matrix
    (number of keywords that appear both of a pair of documents)
    LEAP 4.1 -> Matrix M' (for known data), Matrix M (for encrypted data -> all data)
    Returns matrix and corresponding documents names
    """

    def leak(self, dataset: Dataset, keywords: Iterable[str]) \
            -> Tuple[List[List[int]], List[str]]:
        documents = list(dataset.doc_ids())

        if dataset.has_extension(DocOccurrenceExtension):
            dococc = dataset.get_extension(DocOccurrenceExtension)
            return [[dococc.doc_occurrence(d1, d2) for d1 in documents] for d2 in documents], documents
        elif dataset.has_extension(IdentityExtension):
            identity = dataset.get_extension(IdentityExtension)
            return [[sum([(d1 in identity.doc_ids(k) and d2 in identity.doc_ids(k)) for k in keywords])
                     for d1 in documents] for d2 in documents], documents
        else:
            doc_ids: Dict[str, List[int]] = {q: map(lambda doc: doc.id(), dataset(q)) for q in keywords}
            return [[sum([(d1 in doc_ids[k] and d2 in doc_ids[k]) for k in keywords])
                     for d1 in documents] for d2 in documents], documents
