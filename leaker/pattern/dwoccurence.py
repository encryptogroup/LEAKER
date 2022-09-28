"""
For License information see the LICENSE file.

Authors: Patrick Ehrler

"""
from typing import Iterable, List, Dict, Tuple

from leaker.api import Dataset, LeakagePattern
from leaker.extension import CoOccurrenceExtension, IdentityExtension


class DwOccurrence(LeakagePattern[List[int]]):
    """
    The dw-occurrence leakage pattern leaking the document-keyword occurrence matrix
    (appearance of a keyword in a document)
    LEAP 4.1 -> Matrix A' (for known data), Matrix B (for encrypted data -> all data)
    Returns matrix, and corresponding keyword and documents names
    """

    def leak(self, dataset: Dataset, keywords: Iterable[str]) -> Tuple[List[List[int]], List[str], List[str]]:
        keywords_list = list(keywords)
        documents_list = list(dataset.doc_ids())

        if dataset.has_extension(IdentityExtension) or dataset.has_extension(CoOccurrenceExtension):
            identity = dataset.get_extension(IdentityExtension)
            return [[int(d in identity.doc_ids(k)) for d in documents_list] for k in
                    keywords_list], keywords_list, documents_list
        else:
            doc_ids: Dict[str, List[int]] = {q: map(lambda doc: doc.id(), dataset(q)) for q in keywords_list}
            return [[int(d in doc_ids[k]) for d in documents_list] for k in
                    keywords_list], keywords_list, documents_list
