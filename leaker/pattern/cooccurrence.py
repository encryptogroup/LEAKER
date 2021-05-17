"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
from typing import Iterable, List, Dict

from ..api import Dataset, LeakagePattern
from ..extension import CoOccurrenceExtension, IdentityExtension


class CoOccurrence(LeakagePattern[List[int]]):
    """
    The co-occurrence (co) leakage pattern leaking the co-occurence count of documents returned from other queries.
    """
    def leak(self, dataset: Dataset, keywords: Iterable[str]) -> List[List[int]]:
        if dataset.has_extension(CoOccurrenceExtension):
            coocc = dataset.get_extension(CoOccurrenceExtension)
            return [[coocc.co_occurrence(q, qp) for qp in keywords] for q in keywords]
        elif dataset.has_extension(IdentityExtension):
            identity = dataset.get_extension(IdentityExtension)
            return [[len([i for i in identity.doc_ids(qp) if i in identity.doc_ids(q)]) for qp in keywords]
                    for q in keywords]
        else:
            doc_ids: Dict[str, List[int]] = {q: map(lambda doc: doc.id(), dataset(q)) for q in keywords}
            return [[len([i for i in doc_ids[qp] if i in doc_ids[q]]) for qp in keywords] for q in keywords]