"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""
from collections import Counter
from logging import getLogger
from typing import Dict, Union, Optional, Set, Any

from .identity import IdentityExtension, Identifier, Keyword
from ..api import Dataset, RelationalDatabase
from ..cache import Cache

log = getLogger(__name__)


class SelectivityExtension(IdentityExtension):
    """
    An extension caching the document identifiers of all documents matching every keyword to provide a lookup
    table for keyword selectivities. This class is a subclass of IdentityExtension and also provides its functionality.
    """

    def __init__(self, dataset: Union[Dataset, RelationalDatabase], doc_ids: Optional[Set[Identifier]] = None,
                 keywords: Optional[Set[Keyword]] = None,
                 original_cache: Optional[Cache[Keyword, Set[Identifier]]] = None,
                 original_identity_extension: Any = None, pickle_filename: str = None):
        self.extension_name = 'Selectivity Cache'
        super(SelectivityExtension, self).__init__(dataset, doc_ids, keywords, original_cache,
                                                   original_identity_extension, pickle_filename)

    def selectivity(self, keyword: str) -> int:
        """
        Returns the selectivity of the given keyword.

        Parameters
        ----------
        keyword : str
            the keyword to look up

        Returns
        -------
        selectivity : int
            the selectivity of the keyword
        """
        return len(self._identity_cache[keyword])

    def max_selectivity(self) -> int:
        """
        Returns the maximum selectivity in the data set.

        Returns
        -------
        max_selectivity : int
            the maximum selectivity value
        """
        return max(map(len, self._identity_cache.values()))

    def min_selectivity(self) -> int:
        """
        Returns the minimum selectivity in the data set.

        Returns
        -------
        min_selectivity : int
            the minimum selectivity value
        """
        return min(map(len, self._identity_cache.values()))

    def selectivity_distribution(self) -> Dict[int, int]:
        """
        Returns the selectivity distrbution of the data set as a dictionary, containing the selectivities as the keys
        and their counts as the values.

        Returns
        -------
        selectivity_distribution : Dict[int, int]
            the selectivity distribution of the data set
        """
        return Counter(map(len, self._identity_cache.values()))

    def sample(self, dataset: Union[Dataset, RelationalDatabase]) -> 'SelectivityExtension':
        if isinstance(dataset, RelationalDatabase):
            return SelectivityExtension(dataset, dataset.doc_ids(), dataset.keywords(), self._identity_cache)
        else:
            return SelectivityExtension(dataset, dataset.doc_ids(), dataset.keywords(), self._identity_cache)
