"""
For License information see the LICENSE file.

Authors: Johannes Leupold

"""
from abc import ABC, abstractmethod
from collections.abc import Collection
from functools import reduce
from logging import getLogger
from typing import Iterable, List, Generic, TypeVar, Set, Dict, Type

from ..api import KeywordAttack, Dataset, LeakagePattern, Extension
from ..extension import IdentityExtension, VolumeExtension
from ..pattern import ResponseIdentity, Volume

T = TypeVar("T", bound=Collection)

E = TypeVar("E", bound=Extension, covariant=True)

log = getLogger(__name__)


class Subgraph(KeywordAttack, ABC, Generic[T]):
    """
    Represents the generic framework for the Subgraph attacks from "Revisiting Leakage Abuse Attacks". It makes use of
    any atomic leakage pattern.

    Other Parameters
    ----------------
    cross_filtering : bool
        whether cross filtering can be applied
    epsilon : int
        the epsilon error parameter
    """
    __delta: float
    __epsilon: int

    __cross_filtering: bool

    def __init__(self, known: Dataset, cross_filtering: bool, epsilon: int = 0):
        super(Subgraph, self).__init__(known)

        self.__delta = known.sample_rate()
        self.__cross_filtering = cross_filtering
        self.__epsilon = epsilon

    @classmethod
    @abstractmethod
    def required_leakage(cls) -> List[LeakagePattern[T]]:
        raise NotImplementedError

    @abstractmethod
    def get_candidates(self, leakage: T) -> Set[str]:
        """
        Returns the keyword candidates for the given query leakage.

        Parameters
        ----------
        leakage : T
            the atomic leakage for one specific query

        Returns
        -------
        get_candidates : Set[str]
            possible candidates for the query
        """
        raise NotImplementedError

    @abstractmethod
    def get_neighbours(self, candidate: str) -> Set:
        """
        Returns all known neighbours for the given keyword

        Parameters
        ----------
        candidate : str
            the candidate to retrieve neighbours for

        Returns
        -------
        get_neighbours : Set
            the set of known neighbours in the graph
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_handles(self) -> Set:
        """Returns all known document handles."""
        raise NotImplementedError

    @abstractmethod
    def invert(self, handle) -> Set[str]:
        """
        Inverts the handle function on the given document handle. It shouldn't be implemented if the respective function
        is not bijective.

        Parameters
        ----------
        handle
            the document handle to invert

        Returns
        -------
        invert : Set[str]
            all keywords in the document identified by the handle
        """
        raise NotImplementedError

    def __filter_candidates(self, leakage: T, candidates: Set[str]) -> Set[str]:
        # performs filtering of the candidates by using the size of the known neighbour set and the leaked neighbour set
        observed_neighbours: Set[str] = set(leakage)
        return {w for w in candidates
                if len(self.get_neighbours(w)) >= self.__delta * len(observed_neighbours) - self.__epsilon}

    def __cross_filter(self, leakage: T, candidates: Set[str]) -> Set[str]:
        # performs the cross filtering step for a set of candidates
        known_handles: Set = set(leakage).intersection(self.get_all_handles())

        if len(known_handles) == 0:
            # cross filtering can't be performed, no handle is known
            return candidates

        possible: Set[str] = reduce(lambda a, b: a.intersection(b), map(self.invert, known_handles), candidates)
        return candidates.intersection(possible)

    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        log.info(f"Running {self.name()} at {self.__delta:.3f}")
        query_list: List[str] = list(queries)

        log.debug("Getting Leakage")
        leakage: List[T] = self.required_leakage()[0].leak(dataset, query_list)

        log.debug("Determining Candidates")
        candidate_sets: List[Set[str]] = [self.get_candidates(leak) for leak in leakage]
        log.debug("Filtering Candidates")
        candidate_sets = [self.__filter_candidates(leak, candidates)
                          for leak, candidates in zip(leakage, candidate_sets)]
        log.debug("Filtering done")

        # Cross Filtering
        if self.__cross_filtering:
            log.debug("Cross Filtering Step")
            candidate_sets = [self.__cross_filter(leak, candidates)
                              for leak, candidates in zip(leakage, candidate_sets)]

        # Iterative Elimination
        log.debug("Iterative Elimination Step")
        stable = False
        processed_indices: Set[int] = set()
        while not stable:
            unique_indices = {i for i in range(len(query_list))
                              if len(candidate_sets[i]) == 1}.difference(processed_indices)
            stable = len(unique_indices) == 0

            if not stable:
                i = unique_indices.pop()
                processed_indices.add(i)

                for j in range(len(query_list)):
                    if j == i:
                        continue
                    candidate_sets[j].difference_update(candidate_sets[i])
                    
        log.info(f"Reconstruction completed.")

        return [max(candidates, key=lambda k: len(self.get_neighbours(k))) if len(candidates) > 0 else ""
                for candidates in candidate_sets]


class SubgraphID(Subgraph[Set[str]]):
    """
    Implements the SubgraphID attack from "Revisiting Leakage Abuse Attacks". It uses the ResponseIdentity pattern.

    Other Parameters
    ----------------
    epsilon : int
        the epsilon error parameter
    """

    __known_identities: Dict[str, Set[str]]
    __identity_inversion: Dict[str, Set[str]]

    def __init__(self, known: Dataset, epsilon: int = 0):
        super(SubgraphID, self).__init__(known, known.sample_rate() > 0.50, epsilon)

        self.__known_identities = dict()
        self.__identity_inversion = dict()

        if not known.has_extension(IdentityExtension):
            known.extend_with(IdentityExtension)

        ident = known.get_extension(IdentityExtension)

        for keyword in known.keywords():
            self.__known_identities[keyword] = ident.doc_ids(keyword)

            for doc_id in ident.doc_ids(keyword):
                if doc_id not in self.__identity_inversion:
                    self.__identity_inversion[doc_id] = set()
                self.__identity_inversion[doc_id].add(keyword)

    @classmethod
    def name(cls) -> str:
        return "SubgraphID"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Set[str]]]:
        return [ResponseIdentity()]

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        return {IdentityExtension}

    def get_candidates(self, leakage: Set[str]) -> Set[str]:
        return set(filter(lambda w: self.__known_identities[w].issubset(leakage), self.__known_identities.keys()))

    def get_neighbours(self, candidate: str) -> Set[str]:
        return self.__known_identities[candidate]

    def get_all_handles(self) -> Set[str]:
        return set(self.__identity_inversion.keys())

    def invert(self, handle: str) -> Set[str]:
        return self.__identity_inversion[handle]


class SubgraphVL(Subgraph[List[int]]):
    """
    Implements the SubgraphVL attack from "Revisiting Leakage Abuse Attacks". It uses the Volume pattern.

    Other Parameters
    ----------------
    epsilon : int
        the epsilon error parameter
    """

    __known_volumes: Dict[str, Set[int]]
    __handles: Set[int]

    def __init__(self, known: Dataset, epsilon: int = 0):
        super(SubgraphVL, self).__init__(known, False, epsilon)

        self.__known_volumes = dict()
        self.__handles = set()

        if not known.has_extension(VolumeExtension):
            known.extend_with(VolumeExtension)

        volume = known.get_extension(VolumeExtension)

        for keyword in known.keywords():
            self.__known_volumes[keyword] = set(volume.volumes(keyword))
            self.__handles.update(self.__known_volumes[keyword])

    @classmethod
    def name(cls) -> str:
        return "SubgraphVL"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[List[int]]]:
        return [Volume()]

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        return {VolumeExtension}

    def get_candidates(self, leakage: List[int]) -> Set[str]:
        leakage_set = set(leakage)
        return set(filter(lambda w: self.__known_volumes[w].issubset(leakage_set),
                          self.__known_volumes.keys()))

    def get_neighbours(self, candidate: str) -> Set[int]:
        return self.__known_volumes[candidate]

    def get_all_handles(self) -> Set[int]:
        return self.__handles

    def invert(self, handle) -> Set[str]:
        raise NotImplementedError("The volume function can't be inverted")
