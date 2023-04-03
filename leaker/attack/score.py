"""
For License information see the LICENSE file.

Author: Nelson Brüchmann

"""
import numpy as np
from leaker.extension.cooccurrence import CoOccurrenceExtension
from leaker.extension.identity import IdentityExtension
from leaker.pattern.cooccurrence import CoOccurrence
from leaker.pattern.identity import ResponseIdentity
from functools import partial, reduce
from ..api import KeywordAttack, Dataset, LeakagePattern, Extension
from typing import Iterable, List, Any, Dict, Set, Counter, Tuple, TypeVar, Type
import random
import copy
from collections import OrderedDict
from logging import getLogger
log = getLogger(__name__)
E = TypeVar("E", bound=Extension, covariant=True)

# The attack needs known keyword-trapdoor pairs. 
# The implementation uses known_query_size of the observed queries.
# Therefore the recovery rate will include the known keyword-trapdoor pairs.
class Score(KeywordAttack):

    _keyword_occ: CoOccurrenceExtension # keyword occurrance matrix, build from known data
    _trapdoor_occ: CoOccurrenceExtension # trapdoor occurrance matrix, build real dataset and used queries

    _known_query_size:int
    _known_keywords: Set[str]
    _known_data_subset: Dataset
    _full_dataset_size: int

    def __init__(self, known: Dataset, known_query_size: int = 23):  # 23 corresponds to 15% of 150 queries
        log.info(f"Setting up Score attack for {known.name()} with {known_query_size} known queries. This might take some time.")
        super(Score, self).__init__(known)
        self._known_data_subset = known
        self._known_qu1ery_size = known_query_size
        self._known_keywords = known.keywords()
        
        if not known.has_extension(CoOccurrenceExtension):
            known.extend_with(CoOccurrenceExtension)
        self._keyword_occ =  known.get_extension(CoOccurrenceExtension)

        self._norm = partial(np.linalg.norm, ord=2)


    @classmethod
    def name(cls) -> str:
        return "Score"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [CoOccurrence()]

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        return {CoOccurrenceExtension}

    def __build_keyword_projection_matrix(self, known_queries: List[str]):
        keyword_occ_projection = []

        for kq in known_queries:
            tmp_occ = []
            for keyword in self._known_keywords:
                tmp_occ.append(self._keyword_occ.co_occurrence(keyword,kq) / len(self._known_data_subset))

            keyword_occ_projection.append(np.array(tmp_occ))
        return np.array(keyword_occ_projection).T
    
    def __build_trapdoor_projection_matrix(self, observed_trapdoors: Iterable[str], known_trapdoors: List[str]):
        trapdoor_occ_projection = []

        for kt in known_trapdoors:
            tmp_occ = []
            for ot in observed_trapdoors:
                tmp_occ.append(self._trapdoor_occ.co_occurrence(ot,kt) / self._full_dataset_size )

            trapdoor_occ_projection.append(np.array(tmp_occ))
        return np.array(trapdoor_occ_projection).T



    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        known_queries = sorted(random.sample(list(queries), self._known_query_size))

        if not dataset.has_extension(CoOccurrenceExtension):
            dataset.extend_with(CoOccurrenceExtension)
        self._full_dataset_size = len(dataset)
        self._trapdoor_occ =  dataset.get_extension(CoOccurrenceExtension)
        
        keyword_occ_projection = self.__build_keyword_projection_matrix(known_queries)
        trapdoor_occ_projection = self.__build_trapdoor_projection_matrix(queries,known_queries)
        prediction = []
        for idx_td,td in enumerate(queries): # queries == trapdoors
            candidates = []
            td_vec = trapdoor_occ_projection[idx_td]
            for idx_kw,kw in enumerate(self._known_keywords):
                kw_vec = keyword_occ_projection[idx_kw]

                vec_diff = kw_vec - td_vec
                td_kw_distance = self._norm(vec_diff)

                if td_kw_distance:
                    score = -np.log(td_kw_distance)
                else:  # If distance==0 => Perfect match
                    score = np.inf

                candidates.append((kw,score))

            candidates.sort(key=lambda tup: tup[1])
            prediction.append(candidates[-1][0])

        return prediction





# The attack needs known keyword-trapdoor pairs. 
# The implementation uses known_query_size of the observed queries.
# Therefore the recovery rate will include the known keyword-trapdoor pairs.
class RefinedScore(KeywordAttack):

    _ref_speed: int

    _keyword_occ: CoOccurrenceExtension # keyword occurrance matrix, build from known data
    _trapdoor_occ: CoOccurrenceExtension # trapdoor occurrance matrix, build real dataset and used queries
    _known_query_size:int
    _known_keywords: Set[str]
    _known_data_subset: Dataset
    _full_dataset_size: int

    def __init__(self, known: Dataset, known_query_size: int = 23, refSpeed:int = 10):  # 23 corresponds to 15% of 150 queries
        
        log.info(f"Setting up RefinedScore attack for {known.name()} with known_query_size={known_query_size} and refSpeed={refSpeed}. This might take some time.")
        super(RefinedScore, self).__init__(known)

        self._known_data_subset = known
        self._known_query_size = known_query_size
        self._known_keywords = known.keywords()
        
        if not known.has_extension(CoOccurrenceExtension):
            known.extend_with(CoOccurrenceExtension)
        self._keyword_occ =  known.get_extension(CoOccurrenceExtension)

        self._norm = partial(np.linalg.norm, ord=2)
        self._ref_speed = refSpeed
    
    @classmethod
    def name(cls) -> str:
        return "RefinedScore"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[Any]]:
        return [CoOccurrence()]

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        return {CoOccurrenceExtension}

    def __build_keyword_projection_matrix(self, known_queries: List[str]):
        keyword_occ_projection = []

        for kq in known_queries:
            tmp_occ = np.array([],dtype=np.float64)
            for keyword in self._known_keywords:
                tmp_occ = np.append(tmp_occ, self._keyword_occ.co_occurrence(keyword,kq) / len(self._known_data_subset))
            keyword_occ_projection.append(tmp_occ)
        return np.array(keyword_occ_projection).T
    
    def __build_trapdoor_projection_matrix(self, observed_trapdoors: Iterable[str], known_trapdoors: List[str]):
        trapdoor_occ_projection = []

        for kt in known_trapdoors:
            tmp_occ = np.array([],dtype=np.float64)
            for ot in observed_trapdoors:
                tmp_occ = np.append(tmp_occ,self._trapdoor_occ.co_occurrence(ot,kt) / self._full_dataset_size )

            trapdoor_occ_projection.append(tmp_occ)
        return np.array(trapdoor_occ_projection).T


    def recover(self, dataset: Dataset, queries: Iterable[str]) -> List[str]:
        known_queries = sorted(random.sample(list(queries), self._known_query_size))
        
        if not dataset.has_extension(CoOccurrenceExtension):
            dataset.extend_with(CoOccurrenceExtension)
        
        self._full_dataset_size = len(dataset)
        self._trapdoor_occ =  dataset.get_extension(CoOccurrenceExtension)

        keyword_occ_projection = self.__build_keyword_projection_matrix(known_queries)
        trapdoor_occ_projection = self.__build_trapdoor_projection_matrix(queries,known_queries)

        # Use OrderedDict for the output to keep the ordering of the observed queries
        final_prediction = OrderedDict.fromkeys(queries,None) 
        for kq in known_queries:
            final_prediction[kq]=kq

        unknown_trapdoors = [(q,index) for index,q in enumerate(queries) ] 

        # first=trapdoor, second=prediction
        known_queries = (known_queries,copy.deepcopy(known_queries))

        while True:
            # remove predicted trapdoors
            unknown_trapdoors = [(td,idx) for td,idx in unknown_trapdoors if td not in known_queries[0]]

            temp_pred = np.ndarray(shape=(0,3))

            for td,idx_td in unknown_trapdoors:
                candidates = []
                td_vec = trapdoor_occ_projection[idx_td]
                for idx_kw,kw in enumerate(self._known_keywords):
                    kw_vec = keyword_occ_projection[idx_kw]
                    vec_diff = kw_vec - td_vec
                    td_kw_distance = self._norm(vec_diff)

                    if td_kw_distance:
                        score = -np.log(td_kw_distance)
                    else:  # If distance==0 => Perfect match
                        score = np.inf
                    candidates.append((kw,score))
                candidates.sort(key=lambda tup: tup[1])
                certainty = candidates[-1][1] - candidates[-2][1]
                certainty = 0 if np.isnan(certainty) else format(certainty, ".9f")
                temp_pred = np.append(temp_pred,[[td,candidates[-1][0], certainty]],axis=0)

            #sort by certainty
            temp_pred = temp_pred[np.argsort(temp_pred[:,2])]

            if len(temp_pred) < self._ref_speed:
                for r in range(len(temp_pred)):
                    final_prediction[temp_pred[-r][0]] = temp_pred[-r][1]
                break
             
            for r in range(1,self._ref_speed+1):
                known_queries[0].append(temp_pred[-r][0])
                known_queries[1].append(temp_pred[-r][1])
                final_prediction[temp_pred[-r][0]] = temp_pred[-r][1]
            
            # appending to existing would be faster
            keyword_occ_projection = self.__build_keyword_projection_matrix(known_queries[1])
            trapdoor_occ_projection = self.__build_trapdoor_projection_matrix(queries,known_queries[0])

        final_prediction = [y for x,y in final_prediction.items()]    
        return final_prediction
