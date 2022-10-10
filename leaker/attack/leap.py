"""
For License information see the LICENSE file.

Authors: Patrick Ehrler

"""
from logging import getLogger

from typing import TypeVar, List, Set, Type, Tuple

from leaker.api import Extension, Dataset, LeakagePattern
from leaker.api.attack import L2KeywordDocumentAttack
from leaker.extension import IdentityExtension
from leaker.extension.dococcurrence import DocOccurrenceExtension
from leaker.pattern.dococcurrence import DocOccurrence
from leaker.pattern.dwoccurence import DwOccurrence

log = getLogger(__name__)

E = TypeVar("E", bound=Extension, covariant=True)


class Leap(L2KeywordDocumentAttack):
    """
    Implements the LEAP attack
    """

    _ed_q_matrix_b = List[List[int]]
    _q_list_b = List[str]
    _ed_list_b = List[str]
    _ed_occ_matrix_m = List[List[int]]
    _ed_list_m = List[str]

    def __init__(self, encrypted_dataset: Dataset):
        log.info(f"Setting up Leap attack for encrypted dataset {encrypted_dataset.name()}. This might take some time.")
        super(Leap, self).__init__(encrypted_dataset)

        if not encrypted_dataset.has_extension(DocOccurrenceExtension):
            encrypted_dataset.extend_with(DocOccurrenceExtension)

        log.info("Creating encrypted documents - query matrix")

        self._ed_q_matrix_b, self._q_list_b, self._ed_list_b = self.required_leakage()[0](encrypted_dataset,
                                                                                          encrypted_dataset.keywords())

        log.info("Creating encrypted documents occurrence matrix. This might take a while...")

        self._ed_occ_matrix_m, self._ed_list_m = \
            self.required_leakage()[1](encrypted_dataset, encrypted_dataset.keywords())

        log.info("Setup complete.")

    @classmethod
    def name(cls) -> str:
        return "Leap"

    @classmethod
    def required_leakage(cls) -> List[LeakagePattern[any]]:
        return [DwOccurrence(), DocOccurrence()]

    @classmethod
    def required_extensions(cls) -> Set[Type[E]]:
        return {IdentityExtension}

    @staticmethod
    def __occurrence(c, m, m_, a__, b, ed_list_m, d_list_m_):
        """
        Implements the occurrence algorithm from the LEAP-paper
        """

        s = {1}
        c_ = c.copy()
        c_new = set()  # to fix infinite loop

        assert len(m) == len(m[0])
        assert len(m_) == len(m_[0])
        assert len(m_) <= len(m)
        assert len(b) == len(a__)
        assert len(m_) == len(a__[0])
        assert len(a__) == len(b)
        assert len(b[0]) == len(m)

        while len(s) != 0:
            s = set()
            mapped_ed = [doc[0] for doc in c_]
            mapped_d = [doc[1] for doc in c_]
            for j_ in range(len(m_)):
                if d_list_m_[j_] not in mapped_d:
                    ed_list = []
                    for j in range(len(m)):
                        if ed_list_m[j] not in mapped_ed:
                            c__j_ = sum([row[j_] for row in a__])
                            c_j = sum([row[j] for row in b])
                            if c__j_ == c_j:
                                ed_list.append(ed_list_m[j])
                    ed_list_new = []
                    for ed in ed_list:
                        for mapping in c_:
                            k = ed_list_m.index(mapping[0])
                            k_ = d_list_m_.index(mapping[1])
                            j = ed_list_m.index(ed)
                            if m[j][k] == m_[j_][k_]:
                                ed_list_new.append(ed)
                    ed_list = ed_list_new
                    if len(ed_list) == 1:
                        s = s | {(ed_list[0], d_list_m_[j_])}
                        c_ = c_ | s
                        c_new = c_new | s  # to fix infinite loop in leap algorithm

        return c_new

    def recover(self, known_dataset: Dataset, known_keywords: List[str]) -> \
            Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
        log.info(f"Running Leap on encrypted dataset {self._encrypted().name()} and known dataset {known_dataset.name()}")
        log.info(f"Encrypted documents: {len(self._encrypted().doc_ids())}, "
                 f"Encrypted queries: {len(self._encrypted().keywords())}")
        log.info(f"Known documents: {len(known_dataset.doc_ids())}, Known keywords: {len(known_keywords)}")

        if not known_dataset.has_extension(DocOccurrenceExtension):
            known_dataset.extend_with(DocOccurrenceExtension)

        log.info("Creating known documents - keyword matrix")
        a_, w_list_a_, d_list_a_ = self.required_leakage()[0](known_dataset, known_keywords)

        b = self._ed_q_matrix_b.copy()
        ed_list_b = self._ed_list_b.copy()
        q_list_b = self._q_list_b.copy()

        assert len(a_) <= len(b)
        assert len(a_[0]) <= len(b[0])

        # Step 0
        c = set()
        r = set()

        # Step 1
        a__ = a_ + [[0 for _ in range(len(a_[0]))] for _ in range(len(b) - len(a_))]

        assert len(a__) == len(b)
        assert len(a__[0]) == len(a_[0])

        b_map = b
        a_map__ = a__

        # Step 2
        vb_list = []
        va_list = []
        for j in range(len(b[0])):
            c_j = sum([row[j] for row in b])
            vb_list.append([c_j])
        for j_ in range(len(a_[0])):
            c_j_ = sum([row[j_] for row in a__])
            va_list.append([c_j_])

        for j in range(len(vb_list)):
            if vb_list.count(vb_list[j]) == 1:
                for j_ in range(len(va_list)):
                    if va_list[j_][0] == vb_list[j][0]:
                        c = c | {(ed_list_b[j], d_list_a_[j_])}

        log.debug(f"Initially found mappings: {len(c)}")

        # Step 3
        m = self._ed_occ_matrix_m.copy()
        ed_list_m = self._ed_list_m.copy()

        log.info("Creating known documents occurrence matrix")
        m_, d_list_m_ = self.required_leakage()[1](known_dataset, known_keywords)

        assert len(m_) <= len(m)
        assert len(m) == len(m[0])
        assert len(m_) == len(m_[0])

        s = self.__occurrence(c, m, m_, a_map__, b_map, ed_list_m, d_list_m_)
        c = c | s

        while True:
            # Step 4
            r_new = set()
            c_new = set()

            # Step 5
            b_c = []
            a_c__ = []
            b_map_transposed = list(map(list, zip(*b_map)))  # transpose to get column
            a_map___transposed = list(map(list, zip(*a_map__)))
            for x in c:
                b_c.append(b_map_transposed[ed_list_b.index(x[0])])
                a_c__.append(a_map___transposed[d_list_a_.index(x[1])])
            b_c = list(map(list, zip(*b_c)))  # reverse transpose
            a_c__ = list(map(list, zip(*a_c__)))

            for x in b_c:
                if b_c.count(x) == 1:
                    if x in a_c__:
                        i = b_c.index(x)
                        i_ = a_c__.index(x)
                        r_new = r_new | {(q_list_b[i], w_list_a_[i_])}
                        r = r | {(q_list_b[i], w_list_a_[i_])}

            # Step 6
            b_r = []
            a_r__ = []
            for x in r:
                b_r.append(b_map[q_list_b.index(x[0])])
                a_r__.append(a_map__[w_list_a_.index(x[1])])
            b_r_transposed = list(map(list, zip(*b_r)))  # transpose to easier access columns
            a_r___transposed = list(map(list, zip(*a_r__)))

            for x in b_r_transposed:
                if b_r_transposed.count(x) == 1:
                    if x in a_r___transposed:
                        j = b_r_transposed.index(x)
                        j_ = a_r___transposed.index(x)
                        c_new = c_new | {(ed_list_b[j], d_list_a_[j_])}
                        c = c | {(ed_list_b[j], d_list_a_[j_])}

            # Step 7
            for x in r:
                index_b = q_list_b.index(x[0])
                index_a__ = w_list_a_.index(x[1])
                b[index_b] = [0 for _ in range(len(b[index_b]))]
                a__[index_a__] = [0 for _ in range(len(a__[index_a__]))]

            b_transposed = list(map(list, zip(*b)))  # transpose to easier access columns
            a___transposed = list(map(list, zip(*a__)))
            mapped_ed = [doc[0] for doc in c]
            mapped_d = [doc[1] for doc in c]
            for j in range(0, len(b_transposed)):
                if ed_list_b[j] not in mapped_ed:
                    c_j = sum(b_transposed[j])
                    vb_list[j].append(c_j)
            for j_ in range(0, len(a___transposed)):
                if d_list_a_[j_] not in mapped_d:
                    c__j_ = sum(a___transposed[j_])
                    va_list[j_].append(c__j_)

            vb_unmapped = [vb_list[i] for i in range(len(vb_list)) if ed_list_b[i] not in mapped_ed]
            for j in range(len(vb_list)):
                if ed_list_b[j] not in mapped_ed:
                    if vb_unmapped.count(vb_list[j]) == 1:
                        for j_ in range(len(va_list)):
                            if d_list_a_[j_] not in mapped_d:
                                if vb_list[j] == va_list[j_]:
                                    c = c | {(ed_list_b[j], d_list_a_[j_])}
                                    c_new = c_new | {(ed_list_b[j], d_list_a_[j_])}

            # Step 8
            s_ = self.__occurrence(c, m, m_, a_map__, b_map, ed_list_m, d_list_m_)
            c_new = c_new | s_
            c = c | s_

            # Step 9
            if len(r_new) != 0 or len(c_new) != 0:
                pass
                log.info(f"Found until now: Recovered Keyword/Query mappings: {len(r)}, "
                         f"Recovered Document/Encrypted-Document mappings: {len(c)}")
            else:
                break
        log.info(f"Reconstruction completed.")

        # calculate accuracy
        if len(c) > 0:
            accuracy_documents = [x[0] == x[1] for x in c].count(True) / len(c)
        else:
            accuracy_documents = 1
        if len(r) > 0:
            accuracy_keywords = [x[0] == x[1] for x in r].count(True) / len(r)
        else:
            accuracy_keywords = 1

        # output warning if accuracy is below 100%
        if accuracy_keywords < 1:
            log.warning(f"Keyword Accuracy: { accuracy_keywords }")
        if accuracy_documents < 1:
            log.warning(f"Document Accuracy: { accuracy_documents }")

        return r, c
