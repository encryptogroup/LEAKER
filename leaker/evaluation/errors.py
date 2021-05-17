"""
For License information see the LICENSE file.

Authors: Abdelkarim Kati, Amos Treiber, Michael Yonli

"""
from abc import ABC
from itertools import combinations, chain
from logging import getLogger
from typing import List, Any

import numpy as np

from ..api import RangeDatabase

log = getLogger(__name__)


class Error(ABC):
    """Abstract class to implement errors used to evaluate range attacks"""

    def __new__(cls, db: RangeDatabase, recovered: List[int], normalize: bool = True) -> float:
        return cls.calc_error(db, recovered, normalize)

    @classmethod
    def calc_error(cls, db: RangeDatabase, recovered: List[int], normalize: bool = True) -> float:
        vals = db.get_numerical_values()
        r_vals = np.subtract(db.get_max() + 1, list(vals))

        return min(cls._calc_error(db, vals, recovered, normalize), cls._calc_error(db, r_vals, recovered, normalize))

    @classmethod
    def _calc_error(cls, db: RangeDatabase, vals: List[int], recovered: List[int], normalize: bool) -> float:
        raise NotImplementedError


class MSDError(Error):
    """Mean Symmetric Difference Error (Cardinality of the set difference)"""

    @classmethod
    def _calc_error(cls, db: RangeDatabase, vals: List[int], recovered: List[int], normalize: bool) -> float:
        err = len(set(vals).symmetric_difference(set(recovered)))
        if normalize:
            err /= len(db)
        return err


class MSError(Error):
    """Mean Square Error"""

    @classmethod
    def _calc_error(cls, db: RangeDatabase, vals: List[int], recovered: List[int], normalize: bool) -> float:
        if normalize:
            errors = [((recovered[x] - vals[x]) / db.get_max()) ** 2 for x in range(min(len(recovered), len(vals)))]
        else:
            errors = [(recovered[x] - vals[x]) ** 2 for x in range(min(len(recovered), len(vals)))]
        actual_mse = float(sum(errors) / len(errors))

        return actual_mse


class OrderedMSError(MSError):
    """Ordered Mean Squared Error (MAE disregarding order, i.e., of ordered values)"""

    @classmethod
    def _calc_error(cls, db: RangeDatabase, vals: List[int], recovered: List[int], normalize: bool) -> float:
        vals = sorted(vals)
        recovered = list(recovered)
        return super()._calc_error(db, vals, recovered, normalize)


class MAError(Error):
    """Mean Absolute Error"""

    @classmethod
    def _calc_error(cls, db: RangeDatabase, vals: List[int], recovered: List[int], normalize: bool) -> float:
        errors = np.absolute([(recovered[x] - vals[x]) for x in range(min(len(recovered), len(vals)))])
        if normalize:
            errors = [err / db.get_max() for err in errors]
        actual_mae = float(sum(errors) / len(errors))

        return abs(actual_mae)


class OrderedMAError(MAError):
    """Ordered Mean Absolute Error (MAE disregarding order, i.e., of ordered values)"""

    @classmethod
    def _calc_error(cls, db: RangeDatabase, vals: List[int], recovered: List[int], normalize: bool) -> float:
        vals = sorted(vals)
        recovered = list(recovered)
        return super()._calc_error(db, vals, recovered, normalize)


class MaxASymError(Error):
    """Maximum Absolute Symmetric Error as in [GLMP19]
    The error is defined as max(|min{est-val(r), N + 1 -est-val(r)} - symval(r)|) over all records r"""

    @classmethod
    def calc_error(cls, db: RangeDatabase, recovered: List[int], normalize: bool = True) -> float:
        return cls._calc_error(db, db.get_numerical_values(), recovered, normalize)

    @classmethod
    def _calc_error(cls, db: RangeDatabase, vals: List[int], recovered: List[int], normalize: bool) -> float:
        errors = np.absolute([(min(recovered[x], db.get_max() + 1 - recovered[x])
                               - min(vals[x], db.get_max() + 1 - vals[x]))
                              for x in range(min(len(recovered), len(vals)))])
        error = max(errors)
        if normalize:
            error /= db.get_max()

        return error


class MaxABucketError(Error):
    """
    Maximum Absolute Bucket Error as in [GLMP19]
    The error is defined as the max(diameter(Val(recovered buckets))
    """

    @classmethod
    def calc_error(cls, db: RangeDatabase, recovered: List[Any], normalize: bool = True) -> float:
        return cls._calc_error(db, db.get_numerical_values(), recovered, normalize)

    @classmethod
    def _calc_error(cls, db: RangeDatabase, vals: List[int], recovered: List[Any], normalize: bool) -> float:
        error = 0

        buckets = [sub for sub in recovered if len(sub) > 1]  # remove buckets of size 1 since their diam = 0
        if len(buckets) > 0:
            buckets = [[vals[element] for element in bucket] for bucket in buckets]
            # return respective value for recovered ID
            diam = [np.amax([abs(a - b) for a, b in combinations(bucket, 2)]) for bucket in buckets]
            error = max(diam)
        elif len(recovered) < len(vals):
            # Fallback in case the recovered buckets are all singular valued and len(recovered)<R
            # return min(diam(missing values))
            recover = list(chain(*recovered))
            missing = [i for j, i in enumerate(vals) if j not in recover]
            error = np.amin([abs(a - b) for a, b in combinations(missing, 2)])

        if normalize:
            error /= db.get_max()

        return error


class CountSError(Error):
    """Mean Square Error for attacks that only recover the counts/volumes of the database values.
    The error shows the deviation of reconstructed counts and real counts. It is only calculated on values
    occurring in the database."""

    @classmethod
    def _calc_error(cls, db: RangeDatabase, vals: List[int], recovered: List[int], normalize: bool) -> float:
        s_vals = list(sorted(set(vals)))
        vals = list(vals)
        while 0 in recovered:
            recovered.remove(0)

        while len(recovered) < len(s_vals):
            recovered.append(0)

        while len(s_vals) < len(recovered):
            s_vals.append(-1)

        assert len(s_vals) == len(recovered)
        err = 0

        for i in range(len(s_vals)):
            if normalize:
                if vals.count(s_vals[i]) == 0 and recovered[i] == 0:
                    err += 0
                elif recovered[i] == 0:
                    err += 1
                else:
                    err += 1 - (min(vals.count(s_vals[i]), recovered[i]) / max(vals.count(s_vals[i]), recovered[i]))**2
            else:
                err += abs(vals.count(s_vals[i]) - recovered[i])**2

        err /= len(s_vals)

        return err


class CountAError(Error):
    """Mean Absolute Error for attacks that only recover the counts/volumes of the database values.
    The error shows the deviation of reconstructed counts and real counts."""

    @classmethod
    def _calc_error(cls, db: RangeDatabase, vals: List[int], recovered: List[int], normalize: bool) -> float:
        s_vals = list(sorted(set(vals)))
        vals = list(vals)

        while len(recovered) < len(s_vals):
            recovered.append(0)

        while len(s_vals) < len(recovered):
            s_vals.append(0)

        assert len(s_vals) == len(recovered)
        err = 0

        for i in range(len(s_vals)):
            if normalize:
                if vals.count(s_vals[i]) == 0 and recovered[i] == 0:
                    err += 0
                elif recovered[i] == 0:
                    err += 1
                else:
                    err += 1 - min(vals.count(s_vals[i]), recovered[i]) / max(vals.count(s_vals[i]),
                                                                               recovered[i])
            else:
                err += abs(vals.count(s_vals[i]) - recovered[i])

        err /= len(s_vals)

        return err


class SetCountAError(Error):
    """Mean Absolute Error for attacks that only recover the counts/volumes of the database values.
    The error shows the deviation of reconstructed counts and real counts. It is only calculated on values
    occurring in the database."""

    @classmethod
    def _calc_error(cls, db: RangeDatabase, vals: List[int], recovered: List[int], normalize: bool) -> float:
        s_vals = list(sorted(set(vals)))
        vals = list(vals)
        while 0 in recovered:
            recovered.remove(0)

        while len(recovered) < len(s_vals):
            recovered.append(0)

        while len(s_vals) < len(recovered):
            s_vals.append(-1)

        assert len(s_vals) == len(recovered)
        err = 0

        for i in range(len(s_vals)):
            if normalize:
                if vals.count(s_vals[i]) == 0 and recovered[i] == 0:
                    err += 0
                elif recovered[i] == 0:
                    err += 1
                else:
                    err += 1 - min(vals.count(s_vals[i]), recovered[i]) / max(vals.count(s_vals[i]), recovered[i])
            else:
                err += abs(vals.count(s_vals[i]) - recovered[i])

        err /= len(s_vals)

        return err


class CountPartialVolume(Error):
    """For use with GJWpartial, returns percentage of unexplained volumes"""

    @classmethod
    def _calc_error(cls, db: RangeDatabase, vals: List[int], recovered: List[int], normalize: bool) -> float:
        vals = list(vals)
        unique_vals = range(db.get_min(), db.get_max() + 1)

        counts = [vals.count(x) for x in unique_vals]
        vol = db.get_n()

        idx = cls.subfinder(counts, recovered)
        assert len(idx) > 0

        recovered_vol = sum(recovered)

        return 1 - recovered_vol / vol

    @staticmethod
    def subfinder(mylist, pattern):
        # https://stackoverflow.com/questions/10106901/elegant-find-sub-list-in-list
        matches = []
        for i in range(len(mylist)):
            if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
                matches.append(i)
        return matches

