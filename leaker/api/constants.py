"""
For License information see the LICENSE file.

Authors: Johannes Leupold, Amos Treiber

"""
from enum import Enum

WHOOSH_INDEX_DIRECTORY: str = "data/whoosh/"

PICKLE_DIRECTORY: str = "data/pickle/"

RANGE_PICKLE_ID: str = "idx_values"

RANGE_QLOG_PICKLE_ID: str = "idx_queries"

FIGURE_DIRECTORY: str = "data/figures/"

WRITING_INTERVAL: int = 5000000

SQL_WRITING_INTERVAL: int = 50000

MIN_USER_QUERYLOG_ACTIVITY: int = 2000

COMPILE_TIMEOUT: int = 1200  # in seconds

PYTHON_DIST_PACKAGES_DIRECTORY = "/usr/lib/python3/dist-packages/"

MYSQL_IDENTIFIER = "leaker"  # used to identify MySQL databases used for leaker

MYSQL_USER_NAME = "leaker-user"

MYSQL_USER_PASSWORD = "abcdefg"


class AbortException(Exception):
    pass


class Selectivity(Enum):
    """
    Possible values for the selectivity of keywords.

    Independent - the query space is populated uniformly at random from the keywords in the data set
    High - the query space is populated with the highest selectivity keywords in the data set
    Low - the query space is populated with the lowest selectivity keywords in the data set
    PseudoLow - the query space is populated with the lowest selectivity keywords with a selectivity of at least 10
    PseudoLowTwo - the query space is populated with the lowest selectivity keywords with a selectivity of at least 2
                    (only for relational database)
    IndependentNotOne - the query space is populated uniformly at random from the keywords in the data set with selectivity >= 2 
                    (only for relational database)
    PseudoLowFive - the query space is populated with the lowest selectivity keywords with a selectivity of at least 5
                    (only for relational database)
    HighExceptTopOneHundred - the query space is populated with the highest selectivity keywords in the data set, 
                    except the 100 keywords with the highest selectivity (only for relational database)
    """
    Independent = -1
    High = 0
    Low = 1
    PseudoLow = 2
    PseudoLowTwo = 3
    IndependentNotOne = 4
    PseudoLowFive = 5
    HighExceptTopOneHundred = 6
