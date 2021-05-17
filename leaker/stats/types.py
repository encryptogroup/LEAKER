"""
For License information see the LICENSE file.

Authors: Amos Treiber

"""
import os
from abc import abstractmethod, ABC
from collections import Counter, namedtuple
from logging import getLogger
from typing import List, Type, Union, Optional, NamedTuple, Iterator, Tuple

from ..api import Dataset, KeywordQueryLog, RangeDatabase, RangeQuerySpace, RangeQueryLog
from ..api.constants import FIGURE_DIRECTORY
from ..attack import QueryLogRangeQuerySpace, UserQueryLogRangeQuerySpace
from ..extension import SelectivityExtension
from ..plotting.statistics import FrequencyPlotter, SelectivityPlotter, PowerLawFittingResults, HeatMapPlotter, \
    RangesPlotter

log = getLogger(__name__)


class StatisticsTypes(ABC):
    """
    A class that produces the desired type of statistics for a StatisticsCase
    using the StatisticsCase's data and plots them in corresponding StatisticsPlotters.
    Should only be used by the Statistics class.

    Parameters
    ----------
    file_desciption: str
        additional identifier used in filenames of resulting plot figures
    """

    __file_description: str

    def __init__(self, file_description: str = ""):
        self.__file_description = file_description

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError

    @classmethod
    def required_input_data(cls) -> Iterator[Union[Type[KeywordQueryLog], Type[Dataset]]]:
        """
        Returns
        ----------
        input_data: Iterator[Union[Type[KeywordQueryLog], Type[Dataset]]]
            All typed of input data necessary to compute this statistic (these instances MUST be supplied to
            offer_data())
        """
        raise NotImplementedError

    def file_description(self) -> str:
        return self.__file_description

    @staticmethod
    def figure_filename(name: str, description: Optional[str] = None) -> str:
        """
        Creates filename FIGURE_DIRECTORY + {name}[_description].png

        Parameters
        ----------
        name: str
            the name to display
        description : Optional[str]
            further qualifier of the figure instance

        Returns
        -------
        filename : str
            the figure filename
        """
        desc = ""
        if description is not None and description != "":
            desc = "_" + description
        if not os.path.exists(FIGURE_DIRECTORY):
            os.makedirs(FIGURE_DIRECTORY)
        return FIGURE_DIRECTORY + name + "_" + desc + ".png"

    @abstractmethod
    def offer_data(self, data: Union[Union[Dataset, RangeDatabase, KeywordQueryLog, RangeQueryLog],
                                     List[Union[Dataset, RangeDatabase, KeywordQueryLog, RangeQueryLog]]]) -> None:
        """
        Passes data to the StatisticsType, which computes the statistic on the data offered here and additionally
        stores the offered data to compute the statistic on all aggregated data when gather() is called.

        Parameters
        ----------
        data: Union[Union[Dataset, KeywordQueryLog, RangeQueryLog], List[Union[Dataset, KeywordQueryLog,
        RangeQueryLog]]]
            One or multiple datasets or querylogs supplied for this iteration of statistics and the aggregated statistic
        """
        raise NotImplementedError

    @abstractmethod
    def gather(self) -> Union[None, NamedTuple]:
        """
        Main function to gather aggregated statistics offered to this statistics type. Write data to plotters and
        return all computed data as a named tuple.

        Returns
        ----------
        statistics_result: Union[None, NamedTuple]
            If statistics besides plots are gathered, return them in a corresponding named tuple
        """
        raise NotImplementedError


QueryDistributionResults = namedtuple("QueryDistributionResults",
                                      ["overall_exponent", "iteration_exponents", "user_exponents"])


class QueryDistribution(StatisticsTypes):
    """
    A type of statistics that gathers the query distribution of QueryLogs. It produces a plot of frequencies of supplied
    query frequencies and fits a curve to it. Outputs graphs for all users and aggregated users as well as mean fitted
    parameters.

    Parameters
    ----------
    max_user_plot: int
        the maximum amount of most active users that a dedicated plot should be created for
    """
    __max_user_plot: int

    __plotter: Union[None, FrequencyPlotter]

    __user_exponents: List[PowerLawFittingResults]
    __iteration_exponents: List[PowerLawFittingResults]

    def __init__(self, file_description: str = "", max_user_plot: int = 15):
        super(QueryDistribution, self).__init__(file_description)

        self.__max_user_plot = max_user_plot

        self.__plotter = None

        self.__user_exponents = []
        self.__iteration_exponents = []

    @classmethod
    def name(cls) -> str:
        return "Query Distribution"

    @classmethod
    def required_input_data(cls) -> Iterator[Type[KeywordQueryLog]]:
        yield KeywordQueryLog

    def offer_data(self, data: Union[KeywordQueryLog, List[KeywordQueryLog]]) -> None:
        if isinstance(data, KeywordQueryLog):
            query_logs = [data]
        else:
            query_logs = data

        for query_log in query_logs:
            log.info(f"Computing query distribution on '{query_log.name()}")
            current_plotter = self.new_plotter(query_log.name(), query_log.name())
            if self.__plotter is None:
                self.__plotter = self.new_plotter(f"{query_log.name()}_aggregated", query_log.name())

            """Freq Distribution over all received offer_data() calls"""
            self.__plotter.offer_keywords(query_log.keywords_list())

            """Freq Distribution over current call"""
            self.__iteration_exponents.append(current_plotter(query_log.keywords_list()))

            log.debug(f"Computing query distribution on {len(query_log.user_ids())} users of '{query_log.name()}'")
            """Freq Distribution over individual users"""
            for i, user_id in enumerate(query_log.user_ids()):
                log.debug(f"Computing Query Distribution on user {i}.")
                user_plotter = self.new_plotter(f"{query_log.name()}_u_{user_id}", f"{query_log.name()} User {user_id}")
                user_plotter.offer_keywords(list(query_log(user_id)))
                self.__user_exponents.append(user_plotter.fitted_exponent())
                if i < self.__max_user_plot:
                    user_plotter.plot()

            log.info(f"Done computing query distribution on '{query_log.name()}'")

    def new_plotter(self, filestr: str, title_name: str) -> FrequencyPlotter:
        return FrequencyPlotter(StatisticsTypes.figure_filename(f"qdist_{filestr}", self.file_description()),
                                title=f"Query Distribution of {title_name}")

    def gather(self) -> QueryDistributionResults:
        if len(self.__iteration_exponents) == 1:
            """We only have one overall plot and no iterations"""
            overall_exponent = self.__iteration_exponents[0]
        else:
            log.info(f"Gathering aggregated statistics of all query logs...")
            overall_exponent = self.__plotter()
        return QueryDistributionResults(overall_exponent=overall_exponent,
                                        iteration_exponents=self.__iteration_exponents,
                                        user_exponents=self.__user_exponents)


QuerySelectivityDistributionResults = namedtuple("QuerySelectivityDistributionResults",
                                                 ["overall_coefficient", "overall_exponent", "iteration_coefficients",
                                                  "iteration_exponents", "user_coefficients", "user_exponents"])


class QuerySelectivityDistribution(StatisticsTypes):
    """
    A type of statistics that gathers the selectivities of all queries in a query log according to provided datasets.
    Outputs graphs displaying correlation between query frequency and their selectivities on the datasets
    for users and aggregated users.

    Parameters
    ----------
    max_user_plot: int
        the maximum amount of most active users that a dedicated plot should be created for
    """
    __max_user_plot: int

    __selectivity_plotter: Union[None, SelectivityPlotter]
    __frequency_plotter: Union[None, FrequencyPlotter]

    __user_coefficients: List[Union[None, float]]
    __iteration_coefficients: List[Union[None, float]]
    __user_exponents: List[Union[None, PowerLawFittingResults]]
    __iteration_exponents: List[Union[None, PowerLawFittingResults]]

    __plot_overall: bool

    def __init__(self, file_description: str = "", max_user_plot: int = 15):
        super(QuerySelectivityDistribution, self).__init__(file_description)

        self.__max_user_plot = max_user_plot

        self.__selectivity_plotter = None
        self.__frequency_plotter = None

        self.__plot_overall = False

        self.__user_coefficients = []
        self.__iteration_coefficients = []
        self.__user_exponents = []
        self.__iteration_exponents = []

    @classmethod
    def name(cls) -> str:
        return "Query Selectivity Distribution"

    @classmethod
    def required_input_data(cls) -> Iterator[Union[Type[KeywordQueryLog], Type[Dataset]]]:
        yield from [KeywordQueryLog, Dataset]

    @classmethod
    def __data_error(cls) -> str:
        return f"Data offered to {cls.name()} has to be a list of at least one KeywordQueryLog and one Dataset"

    def offer_data(self, data: List[Union[Dataset, KeywordQueryLog]]) -> None:
        query_logs: List[KeywordQueryLog] = []
        datasets: List[Dataset] = []
        if not isinstance(data, List):
            raise AttributeError(self.__data_error())
        for d in data:
            if isinstance(d, KeywordQueryLog):
                query_logs.append(d)
            elif isinstance(d, Dataset):
                datasets.append(d)
        if len(query_logs) < 1 or len(datasets) < 1:
            raise AttributeError(self.__data_error())

        """See if we have to use __plotter to gather aggregated statistics on more than one query_log dataset pair"""
        self.__plot_overall = self.__selectivity_plotter is not None or len(query_logs) > 1 or len(datasets) > 1

        for query_log in query_logs:
            for dataset in datasets:
                log.info(f"Computing queries from '{query_log.name()}' on '{dataset.name()}'")
                if not dataset.has_extension(SelectivityExtension):
                    dataset.extend_with(SelectivityExtension)
                sel: SelectivityExtension = dataset.get_extension(SelectivityExtension)

                filestr: str = f"{query_log.name()}_{dataset.name()}"
                name: str = f"{query_log.name()} on {dataset.name()}"
                current_selectivity_plotter = self.new_selectivity_plotter(filestr, name)
                current_frequency_plotter = self.new_frequency_plotter(filestr, name)
                if self.__selectivity_plotter is None:
                    self.__selectivity_plotter = self.new_selectivity_plotter(filestr + "_aggregated", name)
                    self.__frequency_plotter = self.new_frequency_plotter(filestr + "_aggregated", name)

                """Distribution over all received calls"""
                counts: Counter[str] = Counter(query_log.keywords_list())
                log.debug(f"Computing selectivities of {len(query_log.keywords())} "
                          f"keywords'{query_log.name()}' on '{dataset.name()}'")
                current_data: List[Tuple[int, int]] = [(sel.selectivity(kw), freq) for kw, freq in counts.most_common()
                                                       if sel.selectivity(kw) > 0]
                """We have to repeat the selectivities here to get them depending on their occurrences in the qlog"""
                current_selectivities: List[int] = [selectivity for selectivity, freq in current_data
                                                    for _ in range(freq)]
                self.__selectivity_plotter.offer_data(current_data)
                """Plot selectivities of just the observed queries"""
                self.__frequency_plotter.offer_occurrences(current_selectivities)

                """Freq Distribution over current call"""
                log.debug(f"Fitting curves of '{query_log.name()}' on '{dataset.name()}'")
                self.__iteration_coefficients.append(current_selectivity_plotter(current_data))
                current_frequency_plotter.offer_occurrences(current_selectivities)
                current_frequency_plotter.plot()
                self.__iteration_exponents.append(current_frequency_plotter.fitted_exponent())

                log.debug(f"Computing on {len(query_log.user_ids())} individual users of '{query_log.name()}'...")

                """Freq Distribution over individual users"""
                for i, user_id in enumerate(query_log.user_ids()):
                    log.debug(f"Computing Query Distribution on user {i}.")
                    user_selectivity_plotter = self.new_selectivity_plotter(f"{filestr}_u_{user_id}",
                                                                            f"{name} User {user_id}")
                    user_frequency_plotter = self.new_frequency_plotter(f"{filestr}_u_{user_id}",
                                                                        f"{name} User {user_id}")
                    counts: Counter[str] = Counter(query_log(user_id))
                    data: List[Tuple[int, int]] = \
                        [(sel.selectivity(kw), freq) for kw, freq in counts.most_common() if sel.selectivity(kw) > 0]
                    sels: List[int] = [selectivity for selectivity, freq in data for _ in range(freq)]
                    user_selectivity_plotter.offer_data(data)
                    user_frequency_plotter.offer_occurrences(sels)
                    if i < self.__max_user_plot:
                        user_selectivity_plotter.plot()
                        user_frequency_plotter.plot()

                    self.__user_coefficients.append(user_selectivity_plotter.correlation_coefficient())
                    self.__user_exponents.append(user_frequency_plotter.fitted_exponent())

                log.info(f"Done computing queries from '{query_log.name()}' on '{dataset.name()}'")

    def offer_selectivities(self, query_log: KeywordQueryLog):
        """
        A special case of offer_data that can be used to compute QuerySelectivityDistributionResults on a QueryLog
        that does not have a corresponding dataset but has recorded selectivities. The selectivites need to be supplied
        via keywords_list() of the QueryLog.

        Parameters
        ----------
        query_log : KeywordQueryLog
            the query log to compute on. query_log.keywords_list() has to yield selectivities instead of keywords!
        """
        log.info(f"Computing on '{query_log.name()}'")

        filestr: str = f"{query_log.name()}_{query_log.name()}"
        name: str = f"{query_log.name()} on {query_log.name()}"
        current_frequency_plotter = self.new_frequency_plotter(filestr, name)
        if self.__frequency_plotter is None:
            self.__frequency_plotter = self.new_frequency_plotter(filestr + "_aggregated", name)

        """Distribution over all received calls"""
        log.debug(f"Computing selectivities of {len(query_log.keywords())} "
                  f"keywords'{query_log.name()}' on '{query_log.name()}'")
        """We have to repeat the selectivities here to get them depending on their occurrences in the qlog"""
        current_selectivities: List[int] = [int(sel) for sel in query_log.keywords_list()]
        """Plot selectivities of just the observed queries"""
        self.__frequency_plotter.offer_occurrences(current_selectivities)

        """Freq Distribution over current call"""
        log.debug(f"Fitting curves of '{query_log.name()}'")
        current_frequency_plotter.offer_occurrences(current_selectivities)
        current_frequency_plotter.plot()
        self.__iteration_exponents.append(current_frequency_plotter.fitted_exponent())
        self.__iteration_coefficients.append(None)

        log.debug(f"Computing on {len(query_log.user_ids())} individual users of '{query_log.name()}'...")

        """Freq Distribution over individual users"""
        for i, user_id in enumerate(query_log.user_ids()):
            log.debug(f"Computing Query Distribution on user {i}.")
            user_frequency_plotter = self.new_frequency_plotter(f"{filestr}_u_{user_id}",
                                                                f"{name} User {user_id}")
            sels: List[int] = [int(sel) for sel in query_log.keywords_list(user_id)]
            user_frequency_plotter.offer_occurrences(sels)
            if i < self.__max_user_plot:
                user_frequency_plotter.plot()

            self.__user_exponents.append(user_frequency_plotter.fitted_exponent())
            self.__user_coefficients.append(None)

        log.info(f"Done computing queries from '{query_log.name()}' on '{query_log.name()}'")

    def new_selectivity_plotter(self, filestr: str, title_name: str) -> SelectivityPlotter:
        return SelectivityPlotter(StatisticsTypes.figure_filename(f"qseldist_{filestr}", self.file_description()),
                                  title=f"Query-Selectivity Distribution of {title_name}")

    def new_frequency_plotter(self, filestr: str, title_name: str) -> FrequencyPlotter:
        return FrequencyPlotter(StatisticsTypes.figure_filename(f"qseldistfreq_{filestr}", self.file_description()),
                                title=f"Query-Selectivity Distribution of {title_name}")

    def gather(self) -> QuerySelectivityDistributionResults:
        if self.__plot_overall:
            log.info(f"Gathering aggregated statistics of all data pairs...")
            overall_coefficient = self.__selectivity_plotter()
            overall_exponent = self.__frequency_plotter()
        else:
            overall_coefficient = self.__iteration_coefficients[0]
            overall_exponent = self.__iteration_exponents[0]

        return QuerySelectivityDistributionResults(overall_coefficient=overall_coefficient,
                                                   overall_exponent=overall_exponent,
                                                   iteration_coefficients=self.__iteration_coefficients,
                                                   iteration_exponents=self.__iteration_exponents,
                                                   user_coefficients=self.__user_coefficients,
                                                   user_exponents=self.__user_exponents)


SelectivityDistributionResults = \
    namedtuple("SelectivityDistributionResults", ["overall_exponent", "iteration_exponents"])


class SelectivityDistribution(StatisticsTypes):
    """
    A type of statistics that gathers the selectivity distribution of a dataset or range DB. It produces a plot of
    frequencies of supplied keywords/values and fits a curve to it. Outputs graphs for all data as well as mean
    fitted parameters.
    """

    __plotter: Union[None, FrequencyPlotter]
    __is_range: bool
    __iteration_exponents: List[PowerLawFittingResults]

    def __init__(self, file_description: str = ""):
        super(SelectivityDistribution, self).__init__(file_description)

        self.__plotter = None

        self.__iteration_exponents = []

    @classmethod
    def name(cls) -> str:
        return "Selectivity Distribution"

    @classmethod
    def required_input_data(cls) -> Iterator[Type[Union[Dataset, RangeDatabase]]]:
        for req in [Dataset, RangeDatabase]:
            yield req

    def offer_data(self, data: Union[Union[Dataset, RangeDatabase], List[Union[Dataset, RangeDatabase]]]) -> None:
        if isinstance(data, Dataset) or isinstance(data, RangeDatabase):
            datasets = [data]
        else:
            datasets = data

        for dataset in datasets:
            log.info(f"Computing selectivity distribution of '{dataset.name()}'")
            current_plotter = self.new_plotter(dataset.name(), dataset.name())
            if self.__plotter is None:
                self.__plotter = self.new_plotter(f"{dataset.name()}_aggregated", dataset.name())

            if isinstance(dataset, Dataset):
                keywords = [kw for kw in dataset.keywords() for _ in range(dataset.selectivity(kw))]
            else:
                keywords = [str(val) for val in dataset.get_numerical_values()]

            """Freq Distribution over all received offer_data() calls"""
            self.__plotter.offer_keywords(keywords)

            """Freq Distribution over current call"""
            self.__iteration_exponents.append(current_plotter(keywords))

            log.info(f"Done computing selectivity distribution of '{dataset.name()}'")

    def new_plotter(self, filestr: str, title_name: str) -> FrequencyPlotter:
        return FrequencyPlotter(StatisticsTypes.figure_filename(f"seldist_{filestr}", self.file_description()),
                                title=f"Selectivity Distribution of {title_name}")

    def gather(self) -> SelectivityDistributionResults:
        if len(self.__iteration_exponents) == 1:
            """We only have one overall plot and no iterations"""
            overall_exponent = self.__iteration_exponents[0]
        else:
            log.info(f"Gathering aggregated statistics of all datasets...")
            overall_exponent = self.__plotter()
        return SelectivityDistributionResults(overall_exponent=overall_exponent,
                                              iteration_exponents=self.__iteration_exponents)


class RangeQueryDistribution(StatisticsTypes):
    """
    A type of statistics that gathers the query distribution of QueryLogs. It produces a heatmap similar to [KPT20] as
    well as plots of QueryDistribution for the range case (regarding each (a,b) range query as a 'keyword').

    Parameters
    ----------
    max_user_plot: int
        the maximum amount of most active users that a dedicated plot should be created for
    """
    __max_user_plot: int

    def __init__(self, file_description: str = "", max_user_plot: int = 15):
        super(RangeQueryDistribution, self).__init__(file_description)

        self.__max_user_plot = max_user_plot

    @classmethod
    def name(cls) -> str:
        return "Query Distribution"

    @classmethod
    def required_input_data(cls) -> Iterator[Union[Type[RangeQuerySpace], Type[RangeQueryLog], Type[RangeDatabase]]]:
        yield from [RangeQuerySpace, RangeQueryLog, RangeDatabase]

    @classmethod
    def __data_error(cls) -> str:
        return f"Data offered to {cls.name()} has to be a list of at least one RangeQueryLog/RangeQueryDistribution " \
               f"and one RangeDatabase"

    def offer_data(self, data: List[Union[RangeDatabase, RangeQueryLog, RangeQuerySpace]]) -> None:
        query_data: Union[List[RangeQueryLog], List[RangeQuerySpace]] = []
        dbs: List[RangeDatabase] = []
        if not isinstance(data, List):
            raise AttributeError(self.__data_error())
        for d in data:
            if isinstance(d, RangeQueryLog) or isinstance(d, RangeQuerySpace):
                query_data.append(d)
            elif isinstance(d, RangeDatabase):
                dbs.append(d)
            else:
                log.warning(f"Unrecognized type {type(d)}.")
        if len(query_data) < 1 or len(dbs) < 1:
            raise AttributeError(self.__data_error())

        for query_data in query_data:
            for db in dbs:
                max_val = db.get_max()
                max_xval = (max_val*(max_val + 1))//2
                if isinstance(query_data, RangeQueryLog):
                    qsp = QueryLogRangeQuerySpace(db, qlog=query_data)
                    uqsp = UserQueryLogRangeQuerySpace(db, qlog=query_data)
                    name = query_data.name()
                else:
                    qsp = uqsp = query_data
                    name = query_data.__class__.__name__

                log.debug(f"Computing query distribution on '{name}' with {db.name()}")

                current_hm_plotter = self.new_hm_plotter(max_val, name, name)
                current_sel_plotter = self.new_freq_plotter(name, name, "Selectivity")
                current_freq_plotter = self.new_freq_plotter(name, name, "Frequency", max_xval, True)
                current_bound_plotter = self.new_freq_plotter(name, name, "Bound")
                current_ranges_plotter = self.new_ranges_plotter(max_val, name, name)

                """Query Distribution over all received offer_data() calls"""
                queries = [q for u_q in qsp.select(-1) for q in u_q]
                current_hm_plotter.offer_data(queries)
                current_hm_plotter.plot()
                current_freq_plotter.offer_keywords([str(query) for query in queries])
                current_freq_plotter.plot()
                current_sel_plotter.offer_occurrences([len(db.query(query)) for query in queries])
                current_sel_plotter.plot()
                current_bound_plotter.offer_occurrences([upper - lower + 1 for lower, upper in queries])
                current_bound_plotter.plot()
                current_ranges_plotter.offer_data(queries)
                current_ranges_plotter.plot()

                """Distribution over individual users"""
                for i, queries in enumerate(uqsp.select(-1)):
                    log.debug(f"Computing Query Distribution on user {i}.")
                    user_hm_plotter = self.new_hm_plotter(max_val, f"{name}_u_{i}", f"nameUser {i}")
                    user_hm_plotter.offer_data(queries)

                    user_sel_plotter = self.new_freq_plotter(f"{name}_u_{i}", f"{name} User {i}", "Selectivity")
                    user_freq_plotter = self.new_freq_plotter(f"{name}_u_{i}", f"{name} User {i}", "Frequency",
                                                              max_xval, True)
                    user_bound_plotter = self.new_freq_plotter(f"{name}_u_{i}", f"{name} User {i}", "Bound")
                    user_ranges_plotter = self.new_ranges_plotter(max_val, f"{name}_u_{i}", f"{name} User {i}")

                    user_sel_plotter.offer_occurrences([len(db.query(query)) for query in queries])
                    user_freq_plotter.offer_keywords([str(query) for query in queries])
                    user_bound_plotter.offer_occurrences([upper - lower + 1 for lower, upper in queries])
                    user_ranges_plotter.offer_data(queries)

                    if i < self.__max_user_plot:
                        user_hm_plotter.plot()
                        user_sel_plotter.plot()
                        user_freq_plotter.plot()
                        user_bound_plotter.plot()
                        user_ranges_plotter.plot()

                log.debug(f"Computed query distribution on {i+1} users of '{name}' with {db.name()}")

            log.info(f"Done computing query distribution on '{name}'")

    def new_hm_plotter(self, max_val: int, filestr: str, title_name: str) -> HeatMapPlotter:
        return HeatMapPlotter(max_val, StatisticsTypes.figure_filename(f"qdist_hm_{filestr}", self.file_description()),
                              title=f"Query Distribution of {title_name}")

    def new_freq_plotter(self, filestr: str, title_name: str, y_info: str, max_xval: int = None,
                         normalize: bool = False) -> FrequencyPlotter:
        return FrequencyPlotter(StatisticsTypes.figure_filename(f"qdist_{y_info.lower()}_{filestr}",
                                                                self.file_description()),
                                title=f"Query-{y_info} Distribution of {title_name}", max_xval=max_xval,
                                normalize=normalize)

    def new_ranges_plotter(self, max_val: int, filestr: str, title_name: str) -> RangesPlotter:
        return RangesPlotter(max_val, StatisticsTypes.figure_filename(f"qdist_ranges_{filestr}",
                                                                      self.file_description()),
                             title=f"Query-Range Distribution of {title_name}")

    def gather(self) -> None:
        """All necessary computations already occurred in-place for this type."""
        pass
