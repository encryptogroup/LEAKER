import logging
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from operator import itemgetter
import tikzplotlib
from math import ceil

from leaker.pattern.cooccurrence import CoOccurrence
from leaker.api import Dataset, Selectivity
from leaker.evaluation import SampledDatasetSampler
from leaker.extension import CoOccurrenceExtension
from ..api.constants import FIGURE_DIRECTORY


f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}', style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('statistical_closeness.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.INFO)

log = logging.getLogger(__name__)

class StatisticalCloseness():
    """
    An evaluation class to compute and plot the statistical closeness of datasets.

    Parameters
    ----------
    min_n: int
        the minimum number of keywords to evaluate for a criterion
    min_n: int
        the maximum number of keywords to evaluate for a criterion
        if larger than the number of keywords in the smallest datast, it will be reduced
    stepsize: int
        stepsize for computing criterion CR 1 and CR 2
    co_sim: bool
        if True, the co-occurrence similarity is computed and
        compared to the statistical  closeness metric
        WARNING: will take much longer
    out_file: str
        name of the output files to store the plots
    """
    __min_n: int
    __max_n: int
    __stepsize: int
    __co_sim: bool
    __out_file: str

    def __init__(self, min_n:int = 10, max_n: int = 10000, stepsize: int = 10, co_sim: bool = False, out_file: str = None) -> None:
        self.__min_n = min_n
        self.__max_n = max_n
        self.__stepsize = stepsize
        self.__co_sim = co_sim
        self.__out_file = out_file
        if self.__out_file is not None:
            if '.' in self.__out_file:
                self.__out_file = self.__out_file.split('.')[0]
            self.__out_file = FIGURE_DIRECTORY + self.__out_file
            if not os.path.exists(FIGURE_DIRECTORY):
                os.makedirs(FIGURE_DIRECTORY)

    def _get_kw_list(self, ds:Dataset):
        kw_lst = ds.keyword_counts().most_common()
        return kw_lst

    def _process_data(self, data:list[tuple[Dataset]]):
        processed = []
        datasets = []
        names = []
        for tup in data:
            if self.__co_sim:
                tup[0].extend_with(CoOccurrenceExtension)
            if isinstance(tup[1],Dataset):
                if self.__co_sim:
                    tup[1].extend_with(CoOccurrenceExtension)
                train = tup[0]
                test = tup[1]
                names.append(train.name()+" - "+test.name())
                if len(train) < len(test):
                    test = test.restrict_keyword_size(len(train),selectivity=Selectivity.High)
                elif len(test) < len(train):
                    train  = train.restrict_keyword_size(len(test),selectivity=Selectivity.High)
                processed.append((self._get_kw_list(train),self._get_kw_list(test)))
                datasets.append((train,test))
            elif isinstance(tup[1],float):
                dataset_sampler = SampledDatasetSampler(kdr_samples=[tup[1]])
                [(train,rate,test)] = list(dataset_sampler.sample([tup[0]]))
                processed.append((self._get_kw_list(train),self._get_kw_list(test)))
                names.append(tup[0].name()+" sampled "+str(rate))
                datasets.append((train,test))
            else:
                dataset_sampler = SampledDatasetSampler(kdr_samples=tup[1])
                result = list(dataset_sampler.sample([tup[0]]))
                for (train,rate,test) in result:
                    processed.append((self._get_kw_list(train),self._get_kw_list(test)))
                    names.append(tup[0].name()+" sampled "+str(rate))
                    datasets.append((train,test))
        return processed, names, datasets


    def _get_n_most_common(self, dataset:list, n:int, isSet: bool=True) -> set:
        keywords = [kw[0] for kw in dataset]
        if n > len(keywords):
            log.warning(f"{n} is larger than the list's length of {len(keywords)}")
            if isSet:
                return set(keywords)
            return keywords
        if isSet:
            return set(keywords[:n])
        return keywords[:n]


    def _get_keywords(self, dataset:list) -> set:
        keywords = [kw[0] for kw in dataset]
        return set(keywords)


    def _comp_CR1(self, D1:list, D2:list):
        cr1 = []
        for n in range(self.__min_n,self.__max_n,self.__stepsize):
            cr1.append(len(self._get_n_most_common(D1,n).intersection(self._get_n_most_common(D2,n)))/n)
        
        return cr1

    def _comp_CR2(self, D1:list, D2:list):
        cr2 = []
        for n in range(self.__min_n,self.__max_n,self.__stepsize):
            cr2.append(len(self._get_n_most_common(D1,n).intersection(self._get_keywords(D2)))/n)
        
        return cr2

    def _comp_CR3(self, D1:list, D2:list):
        rlen_deltas = []
        train_dict = dict(D1)
        test_dict = dict(D2)
        keywords = self._get_n_most_common(D1,self.__max_n,False)
        for kw in keywords:
            train = train_dict[kw]
            test = 0
            if kw in test_dict:
                test = test_dict[kw]
            rlen_deltas.append(1-abs(test-train)/max(test,train))
        return rlen_deltas

    def _mean_err(self, input:list):
        means = []
        errors = []
        for elem in input:
            means.append(np.mean(elem))
            errors.append(np.std(elem))
        return means, errors

    def _error_propagation(self, vals:np.ndarray, err:np.ndarray):
        n = len(vals)
        res = 0
        for i in range(n):
            res += np.sum(np.delete(vals,i))**2*err[i]**2
        return 1/n*np.sqrt(res)
    
    def _co_sim(self, train_db: Dataset, test_db: Dataset, kw_list:list):
        dim = len(kw_list)
        delta = np.zeros((dim,dim))
        epsilon = np.zeros((dim,dim))
        K_overlap = 0
        test_kw = test_db.keywords()
        train_kw = train_db.keywords()
        co = CoOccurrence()
        cooc_train = np.array(co.leak(train_db,kw_list))/len(train_db)
        cooc_test = np.array(co.leak(test_db,kw_list))/len(test_db)
        for i in range(dim):
            kw_i = kw_list[i]
            if not kw_i in test_kw:
                continue
            if not kw_i in train_kw:
                continue
            K_overlap += 1
            for j in range(dim):
                kw_j = kw_list[j]
                if not kw_j in test_kw:
                    continue
                if not kw_j in train_kw:
                    continue
                delta[i,j] = np.square(cooc_train[i,j]-cooc_test[i,j])
                epsilon[i,j] = 1
        delta_tot = np.sum(delta)
        epsilon_tot = np.sum(epsilon)
        co_sim = (1 - delta_tot/epsilon_tot)*(K_overlap/dim)
        return co_sim

    def _plot_metric(self,metric_val:list, metric_stat:list, metric_sys:list, names: list, co_sim: list = []):
        sorted_idx = np.argsort(metric_val)
        fig,ax = plt.subplots(1)

        errorbar(list(range(len(metric_val))),list(itemgetter(*sorted_idx)(metric_val)),list(itemgetter(*sorted_idx)(metric_sys)),color='orange',elinewidth=30,markersize=0,linestyle='',alpha=0.5)
        errorbar(list(range(len(metric_val))),list(itemgetter(*sorted_idx)(metric_val)),list(itemgetter(*sorted_idx)(metric_stat)),color='darkblue',capsize=3,elinewidth=2,linestyle='',marker='s', label="statistical closeness metric")
        if len(co_sim) == len(metric_val):
            plot(list(range(len(metric_val))),list(itemgetter(*sorted_idx)(co_sim)),linestyle='', marker='o', color='green', label="co-occurrence similarity")
        labels = list(itemgetter(*sorted_idx)(names))
        legend()
        ylabel("statistical closeness")
        xticks(list(range(len(metric_val))),labels)

        if self.__out_file is not None:
            out_name = self.__out_file + '_metric'
            savefig(f"{out_name}.png")
            tikzplotlib.save(f"{out_name}.tikz")
        else:
            show()
    
    def _plot_cr3(self, val:list, names: list):
        N=500
        if self.__max_n <= 1000:
            N = ceil(self.__max_n/10)
        fig,ax = plt.subplots(1)
        for i in range(len(val)):
            plot(np.convolve(list(range(self.__max_n)), np.ones(N)/N, mode='valid'),np.convolve(val[i], np.ones(N)/N, mode='valid'), linewidth=2, label=names[i])
        
        ylim((0,1))
        xlabel("keyword id")
        ylabel("CR 3")
        legend()
        if self.__out_file is not None:
            out_name = self.__out_file + '_cr3'
            savefig(f"{out_name}.png")
            tikzplotlib.save(f"{out_name}.tikz")
        else:
            show()

    def _plot_cr1_2(self, val:list, names: list, label_name: str):
        ns = list(range(self.__min_n,self.__max_n,self.__stepsize))
        fig,ax = plt.subplots(1)
        for i in range(len(val)):
            plot(ns,val[i],label=names[i])
        xlabel("# keywords")
        ylabel(label_name)
        legend()

        if self.__out_file is not None:
            out_name = self.__out_file + '_'+label_name.replace(" ","").lower()
            savefig(f"{out_name}.png")
            tikzplotlib.save(f"{out_name}.tikz")
        else:
            show()

    def compute_metric(self,input_data:list[tuple[Dataset]]):
        """
        Computes and plots the metric on the input data.

        Parameters
        ----------
        input_data: list[tuple[Dataset]]
            A list containing tuples of either
                (1) two datasets
                (2) a dataset and a float
                (3) a dataset and a list of floats

        Output:
        ----------
            The statistical closeness of
                (1) the two datasets
                (2) the statistical closeness of the dataset sampled to the given rate
                (3) the statistical closeness of the dataset sampled to all rates given in the list
    """
        data, names, datasets = self._process_data(input_data)
        min_ds_len = min(list(map(len,[kw for ds in data for kw in ds])))
        if min_ds_len < self.__max_n:
            log.warning(f"Reducing maximum number of keywords to smallest dataset size of {min_ds_len}.")
            self.__max_n = min_ds_len
        results_cr1 = []
        results_cr2 = []
        results_cr3 = []
        co_sims = []
        for tup in data:
            results_cr1.append(self._comp_CR1(tup[0],tup[1]))
            results_cr2.append(self._comp_CR2(tup[0],tup[1]))
            results_cr3.append(self._comp_CR3(tup[0],tup[1]))
        means_cr1, errors_cr1 = self._mean_err(results_cr1)
        means_cr2, errors_cr2 = self._mean_err(results_cr2)
        means_cr3, errors_cr3 = self._mean_err(results_cr3)

        means = np.array(list(zip(means_cr1,means_cr2,means_cr3)))
        errors = np.array(list(zip(errors_cr1,errors_cr2,errors_cr3)))
        scm_val = np.mean(means,axis=1)
        scm_err_sys = np.std(means,axis=1)
        scm_err_stat = []

        for idx in range(len(means)):
            scm_err_stat.append(self._error_propagation(means[idx],errors[idx]))

        print("Metric")
        for i in range(len(scm_val)):
            print(names[i],": ",scm_val[i],"+/-",scm_err_stat[i],"+/-",scm_err_sys[i])
        
        print("CR 1")
        for i in range(len(means_cr1)):
            print(names[i],": ",means_cr1[i],"+/-",errors_cr1[i])
        
        print("CR 2")
        for i in range(len(means_cr2)):
            print(names[i],": ",means_cr2[i],"+/-",errors_cr2[i])
        
        print("CR 3")
        for i in range(len(means_cr3)):
            print(names[i],": ",means_cr3[i],"+/-",errors_cr3[i])

        if self.__co_sim:
            print("Co-Occurrence Similarity")
            for i, (train,test) in enumerate(datasets):
                if self.__max_n > 2000:
                    n_kw = 2000
                else:
                    n_kw = self.__max_n
                kw_lst = self._get_n_most_common(data[i][0],n_kw,False)
                cs = self._co_sim(train,test,kw_lst)
                co_sims.append(cs)
                print(names[i],": ",cs)
        
        self._plot_metric(scm_val,scm_err_stat,scm_err_sys,names,co_sims)
        self._plot_cr1_2(results_cr1,names,"CR 1")
        self._plot_cr1_2(results_cr2,names,"CR 2")
        self._plot_cr3(results_cr3,names)
