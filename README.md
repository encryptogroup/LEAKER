## LEAKER
### A framework for LEakage AttacK Evaluation on Real-world data

###### This framework allows for an easy evaluation of leakage attacks against encrypted search.

---

#### Included Attacks
Currently, LEAKER includes the following attacks:
- Range Attacks:
    - [[LMP18]](https://eprint.iacr.org/2017/701)
    - [[GLMP18]](https://eprint.iacr.org/2018/965)
    - [[GLMP19]](https://eprint.iacr.org/2019/011)
    - [[GJW19]](https://eprint.iacr.org/2019/1198)
    - [[KPT20]](https://eprint.iacr.org/2019/441)
    - [[KPT21]](https://eprint.iacr.org/2021/093.pdf)
- Keyword Attacks:
    - [[IKK12]](https://www.ndss-symposium.org/ndss2012/ndss-2012-programme/access-pattern-disclosure-searchable-encryption-ramification-attack-and-mitigation/)
    - [[CGPR15]](https://eprint.iacr.org/2016/718)
    - [[BKM20]](https://eprint.iacr.org/2019/1175)
    - [[RPH21]](https://doi.org/10.1007/978-3-030-78375-4_7)
    - [[NHP+21]](https://doi.org/10.1145/3460120.3484540)
    - [[DHP21]](https://www.usenix.org/system/files/sec21-damie.pdf)
---

#### Requirements
The framework has been written in Python 3.8. To install all requirements, you can use the `requirements.txt` file:

    pip install -r requirements.txt
    
Additional steps are necessary for some attacks or optimizations:
* For GLMP to use `graph-tool`, you optionally can install [python3-graph-tool](https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions) 
and set the `PYTHON_DIST_PACKAGES_DIRECTORY` variable in `api/constants.py` appropriately.
* For ApproxOrder, [pybind11](https://github.com/pybind/pybind11) and [PQ-trees](https://github.com/Gregable/pq-trees)
 need to be downloaded, slighlty modified, and built. A script that does this is automatically called if the
 requirements are met. To enable this, you need to give that script executable permissions: 

    `chmod +x leaker/pq-trees/compile_pq-trees.sh`
* For speed ups to ARR and APA using [numba](http://numba.pydata.org/), you need to ensure its
[dependencies](https://numba.pydata.org/numba-doc/latest/user/installing.html#dependency-list) are met on your system.

To install LEAKER on your system, run:

    pip install -e .

Generating the documentation will require pdoc.

---

#### Structure
* `data` will be created by LEAKER to store indexed data and caches (in `data/pickle` and `data/whoosh`) as well as the
output of evaluations (`data/figures`).
* `data_sources` is a folder to input in the raw data to be indexed by LEAKER. Our examples and evaluation scripts use it, but
you can use any input directory with LEAKER.
* `evaluations` contains the scripts to replicate the experiments in our paper. The `GOOGLE_README.txt` contains the
instructions given to the participants that evaluated attacks on their private Google data. `leap_experiment.py` contains the reproduction of the LEAP experiment.
* `examples.py` contains simple examples to show the usage of LEAKER.
* `leaker` contains the core LEAKER module.
* `tests` contains tests.

---

#### Usage
Refer to `examples.py` to see how to use LEAKER.
First, you need to download/extract the raw data into a corresponding subdirectory of `data_sources`. Then, you can index
this data source (necessary only once) and load it with LEAKER to perform evaluations.

To generate the documentation: enter `pdoc --html leaker` with LEAKER/ as the current working directory.

---

#### Acknowledgements

This framework has been developed by Nelson Brüchmann, Patrick Ehrler, Abdelkarim Kati, Johannes Leupold, Tobias Stöckert, Amos Treiber, and Michael Yonli.

The framework also uses [code by Ruben Groot Roessink](https://github.com/rubengrootroessink/IKK-query-recovery-attack) for its IKK attack optimization, which is located in the folder `ikk_roessink` and released under the
license `ikk_roessink/LICENSE.MD`.
