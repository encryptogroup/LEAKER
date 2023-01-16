"""
For License information see the LICENSE file.

Authors: Patrick Ehrler

"""
import logging
import sys

from leaker.api import Dataset
from leaker.extension import SelectivityExtension
from leaker.sql_interface import SQLBackend
import matplotlib.pyplot as plt

f = logging.Formatter(fmt='{asctime} {levelname:8.8} {process} --- [{threadName:12.12}] {name:32.32}: {message}',
                      style='{')

console = logging.StreamHandler(sys.stdout)
console.setFormatter(f)

file = logging.FileHandler('eval_dmv_red.log', 'w', 'utf-8')
file.setFormatter(f)

logging.basicConfig(handlers=[console, file], level=logging.DEBUG)

log = logging.getLogger(__name__)

backend = SQLBackend()
log.info(f"has dbs {backend.data_sets()}")
dmv_db: Dataset = backend.load("dmv_100k_11cols")

log.info(f"Loaded {dmv_db.name()} data. {len(dmv_db)} documents with {len(dmv_db.keywords())} words. {dmv_db.has_extension(SelectivityExtension)}")

dmv_db.open()

rlens = []
for q in dmv_db.keywords():
    rlens.append(len(list(dmv_db.query(q))))
rlens.sort(reverse=True)
print(rlens)

plt.plot(rlens)
plt.title("DMV 10k")
#plt.yscale('log')
plt.savefig("power_law_dmv_100k.png")
dmv_db.close()

