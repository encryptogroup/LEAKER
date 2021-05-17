import subprocess

from pathlib import Path
from os import listdir
from os.path import exists

from leaker.api.constants import COMPILE_TIMEOUT

path = Path(__file__).resolve().parents[1]  # here path.parents[1]` is the same as `path.parent.parent

new_compilation = True
if exists(str(path) + f'/pq-trees/build/'):
    new_compilation = not any(fname.endswith('.so') for fname in listdir(str(path) + '/pq-trees/build/'))

if new_compilation:
    print(f"No compiled pq-trees C++ files found. Cloning and compiling them now")

    proc = subprocess.Popen(str(path) + '/pq-trees/compile_pq-trees.sh')

    ret = proc.wait(timeout=COMPILE_TIMEOUT)
    if ret == 0:
        print(f"Created pq-trees files.")
    else:
        print(f"Compilation of pq-trees failed! Please try manual compilation.")
