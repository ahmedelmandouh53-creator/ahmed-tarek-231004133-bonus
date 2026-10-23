"""Self-check: compare MapReduce output with a plain single-process count and
with the dataset's expected_word_count.txt.

Usage:  python verify.py <input_dir> <output_dir>
"""
import glob
import os
import re
import sys
from collections import Counter

from wordcount import partition

here = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) == 3:
    in_dir, out_dir = sys.argv[1], sys.argv[2]
else:                         # e.g. pressing Run in the IDE
    in_dir = os.path.join(here, "..", "datasets", "text-files")
    out_dir = os.path.join(here, "output")

# reference count (no MapReduce)
ref = Counter()
for p in glob.glob(os.path.join(in_dir, "book_*.txt")):
    with open(p, encoding="utf-8") as f:
        ref.update(re.findall(r"[a-z0-9]+", f.read().lower()))

# MapReduce output
got, per_reducer = {}, {}
for p in sorted(glob.glob(os.path.join(out_dir, "part-r-*"))):
    with open(p, encoding="utf-8") as f:
        rows = [l.rstrip("\n").split("\t") for l in f]
    r = int(p[-5:])
    per_reducer[r] = rows
    assert [w for w, _ in rows] == sorted(w for w, _ in rows), "not sorted"
    for w, c in rows:
        assert w not in got, "word in two reducers: " + w
        assert partition(w) == r, "wrong reducer for " + w
        got[w] = int(c)

print("reducer files:", {r: len(v) for r, v in per_reducer.items()})
print("total words :", sum(got.values()), "| distinct:", len(got))
print("matches independent Counter:", got == dict(ref))

exp_path = os.path.join(in_dir, "expected_word_count.txt")
if os.path.exists(exp_path):
    with open(exp_path, encoding="utf-8") as f:
        exp = {}
        for l in f:
            w, c = l.split()
            exp[w] = exp.get(w, 0) + int(c)
    diff = {w: (exp.get(w), got.get(w)) for w in set(exp) | set(got)
            if exp.get(w) != got.get(w)}
    print("matches expected_word_count.txt:", not diff)
    for w in sorted(diff):
        print("   %-12s expected=%s mapreduce=%s" % (w, *diff[w]))
