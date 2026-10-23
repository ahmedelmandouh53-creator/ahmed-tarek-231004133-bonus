"""Word count with hand-written map / combine / partition / shuffle / sort / reduce.

Pipeline
    input folder (1 file = 1 split)
      -> MAP      one OS process per split, emits (word, 1)
      -> COMBINE  each mapper sums its own pairs locally
      -> PARTITION each mapper routes every word to reducer sum(char codes) % 2
      -> SHUFFLE  collect partition r from every mapper, group values by word
      -> SORT     order the keys
      -> REDUCE   sum the values per word, write one file per reducer

Usage:  python wordcount.py <input_dir> <output_dir>
"""
import multiprocessing as mp
import os
import sys

NUM_REDUCERS = 2


# ---------------------------------------------------------------- helpers
def tokenize(line):
    # cleans up a line and hands back just the words
    """Lowercase, turn punctuation into spaces, then split on whitespace."""
    cleaned = "".join(ch if ch.isalnum() else " " for ch in line.lower())
    return cleaned.split()


def partition(word, num_reducers=NUM_REDUCERS):
    # picks which reducer a word goes to, same word = same reducer every time
    """Add up the character codes of the word, then take the remainder.
    Python's built-in hash() is salted per process, so two mappers would
    route the same word to different reducers; this always gives the same answer."""
    return sum(ord(ch) for ch in word) % num_reducers


# -------------------------------------------------------------------- map
def map_task(path):
    # one mapper: reads a file, counts its words and sorts them into reducer buckets
    """Runs in its own process.
    Returns (split name, pid, pairs emitted, [one {word: subtotal} dict per reducer])."""
    # MAP: emit (word, 1) for every word in the split
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            for word in tokenize(line):
                pairs.append((word, 1))

    # COMBINE + PARTITION: local subtotals, bucketed by target reducer
    buckets = [{} for _ in range(NUM_REDUCERS)]
    for word, one in pairs:
        b = buckets[partition(word)]
        b[word] = b.get(word, 0) + one
    return os.path.basename(path), os.getpid(), len(pairs), buckets


def map_worker(path, queue):
    # runs map_task in its own process and drops the answer in the queue
    queue.put(map_task(path))


# ------------------------------------------------------- shuffle & reduce
def shuffle_and_sort(mapper_outputs, r):
    # collects one reducer's words from every mapper, groups them and sorts them
    """Gather reducer r's partition from every mapper, group by word, sort."""
    grouped = {}
    for buckets in mapper_outputs:
        for word, subtotal in buckets[r].items():
            grouped.setdefault(word, []).append(subtotal)
    return sorted(grouped.items())          # [(word, [subtotals...]), ...]


def reduce_task(args):
    # one reducer: adds up the counts per word and writes its output file
    r, grouped, out_dir = args
    out_path = os.path.join(out_dir, "part-r-%05d" % r)
    with open(out_path, "w", encoding="utf-8") as f:
        for word, values in grouped:
            f.write("%s\t%d\n" % (word, sum(values)))
    return r, os.getpid(), len(grouped), out_path


def reduce_worker(args, queue):
    # same idea as map_worker, but for a reducer
    queue.put(reduce_task(args))


def run_parallel(worker, arg_list):
    # starts one process per job at the same time and collects what they send back
    """Start one process per argument at the same time, then collect results."""
    queue = mp.Queue()
    procs = [mp.Process(target=worker, args=(a, queue)) for a in arg_list]
    for p in procs:
        p.start()
    results = [queue.get() for _ in procs]   # drain before join to avoid deadlock
    for p in procs:
        p.join()
    return results


# ------------------------------------------------------------------- main
def main(in_dir, out_dir):
    # runs the whole thing: map, then shuffle, then reduce, and prints a report
    splits = sorted(os.path.join(in_dir, n) for n in os.listdir(in_dir)
                    if n.endswith(".txt") and n != "expected_word_count.txt")
    os.makedirs(out_dir, exist_ok=True)

    print("Parent pid %d | %d splits | %d reducers" %
          (os.getpid(), len(splits), NUM_REDUCERS))

    # one process per split, all started before any is joined
    results = sorted(run_parallel(map_worker, splits))
    print("\n== MAP (parallel, one process per split) ==")
    for name, pid, n_pairs, buckets in results:
        print("  %-12s pid=%-6d emitted=%-3d -> reducer0:%d words, reducer1:%d words" %
              (name, pid, n_pairs, len(buckets[0]), len(buckets[1])))

    mapper_outputs = [r[3] for r in results]
    jobs = [(r, shuffle_and_sort(mapper_outputs, r), out_dir)
            for r in range(NUM_REDUCERS)]

    print("\n== REDUCE (parallel, one process per reducer) ==")
    for r, pid, n_words, path in sorted(run_parallel(reduce_worker, jobs)):
        print("  reducer %d pid=%-6d %d distinct words -> %s" % (r, pid, n_words, path))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 1:      # e.g. pressing Run in the IDE
        main(os.path.join(here, "..", "datasets", "text-files"),
             os.path.join(here, "output"))
    else:
        sys.exit(__doc__)
