from collections import defaultdict
import random


def load_dataset(data_path, sep, user_based=True):
    all_users_set = set()
    all_movies_set = set()

    with open(data_path, 'rt') as data:

        train = defaultdict(list)
        tests = defaultdict(list)

        for i, line in enumerate(data):
            uid, mid, rat, timstamp = line.strip().split(sep)

            if uid not in all_users_set:
                all_users_set.add(uid)
            if mid not in all_movies_set:
                all_movies_set.add(mid)

            if random.uniform(0, 1) < 0.2:
                if user_based:
                    tests[uid].append((mid, float(rat)))
                else:
                    tests[mid].append((uid, float(rat)))
            else:
                if user_based:
                    train[uid].append((mid, float(rat)))
                else:
                    train[mid].append((uid, float(rat)))

    return list(all_users_set), list(all_movies_set), train, tests


