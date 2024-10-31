from collections import namedtuple, Counter
import numpy as np
import pandas as pd
from ..preprocessing.build_micro_dist import rh_to_str, r_to_str, t_to_str
from .knapsack_utils import make_one_hot_np
from .census_utils import get_rh_counts, get_over_18_counts, get_age_eth, get_types, get_num_hhs, get_ten_counts, get_hht_counts, get_size_counts, RACE_HIS_ENUM, Race, TYPES, TYPE_INDEX
from dp_data import Dataset


MAX_LEVEL = 3

def encode_row(row):
    rh_counts = get_rh_counts(row)
    age_race = get_over_18_counts(row)
    age_eth = (get_age_eth(row),)
    type_encoding = get_types(row)
    num_hh = (get_num_hhs(row),)
    # print([(t, te) for te, t in zip(type_encoding, TYPES) if te > 0])
    return Encoding1(*(rh_counts + age_race + age_eth + type_encoding + num_hh))

class Encoding1(namedtuple('Encoding1',
        [rh_to_str(rh) for rh in RACE_HIS_ENUM] +\
        ['n_18_' + r_to_str(r) for r in Race] +\
        ['n_18_HISP'] +\
        [t_to_str(t) for t in TYPES] +\
        ['num_hh'])):

    def get_type(self):
        # Type is defined as (Race, is_family, size, ethnicity)
        type_range = [getattr(self, t_to_str(t)) for t in TYPES]
        the_type = None
        for i, t in zip(type_range, TYPES):
            if i == 1:
                the_type = t
                break
        assert the_type is not None
        # Ethnicity as last field
        if sum(type_range) == 2:
            the_type += (1,)
        else:
            the_type += (0,)
        return the_type

    def reduce(self, level, use_age):
        if level == 2:
            # schema: (r X h for all r, h; n18+ (?); num HH)
            rh_counts = tuple(getattr(self, rh_to_str(rh)) for rh in RACE_HIS_ENUM)
            if use_age:
                age = (sum(getattr(self, 'n_18_' + r_to_str(r)) for r in Race),)
            else:
                age = ()
            # num HH
            hh = (self.num_hh,)
            tup = rh_counts + age + hh
            if use_age:
                return Encoding2a(*tup)
            else:
                return Encoding2b(*tup)
        elif level == 3:
            # schema: (r for all r; h; n18+ (?))
            r_counts = self.get_r_counts()
            eth = (self.get_eth_count(),)
            if use_age:
                age = (self.get_n18(),)
            else:
                age = ()
            tup = r_counts + eth + age
            if use_age:
                return Encoding3a(*tup)
            else:
                return Encoding3b(*tup)
        else:
            raise ValueError('Level %d undefined' % level)

    def to_sol(self):
        r_counts = self.get_r_counts()
        eth = (self.get_eth_count(),)
        age = (self.get_n18(),)
        return r_counts + eth + age

    def get_r_counts(self):
        eth_0 = np.array([getattr(self, rh_to_str((r, 0))) for r in Race], dtype=int)
        eth_1 = np.array([getattr(self, rh_to_str((r, 1))) for r in Race], dtype=int)
        r_counts = tuple(eth_0 + eth_1)
        return r_counts

    def get_eth_count(self):
        eth_1 = np.array([getattr(self, rh_to_str((r, 1))) for r in Race], dtype=int)
        return sum(eth_1)

    def get_n18(self):
        return sum(getattr(self, 'n_18_' + r_to_str(r)) for r in Race)

    def __str__(self):
        d = self._asdict()
        return str({k: v for k, v in d.items() if v > 0})

def encode_hh_dist(dist):
    new_dist = Counter()
    for hh, prob in dist.items():
        # Race X eth pairs
        rh_counts = hh.rh_counts
        age_race = tuple(hh.n_over_18 * make_one_hot_np(hh.holder.race.value-1, len(Race)))
        age_eth = (hh.holder.eth * hh.n_over_18,)
        type_encoding = make_one_hot_np(TYPE_INDEX[hh.race_type], len(TYPE_INDEX))
        if hh.holder.eth == 1:
            type_encoding += make_one_hot_np(TYPE_INDEX[hh.eth_type], len(TYPE_INDEX))
        type_encoding = tuple(type_encoding)
        num_hh = (1,)
        new_dist[Encoding1(*(rh_counts + age_race + age_eth + type_encoding + num_hh))] += prob
    return new_dist

class Encoding2a(namedtuple('Encoding2a',
        [rh_to_str(rh) for rh in RACE_HIS_ENUM] +\
        ['n_18'] +\
        ['num_hh'])):

    def get_r_counts(self):
        eth_0 = np.array([getattr(self, rh_to_str((r, 0))) for r in Race], dtype=int)
        eth_1 = np.array([getattr(self, rh_to_str((r, 1))) for r in Race], dtype=int)
        r_counts = tuple(eth_0 + eth_1)
        return r_counts

    def get_eth_count(self):
        eth_1 = np.array([getattr(self, rh_to_str((r, 1))) for r in Race], dtype=int)
        return sum(eth_1)

    def to_sol(self):
        r_counts = self.get_r_counts()
        eth = (self.get_eth_count(),)
        age = (self.n_18,)
        return r_counts + eth + age

    def __str__(self):
        d = self._asdict()
        return str({k: v for k, v in d.items() if v > 0})

class Encoding2b(namedtuple('Encoding2b',
        [rh_to_str(rh) for rh in RACE_HIS_ENUM] +\
        ['num_hh'])):

    def get_r_counts(self):
        eth_0 = np.array([getattr(self, rh_to_str((r, 0))) for r in Race], dtype=int)
        eth_1 = np.array([getattr(self, rh_to_str((r, 1))) for r in Race], dtype=int)
        r_counts = tuple(eth_0 + eth_1)
        return r_counts

    def get_eth_count(self):
        eth_1 = np.array([getattr(self, rh_to_str((r, 1))) for r in Race], dtype=int)
        return sum(eth_1)

    def to_sol(self):
        r_counts = self.get_r_counts()
        eth = (self.get_eth_count(),)
        return r_counts + eth

    def __str__(self):
        d = self._asdict()
        return str({k: v for k, v in d.items() if v > 0})

class Encoding3a(namedtuple('Encoding3a',
        [r_to_str(r) for r in Race] +\
        ['num_hisp'] +\
        ['n_18'])):

    def to_sol(self):
        return self

    def __str__(self):
        d = self._asdict()
        return str({k: v for k, v in d.items() if v > 0})

class Encoding3b(namedtuple('Encoding3b',
        [r_to_str(r) for r in Race] +\
        ['num_hisp'])):

    def to_sol(self):
        return self

    def __str__(self):
        d = self._asdict()
        return str({k: v for k, v in d.items() if v > 0})

ten_range = range(1,4)
size_range = range(1,8)
hht_range = range(1,6)
total_range = range(114)
class Encoding4(namedtuple('Encoding4',
        ["answer"])):
    
    def to_sol(self):
        return self

    def __str__(self):
        d = self._asdict()
        return str({k: v for k, v in d.items() if v > 0})

    def reduce(self, level, use_age):
        return self

def encode4_row(answers):
    # ten_counts = get_ten_counts(row)
    # size_counts = get_size_counts(row)
    # hht_counts = get_hht_counts(row)
    # num_hh = (get_num_hhs(row),)
    # return Encoding4(*(ten_counts  + hht_counts + num_hh)) #+ size_counts
    return answers

def encode4_hh_dist(data, domain, qm):
    df = data.df
    new_dist = Counter()
    for index, row in df.iterrows():
        row_df = row.to_frame().T
        row_data = Dataset(row_df, domain)
        answer = qm.get_answers(row_data)
        new_dist[answer] += 1
    return new_dist

def encode_raprank(path, domain, qm):
    df = pd.read_csv(path)
    d = {}
    for index, row in df.iterrows():
        n = row.iloc[-2]
        row_df = row.iloc[:-2].to_frame().T
        for key in domain:
            if key not in row_df:
                row_df[key] = 0
        row_df = row_df[[x for x in domain]]
        row_data = Dataset(row_df, domain)
        answer = qm.get_answers(row_data)
        d[answer] = n
    return d
