from enum import Enum, auto
from functools import lru_cache
import re
# from config import *

def get_hht_from_line(line):
    hht = int(line[3])
    assert 0 <= hht and hht <= 9
    return hht

def get_ten_from_data(line):
    ten = int(line[0])
    assert 0 <= ten and ten <= 4
    return ten

def get_size_from_data(line):
    size = int(line[2])
    assert 0 <= size and size <= 7
    return size

def get_ten_counts(row):
    return tuple(row['IFF00' + str(i)] for i in range(2, 5))

def get_hht_counts(row):
    return tuple(row['H8C00' + str(i)] for i in [3, 5, 6, 8, 9]) # married, family male hhldr, family female hhldr, nonfamily alone, nonfamily not alone

def get_size_counts(row):
    return tuple(row['IFO00' + str(i)] for i in range(2, 9))

def get_hht_from_h_record(h_record):
    hht = int(h_record[57:58])
    assert 0 <= hht and hht <= 9
    return hht

def get_ten_from_h_record(h_record):
    ten = int(h_record[55:56])
    assert 0 <= ten and ten <= 4
    return ten

###################################################################3
class Race(Enum):
    WHITE = auto()
    BLACK = auto()
    AM_IND_ALASKAN = auto()
    ASIAN = auto()
    HAWAIIAN_PI = auto()
    OTHER = auto()
    TWO_PLUS = auto()

    def __lt__(self, other):
        return self.value < other.value

RACES = [r for r in Race]

RACE_HIS_ENUM = [(r, h) for h in (0, 1) for r in Race]

TYPES = [(r, f, s) for r in Race for f in (False, True) for s in range(1, 8)]
TYPES += [(1, f, s) for f in (False, True) for s in range(1, 8)]

TYPE_INDEX = {k: i for i, k in enumerate(TYPES)}

MICRO_TO_RACE = {
        '01': Race.WHITE,
        '02': Race.BLACK,
        '03': Race.AM_IND_ALASKAN,
        '04': Race.AM_IND_ALASKAN,
        '05': Race.AM_IND_ALASKAN,
        '06': Race.ASIAN,
        '07': Race.HAWAIIAN_PI,
        '08': Race.HAWAIIAN_PI,
        '09': Race.HAWAIIAN_PI,
        '10': Race.OTHER,
        '11': Race.TWO_PLUS,
        }

I_CHAR_MAP = {
        'J': Race.WHITE,
        'K': Race.BLACK,
        'L': Race.AM_IND_ALASKAN,
        'M': Race.ASIAN,
        'N': Race.HAWAIIAN_PI,
        'O': Race.OTHER,
        'P': Race.TWO_PLUS,
        }

def get_race_from_p_record(p_record):
    r = p_record[36:38]
    return MICRO_TO_RACE[r]

def get_eth_from_p_record(p_record):
    r = int(p_record[27:28])
    assert r in (0, 1)
    return r

def get_age_from_p_record(p_record):
    age = int(p_record[21:23])
    assert 0 <= age
    return age

def get_is_family_from_h_record(h_record):
    fam = int(h_record[57:58])
    assert 0 <= fam and fam <= 7
    return fam <= 3

def get_n_under_18_from_h_record(h_record):
    n_under_18 = int(h_record[62:64])
    assert n_under_18 >= 0
    return n_under_18

def get_weight_from_h_record(h_record):
    weight = int(h_record[47:49])
    return weight

# May need this to go back to shape files
# def get_micro_file():
    # return MICRO_FILE

# def get_dist_dir():
    # return OUTPUT_DIR

# def get_block_file():
    # return BLOCK_FILE

# def get_block_out_file():
    # return BLOCK_OUTPUT_FILE

# def get_swapped_file(task_name=''):
    # return get_dist_dir() + task_name + 'swapped.csv'

# def get_person_micro_file(task_name=''):
    # return get_dist_dir() + task_name + 'person_micro.csv'

# def get_dp_tot_file(task_name):
    # return get_dist_dir() + task_name + 'tot_toydown.csv'

# def get_dp_vap_file(task_name):
    # return get_dist_dir() + task_name + 'vap_toydown.csv'

# def get_shape_file(area):
    # shape_dict = {
            # 'BLOCK': SHAPE_FILE,
            # 'BLOCK_GROUP': GROUP_SHAPE_FILE,
            # 'COUNTY': COUNTY_SHAPE_FILE,
            # 'TRACT': TRACT_SHAPE_FILE,
            # 'UP_LEG': UP_LEG_SHAPE_FILE,
            # 'LOW_LEG': LOW_LEG_SHAPE_FILE,
            # 'CONG': CONG_SHAPE_FILE,
            # }
    # return shape_dict[area]

# get_shape_file.AREAS = ['BLOCK', 'BLOCK_GROUP', 'COUNTY', 'TRACT', 'UP_LEG', 'LOW_LEG', 'CONG']

# def get_synthetic_out_file(name=''):
    # return get_dist_dir() + name + 'synthetic.csv'


# Sample codes [H9S-H9Y]
# H9S001:      Total
# H9S002:      Under 18 years
# H9S003:      18 years and over

# Sample codes [IAJ-IAP]
# IAJ001:      Total
# IAJ002:      Family households
# IAJ003:      Family households: 2-person household
# IAJ004:      Family households: 3-person household
# IAJ005:      Family households: 4-person household
# IAJ006:      Family households: 5-person household
# IAJ007:      Family households: 6-person household
# IAJ008:      Family households: 7-or-more-person household
# IAJ009:      Nonfamily households
# IAJ010:      Nonfamily households: 1-person household
# IAJ011:      Nonfamily households: 2-person household
# IAJ012:      Nonfamily households: 3-person household
# IAJ013:      Nonfamily households: 4-person household
# IAJ014:      Nonfamily households: 5-person household
# IAJ015:      Nonfamily households: 6-person household
# IAJ016:      Nonfamily households: 7-or-more-person household
@lru_cache(maxsize=None)
def I_col_to_tup(col):
    assert re.match('IA[J-P]0[0-1][0-9]', col)
    race = I_CHAR_MAP[col[2]]
    code = int(col[-2:])
    if code == 1:
        return (race, None, None)
    elif code == 2:
        return (race, True, None)
    elif code == 9:
        return (race, False, None)
    elif code < 9:
        return (race, True, code-1)
    else:
        return (race, False, code-9)

def is_hh_code(col):
    if not re.match('IA[J-P]0[0-1][0-9]', col):
        return False
    code = int(col[-2:])
    if code in (1, 2, 9):
        return False
    return True

def row_to_hhs(row):
    row = row[[col for col in row.index if is_hh_code(col)]]
    row = row[row > 0]
    return {I_col_to_tup(ind): val for ind, val in row.iteritems()}

def get_race_counts(row):
    return tuple(row['H7X00' + str(i)] for i in range(2, 9))

def get_eth_counts(row):
    return tuple(row['H7X00' + str(i)] for i in range(2, 9))

def get_over_18_counts(row):
    return tuple(row['H9%s003' % s] for s in 'STUVWXY')

def get_over_18_total(row):
    return row['H8A003']

def get_num_hhs(row):
    return row['H8M001']

def get_rh_counts(row):
    return tuple(count_from_rh(row, rh) for rh in RACE_HIS_ENUM)

def get_age_eth(row):
    return row['H9Z003']

def get_types(row):
    return tuple(count_from_type(row, t) for t in TYPES)

RACE_ETH_COL_MAP = {
        Race.WHITE: 'IAJ',
        Race.BLACK: 'IAK',
        Race.AM_IND_ALASKAN: 'IAL',
        Race.ASIAN: 'IAM',
        Race.HAWAIIAN_PI: 'IAN',
        Race.OTHER: 'IAO',
        Race.TWO_PLUS: 'IAP',
        1: 'IAQ',
        }
def count_from_type(row, t):
    a, f, s = t
    if f == True and s == 1:
        return 0
    prefix = RACE_ETH_COL_MAP[a]
    if f:
        offset = 1
    else:
        offset = 9
    offset += s
    return row[prefix + '%03d' % offset]

def count_from_rh(row, rh):
    r, h = rh
    assert h in (0, 1)
    if h == 0:
        offset = 2
    else:
        offset = 10
    offset += r.value
    return row['H7Z0%02d' % offset]

def has_valid_age_data(row):
    if row['H7X001'] != row['H8A001']:
        return False
    elif sum(get_over_18_counts(row)) != get_over_18_total(row):
        return False
    return True

def num_digits(num):
    return len(str(num))

def approx_equal(a, b, tolerance=.001):
    return abs(a-b) < tolerance

def hh_to_race_eth_age_tup(hh):
    return hh.race_counts + (hh.eth_count,) + (hh.n_over_18,)

if __name__ == '__main__':
    print(RACE_HIS_ENUM)
    print(TYPE_INDEX)
