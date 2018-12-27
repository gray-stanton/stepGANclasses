import csv
import numpy as np
import re
import json
import os
from collections import Counter
def load_labelled_op_spam(path='./data/raw/deceptive-opinion.csv'):
    # Generates reviews and labels
    with open(path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        _ = next(spamreader) # Skip header
        for row in spamreader:
            label = row[0]
            review_text = row[-1]
            class_label = 1 if label == 'deceptive' else 0
            yield review_text, label

def get_simplifier():
    # Returns simplifying function
    #a = text_array.copy()
    strip_reg = re.compile('[^a-zA-Z0-9. ]', re.UNICODE)
    single_space_reg = re.compile(' {2,}')
    def simplify(s):
        ns = str.lower(s)
        ns = ns.replace('.', ' . ') # Separate out period as it's own word.
        ns = strip_reg.sub('', ns) # only alphanumeric plus period and space
        ns = single_space_reg.sub(' ', ns) # only single spaces
        return ns
    return simplify

def load_all_reviews(op_spam_path='./data/raw/deceptive-opinion.csv',
                     tripadv_path='.data/raw/tripadvisor_json/'):
    for r, l in op_spam_path:
        yield r, l
    for r, l in tripadv_path:
        yield r, y



def tokenize(text_gen_fun, word_level=True,
             vocab_size=2000, max_len = -1):
    # First pass, count tokens, get max_len
    word_counts = Counter()
    simplifier = get_simplifier()
    get_max_len = True if max_len == -1 else False
    gen = text_gen_fun()
    for text, _ in gen:
        simple_text = simplifier(text)
        if word_level:
            split_text = simple_text.split(' ')
        else:
            split_text = list(simple_text)
        word_counts.update(split_text)
        if get_max_len and len(split_text) > max_len:
            max_len = len(split_text)
    # Create mappings
    print('First pass done')
    in_vocab = word_counts.most_common(vocab_size)
    token_to_idx = {w:(i+2) for i, w in\
                    enumerate(sorted([w[0] for w in in_vocab]))}
    token_to_idx['<P>'] = 0 # 0 is padding
    token_to_idx['<UNK>'] = 1 # 1 is Unknown
    idx_to_token = {i:w for w, i in token_to_idx.items()} 
    # Second pass, create arrays
    arrays = []
    gen = text_gen_fun()
    for text, _ in gen:
        simple_text = simplifier(text)
        if word_level:
            indexes = [token_to_idx[w] if w in token_to_idx else 0\
                       for w in simple_text.split(' ')]
        else:
            indexes = [token_to_idx[c] for c in simple_text]
        array = np.array(indexes, dtype=np.uint16)
        # right padding
        pad_array = np.pad(array,
                           (0, max(0, max_len - len(array))),'constant')[0:max_len]
        arrays.append(pad_array)
    array = np.array(arrays)
    return array, token_to_idx, idx_to_token


def rewrite_trip_advisor(path='./data/raw/tripadvisor_json/',
                         outfile='./data/tripadvisor.txt'):
    gen = load_trip_advisor_json()
    with open(outfile, 'w') as f:
        for r, _ in gen:
            f.write(r + '\n')
    return

def load_trip_advisor_text(path='./data/tripadvisor.txt'):
    with open(path, 'r') as f:
        yield f.readline()

def load_trip_advisor_json(path='./data/raw/tripadvisor_json/'):
    # Yields review + label (None) pair
    abspath = os.path.abspath(path)
    json_files = os.listdir(abspath)
    for f in json_files:
        absfile = os.path.join(abspath, f)
        if not os.path.isfile(absfile):
            continue
        if os.path.splitext(absfile)[1] != '.json':
            continue
        with open(absfile, 'r') as j:
            hotel_data = json.load(j)
            review_dict = hotel_data['Reviews']
            contents = [r['Content'] for r in review_dict]
        for review in contents:
            yield review, None

def make_LM_dataset(array, max_seq_length):
    label_array = array[:,1:max_seq_length]
    input_array = array[:, :max_seq_length-1]
    return tf.data.Dataset.from_tensor_slices((input_array, label_array))
    


        


        

        




            





