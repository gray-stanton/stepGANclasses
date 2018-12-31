import csv
import numpy as np
import re
import json
import os
import tensorflow as tf
import pickle
import logging
from itertools import product
from collections import Counter, namedtuple, deque
import ipdb
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

def op_spam(data_dir):
    # Generates reviews, no labels
    path = os.path.abspath(data_dir) + '/' + 'deceptive-opinion.csv'
    with open(path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        _ = next(spamreader) # Skip header
        for row in spamreader:
            review_text = row[-1]
            yield review_text
 

# Necessary workaround as pickle doesn't work on closures
strip_reg = re.compile('[^a-zA-Z0-9. ]', re.UNICODE)
single_space_reg = re.compile(' {2,}')
def simplify(s):
    ns = str.lower(s)
    ns = ns.replace('.', ' . ') # Separate out period as it's own word.
    ns = strip_reg.sub('', ns) # only alphanumeric plus period and space
    ns = single_space_reg.sub(' ', ns) # only single spaces
    ns = ns.strip()
    return ns


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
        ns = ns.strip()
        return ns
    return simplify

def all_reviews(data_dir):
    for r in op_spam(data_dir):
        yield r
    for r in tripadvisor(data_dir):
        yield r



def tokenize(text_gen_fun, word_level=True,
             vocab_size=2000, max_len = -1):
    # First pass, count tokens, get max_len
    word_counts = Counter()
    simplifier = get_simplifier()
    get_max_len = True if max_len == -1 else False
    gen = text_gen_fun()
    for text in gen:
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
    for text in gen:
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




def tripadvisor(data_dir):
    path = os.path.abspath(data_dir)  + '/' + 'tripadvisor.txt'
    with open(path, 'r') as f:
        for line in f:
            yield line

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

def make_dataset(name, seq_length, vocab_size,
                 char_level, data_dir, cache_path):
    #TODO implement switching
    path_hack =\
        '/home/gray/code/seqgan-opinion-spam/data/deceptive-opinion.csv'
    load_func = lambda: load_labelled_op_spam(path_hack)
    array, ti, it = tokenize(load_func, False, 50, 40)
    ds = make_LM_dataset(array, seq_length)
    DataHolder = namedtuple(
        'DataHolder', ['train_dataset', 'valid_dataset', 'test_dataset'
                       'vocab_size','seq_length', 
                       'token_to_idx', 'idx_to_token'])
    data_holder = DataHolder(ds, vocab_size, seq_length)
    return data_holder


def rewrite_as_tokens(text_iterable, tokenizer, outfile):
    tok_count = 0
    for l in text_iterable:
        tokens = tokenizer.tokenize(l)
        tokens.append(tokenizer.token_to_idx['<EOD>'])
        for t in tokens:
            outfile.write(str(t) + '\n')
            tok_count += 1
    return tok_count


def make_data_cache(cache_path):
    # Char level LM
    path = os.path.abspath(cache_path) + '/'
    simplifiers = [simplify]
    data_generating_functions = [op_spam, trip_advisor]
    # (Word_level, sequence length)
    configs = [('char', 64, 1000), ('word', 12, 5000)]
    overlap_step_size = 3
    # some reviews are REALLLY long
    max_len = 1000
    do_all = product(data_generating_functions,
                     simplifiers,
                     configs)
    for gen_maker, sim, (level, l, v_size) in do_all:
        # 1000 max vocab means keep all characters
        word_level = level == 'word'
        tok = Tokenizer(gen_maker(), sim, word_level, v_size)
        full_stream = []
        gen = gen_maker()
        for text in gen:
            t = tok.tokenize(text)
            # each text is one document, add EOD to end.
            t = t[:max_len]
            t.append(tok.token_to_idx['<EOD>'])
            t = np.array(t, dtype=np.int16)
            full_stream.append(t)
        print('Full stream loaded')
        text_array = np.concatenate(full_stream)
        # Discard remainder
        text_array = text_array[:text_array.shape[0] - (text_array.shape[0] % l)]
        overlap = []
        for i in range(0, text_array.shape[0], overlap_step_size):
            overlap.append(text_array[i: i + l])
        overlap_array = np.concatenate(overlap)
        one_step_ahead = overlap_array[1:]
        one_step_behind = overlap_array[:-1]
        X = one_step_behind[:one_step_behind.shape[0] - one_step_behind.shape[0] % l]
        X = X.reshape(-1, l)
        Y = one_step_ahead[:one_step_ahead.shape[0] - one_step_ahead.shape[0] % l]
        # Save
        name_params = [gen_maker.__name__, level, l, tok.vocab_size]
        Xname = "X_LM_{}_{}_{}_{}".format(*name_params)
        Yname = "Y_LM_{}_{}_{}_{}".format(*name_params)
        tokname = "Tok_{}_{}_{}_{}".format(*name_params)
        print('Cached {} {} {} {}'.format(*name_params))
        with open(path+Xname + '.npy', 'wb') as f:
            np.save(f, X)
        with open(path+Yname + '.npy', 'wb') as f:
            np.save(f, Y)
        with open(path + tokname + '.pkl', 'wb') as f:
            pickle.dump(tok, f)




def rewrite_op_spam_labels(data_dir, outfile):
    labels = []
    csvpath = os.path.abspath(data_dir) + '/' + 'deceptive-opinion.csv'
    with open(csvpath, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        _ = next(spamreader) # Skip header
        for row in spamreader:
            label = row[0]
            class_label = 1 if label == 'deceptive' else 0
            labels.append(class_label)
    
    for l in labels:
        outfile.write(str(l) + '\n')

def rewrite_all_labels(data_dir, outfile):
    labels = []
    csvpath = os.path.abspath(data_dir) + '/' + 'deceptive-opinion.csv'
    txtpath = os.path.abspath(data_dir) + '/' + 'tripadvisor.txt'
    with open(csvpath, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        _ = next(spamreader) # Skip header
        for row in spamreader:
            label = row[0]
            class_label = 1 if label == 'deceptive' else 0
            labels.append(class_label)
        
    with open(txtpath, 'r') as txtfile:
        for r in txtfile:
            labels.append(-1)
    for l in labels:
        outfile.write(str(l) + '\n')

    
    




class Tokenizer():
    def __init__(self, text_iterable, simplifier, word_level, max_vocab_size):
        # Count tokens
        self.word_level = word_level
        self.simplifier = simplifier
        word_counts = Counter()
        for text in text_iterable:
            simple_text = self.simplifier(text)
            if word_level:
                split_text = simple_text.split(' ')
            else:
                split_text = list(simple_text)
            word_counts.update(split_text)
        # Create mappings
        logging.info('Tokenizer pass done')
        in_vocab = word_counts.most_common(max_vocab_size)

        self.token_to_idx = {w:(i+4) for i, w in\
                        enumerate(sorted([w[0] for w in in_vocab]))}
        self.token_to_idx['<P>'] = 0 # 0 is padding
        self.token_to_idx['<UNK>'] = 1 # 1 is Unknown
        self.token_to_idx['<EOS>'] = 2 # 2 is End of Sentence
        self.token_to_idx['<EOD>'] = 3 # 3 is End of Document
        self.vocab_size = len(self.token_to_idx)
        self.idx_to_token = {i:w for w, i in self.token_to_idx.items()}

    def tokenize(self, s):
        simple_s = self.simplifier(s)
        if self.word_level:
            split_s = simple_s.split(' ')
        else:
            split_s = list(simple_s)
        return [self.token_to_idx[w]\
                if w in self.token_to_idx else 1\
                for w in split_s]

    def untokenize(self, t):
        if self.word_level:
            s = ' '.join([self.idx_to_token[i] for i in t])
        else:
            s = ''.join([self.idx_to_token[i] for i in t])
        return s


def labeled_stream_generator(tokenfile, labelfile,
                             seq_length, break_token, pad_token = 0):
    token_seq = deque()
    seeking = False
    with open(tokenfile, 'r') as tokf, open(labelfile, 'r') as labf:
        label = int(next(labf))
        for l in tokf:
            new_token = int(l)
            # Reset when in new doc  
            if not seeking:
                token_seq.append(new_token)
                if new_token == break_token:
                    # Pad to end
                    token_seq.extend([pad_token] * (seq_length - len(token_seq)))
                if len(token_seq) == seq_length:
                    yield (np.array(list(token_seq), dtype=np.int16),
                       np.array([label], dtype=np.int16))
                    token_seq.clear()
                    seeking = True if new_token != break_token else False
                    label = int(next(labf))
            elif new_token == break_token:
                seeking = False




def cache_datasets(rawdata_dir, outputdata_dir):
    rawdata_dir_prefix = os.path.abspath(rawdata_dir) + '/'
    outputdata_dir = os.path.abspath(outputdata_dir) + '/'
    def _cache(dataname, datatype, word_level, vocab_size, firstpass_gen,
               secondpass_gen):
        tokenizer = Tokenizer(firstpass_gen(rawdata_dir), simplify, word_level,
                              vocab_size)
        fname = outputdata_dir + dataname + '_' + datatype
        token_file = (fname + '_data.txt')
        tokenizer_file = fname + '_tokenizer.pkl'
        tok_count_file = fname + '_tokcount.txt'
        with open(token_file, 'w') as f:
            tok_count = rewrite_as_tokens(secondpass_gen(rawdata_dir), tokenizer, f)
        with open(tokenizer_file, 'wb') as f:
            pickle.dump(tokenizer, f)
        with open(tok_count_file, 'w') as f:
            f.write(str(tok_count))


    # Char LM All
    print('Char LM')
    _cache('AllChar', 'LM', False, 50, all_reviews, all_reviews)
    print('WordSmall_LM')
    _cache('AllWordSmall', 'LM', True, 3000, all_reviews, all_reviews)
    print('WordLMBig')
    _cache('AllWordBig', 'LM', True, 12000, all_reviews, all_reviews)
    print('OSChar')
    _cache('OSChar', 'Lab', False, 50, all_reviews, op_spam)
    print('OSWordSmall')
    _cache('OSWordSmall', 'Lab', True, 3000, all_reviews, op_spam)
    print('OSWordBig')
    _cache('OSWordBig', 'Lab', True, 12000, all_reviews, op_spam)

    #Op Spam Labels - Labelled only
    labels_file = outputdata_dir + '/' + 'OS_labels.txt'
    with open(labels_file, 'w') as f:
        rewrite_op_spam_labels(rawdata_dir, f)
    
    # All labels (-1 for unlabelled)
    labels_file = outputdata_dir + '/' + 'All_labels.txt'
    with open(labels_file, 'w') as f:
        rewrite_all_labels(rawdata_dir, f)
    
    return




LAB_DATASETS = {'CharOSLab', 'WordOSLabSmall', 'WordOSLabBig'}
LM_DATASETS = {'CharLM', 'WordLMSmall', 'WordLMBig'}
def make_dataset(dataset_name, dataset_type, data_dir, seq_length):
    data_path_prefix = os.path.abspath(data_dir) + '/'
    prefix = data_path_prefix + dataset_name + '_' + dataset_type
    tokenizer_file = prefix + '_tokenizer.pkl'
    data_file = prefix + '_data.txt'
    tok_count_file = prefix + '_tokcount.txt'
    with open(tokenizer_file, 'rb') as f:
        tokenizer = pickle.load(f)
    if dataset_type == 'LM':
        gen = lambda: token_stream_generator(data_file, seq_length, -1, True)
    elif dataset_type == 'Lab':
        labelfile = data_path_prefix + 'OS_labels.txt'
        break_token = tokenizer.token_to_idx['<EOD>']
        gen = lambda: labeled_stream_generator(data_file, labelfile, seq_length, 
                                       break_token, 0)
        with open(labelfile, 'r') as f:
            lab_count = 0
            for _ in f:
                lab_count += 1
    with open(tok_count_file, 'r') as f:
        tok_count = int(f.read())
    if dataset_type == 'LM':
        size = tok_count / seq_length
    else:
        size = lab_count

    ds = tf.data.Dataset.from_generator(gen, (tf.int16, tf.int16))
    return ds, tokenizer, size







def token_stream_generator(tokenfile, seq_length, offset, return_ys=True):
    seq = deque()
    # -1 indicates non-overlapping sequences
    if offset == -1:
        offset = seq_length
    with open(tokenfile, 'r') as f:
        for l in f:
            seq.append(int(l))
            if len(seq) == seq_length + 1:
                X_tokens = list(seq)[:-1]
                Y_tokens = list(seq)[1:]
                if return_ys:
                    output = (np.array(X_tokens, dtype=np.int16),
                              np.array(Y_tokens, dtype=np.int16))
                else:
                    output = np.array(X_tokens, dtype=np.int16)
                for _ in range(offset):
                    seq.popleft()
                yield output
            







