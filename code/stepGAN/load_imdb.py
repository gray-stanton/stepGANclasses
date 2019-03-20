import texar as tx
import tensorflow as tf
import numpy as np
import os
import random
import re
random.seed(5)
def clean(s, char=False):
    ns = s.lower()
    ns = ns.replace('<br />', ' ')
    ns = re.sub('[0-9]+', 'N', ns)
    ns = re.sub('[^a-zA-Z0-9 \-.,\'\"!?()]', ' ', ns) # Eliminate all but these chars
    ns = re.sub('([.,!?()\"\'])', r' \1 ', ns) # Space out punctuation
    #if char:
    #    ns = re.sub('(\S)', r' \1 ', ns) # Space out all chars
    ns = re.sub('\s{2,}', ' ', ns) # Trim ws
    str.strip(ns)
    return ns

def _rewrite_reviews(filenames, basepath, text_output_path, label_output_path, char):
    with open(text_output_path, 'w') as txtf, open(label_output_path, 'w') as labf:
        for fn in filenames:
            name = os.path.splitext(fn)[0]
            textid, label = name.split('_')
            with open(os.path.join(basepath, fn), 'r') as f:
                review = f.read()
            cl = clean(review, char)
            txtf.write(cl + '\n')
            dis_label = 0 if int(label) < 5 else 1
            labf.write(str(dis_label) + '\n')



def rewrite(basepath, textout, labout, char):
    names = os.listdir(basepath)
    _rewrite_reviews(names, basepath, textout, labout, char)

def split_valid(textpath, labpath, tr_outtxt, tr_outlab,
                val_outtxt, val_outlab, split_count ):
    with open(textpath, 'r') as txtf, open(labpath, 'r') as labf:
        texts = txtf.readlines()
        labs = labf.readlines()
    shfl_idx = random.sample(range(len(texts)), len(texts))
    texts = [texts[i] for i in shfl_idx]
    labs = [labs[i] for i in shfl_idx]

    val_texts = texts[:split_count]
    val_labs = labs[:split_count]
    train_texts = texts[split_count:]
    train_labs = labs[split_count:]
    with open(tr_outtxt, 'w') as txtf, open(tr_outlab, 'w') as labf:
        for r, l in zip(train_texts, train_labs):
            txtf.write(r)
            labf.write(l)
    with open(val_outtxt, 'w') as txtf, open(val_outlab, 'w') as labf:
        for r, l in zip(val_texts, val_labs):
            txtf.write(r)
            labf.write(l)






if __name__=='__main__':
    rewrite('./aclImdb/train/ex', './reviews.txt', './labs.txt', False)
    rewrite('./aclImdb/test/ex', './test_reviews.txt', './test_labs.txt', False)
    rewrite('./aclImdb/train/ex', './reviews_char.txt', './labs_char.txt', True)
    rewrite('./aclImdb/test/ex', './test_reviews_char.txt', './test_labs_char.txt', True)
    rewrite('./aclImdb/train/unsup', './unsup_reviews.txt', './nothing.txt', False)
    rewrite('./aclImdb/train/unsup', './unsup_reviews_char.txt', './nothing.txt', False)
    split_valid('./reviews.txt', './labs.txt', './train_reviews.txt', './train_labs.txt', 
                './val_reviews.txt', './val_labs.txt', 4000)
    split_valid('./reviews_char.txt', './labs_char.txt', 
                './train_reviews_char.txt', './train_labs_char.txt', 
                './val_reviews_char.txt', './val_labs_char.txt', 4000)

    vocab_words = tx.data.make_vocab(['./reviews.txt', './test_reviews.txt'], max_vocab_size=7000)
    with open('./reviews_char.txt', 'r') as f:
        x = f.read()
        vocab_chars = set(x)
    with open('imdb_vocab.txt', 'w') as vf:
        for v in vocab_words:
            vf.write(v + '\n')

    with open('imdb_vocab_chars.txt', 'w') as vf:
        for v in list(vocab_chars):
            if v != '\n':
                vf.write(v + '\n')




    




    

