import os
import sys
import importlib
import stepGAN_train
import texar
import random
BASEDIR = '/home/gray/code/stepGAN/opspam_final/out/'

def get_config_file(trp, usp):
    return 'stepGAN_base_config_nogan' 

unsup_rev_paths = {
    0.5 : '/home/gray/code/stepGAN/chicago_unlab_reviews50.txt',
    0.6 : '/home/gray/code/stepGAN/chicago_unlab_reviews60.txt',
    0.7 : '/home/gray/code/stepGAN/chicago_unlab_reviews70.txt',
    0.8 : '/home/gray/code/stepGAN/chicago_unlab_reviews80.txt',
    0.9 : '/home/gray/code/stepGAN/chicago_unlab_reviews90.txt',
    1.0 : '/home/gray/code/stepGAN/emptyfile.txt'
}

train_revs = '/home/gray/code/stepGAN/opspam_train_reviews.txt'
train_labs = '/home/gray/code/stepGAN/opspam_train_labels.txt'
test_revs = '/home/gray/code/stepGAN/opspam_test_reviews.txt'
test_labs = '/home/gray/code/stepGAN/opspam_test_labels.txt'

def make_data(trp, usp, run):
    with open(train_revs, 'r') as f:
        revs = f.readlines()
    with open(train_labs, 'r') as f:
        labs = f.readlines()
    
    shfl_idx = random.sample(list(range(len(revs))), len(revs))
    revs = [str(revs[i]) for i in shfl_idx]
    labs = [str(labs[i]) for i in shfl_idx]

    vocab = texar.data.make_vocab([train_revs, test_revs, unsup_rev_paths[usp]], 10000)
    tr = revs[:round(trp *len(revs))]
    vr = revs[round(trp * len(revs)):]
    tl = labs[:round(trp * len(revs))]
    vl = labs[round(trp * len(revs)):]
 
    with open(unsup_rev_paths[usp], 'r') as f:
        unsup_revs = f.readlines()
    tr = tr+ unsup_revs
    tl = tl + ['-1\n'] * len(unsup_revs)


    dir_name = 'tr{}_usp{}_{}'.format(int(trp*100), int(usp * 100), run)
    #os.mkdir(os.path.join(BASEDIR, dir_name))
    curdir = os.path.join(BASEDIR, dir_name)
    
    data_paths = {
        'train_data_reviews' : os.path.join(curdir, 'trevs.txt'),
        'train_data_labels'  : os.path.join(curdir, 'tlabs.txt'),
        'val_data_reviews' : os.path.join(curdir, 'vrevs.txt'),
        'val_data_labels' : os.path.join(curdir, 'vlabs.txt'),
        'vocab' : os.path.join(curdir, 'vocab.txt'),
        'clas_test_ckpt' : os.path.join(curdir, 'ckpt-bestclas-base'),
        'clas_pred_output' : os.path.join(curdir, 'testpreds_nogan.txt'),
        'dir' : curdir
    }
    """
    with open(data_paths['train_data_reviews'], 'w') as f: 
        for x in tr: 
            f.write(x)

    with open(data_paths['train_data_labels'], 'w') as f:
        for x in tl:
            f.write(str(x))

    with open(data_paths['val_data_reviews'], 'w') as f:
        for x in vr:
            f.write(x)

    with open(data_paths['val_data_labels'], 'w') as f:
        for x in vl:
            f.write(str(x))

    with open(data_paths['vocab'], 'w') as f:
        for v in vocab:
            f.write(v + '\n')
"""
    return data_paths

# 0.5, 0.8 x 0.5, 0.8
for train_pcent in [0.5, 0.7, 0.8, 0.9]:
    for unsup_pcent in [1.0]:
        for run in range(3):
            base_config_file = get_config_file(train_pcent, unsup_pcent)
            data_paths = make_data(train_pcent, unsup_pcent, run)
            importlib.invalidate_caches()
            base_config = importlib.import_module(base_config_file)
            base_config = importlib.reload(base_config)
            # inject file paths
            base_config.train_data['datasets'][0]['files'] = data_paths['train_data_reviews']
            base_config.train_data['datasets'][1]['files' ] = data_paths['train_data_labels']
            base_config.val_data['datasets'][0]['files'] = data_paths['val_data_reviews']
            base_config.val_data['datasets'][1]['files'] = data_paths['val_data_labels']
            base_config.test_data['datasets'][0]['files'] = test_revs
            base_config.test_data['datasets'][1]['files'] = test_labs
            base_config.train_data['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config.val_data['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config.test_data['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config.clas_test_ckpt = data_paths['clas_test_ckpt']
            base_config.clas_pred_output = data_paths['clas_pred_output']
            base_config.log_dir = data_paths['dir']
            base_config.checkpoint_dir = data_paths['dir']
            print(base_config.train_data['datasets'][0]['files'])
            print('Train Pcent {} Unsup Pcent {} Run {}'.format(train_pcent, unsup_pcent, run))
            # Run
            stepGAN_train.main(base_config)
            # Run test
            base_config.clas_test = True
            stepGAN_train.main(base_config)




