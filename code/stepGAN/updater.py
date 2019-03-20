import os
import sys
import importlib
import stepGAN_train
import texar
import random
import subprocess as sp
CLOUDDIR = 'tr90_usp100_9'
CONFIG = 'stepGAN_base_config_domore'


test_revs = '../../opspam_test_reviews.txt'
test_labs = '../../opspam_test_labels.txt'

def main():
    if os.path.isdir('./fixupdate/{}'.format(CLOUDDIR)):
        print('Dir found locally')
        os.chdir('./fixupdate/{}'.format(CLOUDDIR))
    else:
        command = 'gsutil -m cp -r gs://icjai-results/{} ./fixupdate/'.format(CLOUDDIR)
        print(command)
        sp.run(command, shell=True)
        os.chdir('./fixupdate/{}'.format(CLOUDDIR))
    base_config = importlib.import_module(CONFIG)
    base_config.restore_model = True
    base_config.load_checkpoint_file = 'ckpt-bestclas'
    
    base_config.train_data['datasets'][0]['files'] = ['trevs.txt', 'unsup_trevs.txt']
    base_config.train_data['datasets'][1]['files' ] = ['tlabs.txt', 'unsup_tlabs.txt']
    base_config.clas_train_data['datasets'][0]['files'] = 'trevs.txt'
    base_config.clas_train_data['datasets'][1]['files'] = 'tlabs.txt'
    base_config.val_data['datasets'][0]['files'] = 'vrevs.txt'
    base_config.val_data['datasets'][1]['files'] = 'vlabs.txt'
    base_config.test_data['datasets'][0]['files'] = test_revs
    base_config.test_data['datasets'][1]['files'] = test_labs
    base_config.train_data['datasets'][0]['vocab_file'] = 'vocab.txt'
    base_config.clas_train_data['datasets'][0]['vocab_file'] = 'vocab.txt'
    base_config.val_data['datasets'][0]['vocab_file'] = 'vocab.txt'
    base_config.test_data['datasets'][0]['vocab_file'] = 'vocab.txt'
    base_config.clas_test_ckpt = 'ckpt-bestclas'
    base_config.clas_pred_output = 'testpreds.txt'
    base_config.log_dir = '.'
    base_config.checkpoint_dir = '.'

    stepGAN_train.main(base_config)
    # Run test
    base_config.clas_test = True
    stepGAN_train.main(base_config)


if __name__ == "__main__":
    main()
