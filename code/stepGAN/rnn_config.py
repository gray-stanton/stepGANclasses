#:RNN CONFIG
num_epochs = 100
log_dir = '/tmp/'
train_data_hparams = {
    "batch_size" : 32,
    'shuffle' : True,
    'datasets' : [ 
        {
            "files" : "./train_reviews.txt",
            'vocab_file' : './imdb_vocab.txt',
            'max_seq_length' : 64,
            'length_filter_mode' : 'truncate',
            'bos_token' : '',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'data_name' : 'x',
            'pad_to_max_seq_length' : True
        },
        {
            'files' : './train_labs.txt',
            'data_type' : 'int',
            'data_name' : 'label'
        }
    ]
}

val_data_hparams = {
    "batch_size" : 32,
    'shuffle' : True,
    'datasets' : [ 
        {
            "files" : "./val_reviews.txt",
            'vocab_file' : './imdb_vocab.txt',
            'max_seq_length' : 64,
            'length_filter_mode' : 'truncate',
            'bos_token' : '',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'data_name' : 'x',
            'pad_to_max_seq_length' : True
        },
        {
            'files' : './val_labs.txt',
            'data_type' : 'int',
            'data_name' : 'label'
        }
    ]
}

emb_hparams = {
    "dim" : 64
}

rnn_layers_hparams = [
    {
        'units' : 512, 
        'return_sequences' : True
    },
    {
        'units' : 512,
        'return_sequences' : False
    }
]

output = {
    'units' : 2,
    'activation' : 'linear',
}

# default
opt = {}


