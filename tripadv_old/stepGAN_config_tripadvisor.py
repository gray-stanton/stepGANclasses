import tensorflow
checkpoint_dir='/home/gray/code/stepGAN/tripadv/ckpt-full'
restore_model=True
clear_run_logs = True
log_dir='/home/gray/code/stepGAN/tripadv/logs'
g_pretrain_epochs = 0
d_pretrain_epochs = 10
div_pretrain_epochs =0
c_pretrain_epochs = 0
d_pretrain_critic_epochs = 0
adversarial_epochs = 20
steps_per_summary = 3

gen_patience=3
es_tolerance = 0.05
disc_run_multiplier = 1
is_char=False
prior_prob=0.4
noise_size=3
max_decoding_length = 128
max_decoding_length_infer = 130
lambda_q_blend = 0
clas_loss_lambda = 0
use_unsup=False
sampling_temperature = 1.0
log_verbose_mle = True
log_verbose_rl = True
steps_per_summary = 100

pg_max_ent = False
pg_max_ent_lambda = 0.03

disc_label_smoothing_epsilon = 0.02

mle_loss_in_pg = False
mle_loss_in_pg_lambda = 0.9

adv_max_clip = 50
min_disc_pg_acc = 0.83
min_log_prob = 0.1
max_log_prob = 10

min_pg_loss = -20
max_pg_loss = 20

bleu_test = False

diversity_only = False
diversity_loss_lambda = 1
diversity_discount = 0.95
discriminator_loss_lambda = 1
norm_advantages = True

save_trained_gen = True
load_trained_gen = False
gen_ckpt_dir = '/home/gray/code/stepGAN/tripadv/'
gen_ckpt_file = '/home/gray/code/stepGAN/tripadv/ckpt-gen'

disc_crit_train_on_fake_only = True

train_lm_only = True

train_data = {
    "num_epochs": 1,
    "batch_size": 64,
    "allow_smaller_final_batch": False,
    "shuffle": True,
    "shuffle_buffer_size": None,
    "shard_and_shuffle": False,
    "num_parallel_calls": 3,
    "prefetch_buffer_size": 1,
    "max_dataset_size": 10000,
    "name": "train_data",
    'shuffle' : True,
    'dataset' :  
        {
            "files" : "./tripadvisor_train.txt",
            'vocab_file' : './tripadvisor_vocab.txt',
            'max_seq_length' : 128 ,
            'length_filter_mode' : 'truncate',
            'bos_token' : '<BOS>',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'data_name' : 'x',
            'pad_to_max_seq_length' : True
        },
}

val_data = {
    "num_epochs": 1,
    "batch_size": 64,
    "allow_smaller_final_batch": False,
    "shuffle": True,
    "shuffle_buffer_size": None,
    "shard_and_shuffle": False,
    "num_parallel_calls": 3,
    "prefetch_buffer_size": 1,
    "max_dataset_size": 2000,
    "name": "val_data",

    'dataset' :  
        {
            "files" : "./tripadvisor_val.txt",
            'vocab_file' : './tripadvisor_vocab.txt',
            'max_seq_length' : 128,
            'length_filter_mode' : 'truncate',
            'bos_token' : '<BOS>',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'data_name' : 'x',
            'pad_to_max_seq_length' : True
        }
    
}

test_data = { 
    "num_epochs": 1,
    "batch_size": 64,
    "allow_smaller_final_batch": False,
    "shuffle": True,
    "shuffle_buffer_size": None,
    "shard_and_shuffle": False,
    "num_parallel_calls": 3,
    "prefetch_buffer_size": 1,
    "max_dataset_size": 2000,
    "name": "test_data",
    'shuffle' : True,
    'dataset' :  
        {
            "files" : "./tripadvisor_test.txt",
            'vocab_file' : './tripadvisor_vocab.txt',
            'max_seq_length' : 128,
            'length_filter_mode' : 'truncate',
            'bos_token' : '<BOS>',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'data_name' : 'x',
            'pad_to_max_seq_length' : True
        }
    
}
unsup_data = { 
    "num_epochs": 1,
    "batch_size": 16,
    "allow_smaller_final_batch": True,
    "shuffle": True,
    "shuffle_buffer_size": None,
    "shard_and_shuffle": False,
    "num_parallel_calls": 1,
    "prefetch_buffer_size": 1,
    "max_dataset_size": -1,
    "seed": None,
    "name": "unsup_data",
    'shuffle' : True,
    'dataset' :  
        {
            "files" : "./unsup_reviews.txt",
            'vocab_file' : './imdb_vocab.txt',
            'max_seq_length' : 30,
            'length_filter_mode' : 'truncate',
            'bos_token' : '<BOS>',
            'delimiter' : ' ',
            'eos_token' : '<EOS>',
            'pad_to_max_seq_length' : True
        },
    
}


emb_hparams = {
    "dim": 100,
    "dropout_rate": 0.3,
    "dropout_strategy": 'element',
    "trainable": True,
    "initializer": {
        "type": "random_uniform_initializer",
        "kwargs": {
            "minval": -0.1,
            "maxval": 0.1,
            "seed": None
        }
    },
    "regularizer": {
        "type": "L1L2",
        "kwargs": {
            "l1": 0.,
            "l2": 0.
        }
    },
    "name": "word_embedder",
}

g_decoder_hparams = {
    "rnn_cell": {
            "type": tensorflow.contrib.cudnn_rnn.CudnnCompatibleGRUCell,
            "kwargs": {
                "num_units": 740,
                
            },
            "num_layers": 2,
            "dropout": {
                "input_keep_prob": 1,
                "output_keep_prob": 0.5,
                "state_keep_prob": 1.0,
                "variational_recurrent": True,
                "input_size": [emb_hparams['dim'] + noise_size + 1,
                               740]
            },
            "residual": False,
            "highway": False,
        },

    "max_decoding_length_train": None,
    "max_decoding_length_infer": None,
    "helper_train": {
        "type": "TrainingHelper",
        "kwargs": {}
    },
    "helper_infer": {
        "type": "SampleEmbeddingHelper",
        "kwargs": {}
    },
    "name": "g_decoder"
}

disc_hparams = {
    'encoder' : {

        "rnn_cell": {
               'type':tensorflow.contrib.cudnn_rnn.CudnnCompatibleGRUCell,
              'kwargs': {'num_units': 512},
              'num_layers': 2,
              'dropout': {'input_keep_prob': 1.0,
              'output_keep_prob': 0.5,
              'state_keep_prob': 1,
              'variational_recurrent': True,
              'input_size': [emb_hparams['dim'] + 1, 512],
              '@no_typecheck': ['input_keep_prob',
              'output_keep_prob',
              'state_keep_prob']},
              'residual': False,
              'highway': False,
              '@no_typecheck': ['type']},

        "output_layer": {
            "num_layers": 1,
            "layer_size": 1,
            "activation": "identity",
            "final_layer_activation": None,
            "other_dense_kwargs": None,
            "dropout_layer_ids": [],
            "dropout_rate": 0.5,
            "variational_dropout": True 
        },
        'name' : 'discriminator',
        
        }
}

div_hparams = {
    'encoder' : {

        "rnn_cell": g_decoder_hparams['rnn_cell'],

        "output_layer": {
            "num_layers": 1,
            "layer_size": None,
            "activation": "identity",
            "final_layer_activation": None,
            "other_dense_kwargs": None,
            "dropout_layer_ids": [],
            "dropout_rate": 0,
            "variational_dropout": False 
        },
        'name' : 'discriminator',
        }
}



disc_crit_hparams = {
    'units' : 1,
    'activation' : 'linear'
}


clas_hparams = {
    'encoder' : {

        "rnn_cell": {
               'type':tensorflow.contrib.cudnn_rnn.CudnnCompatibleGRUCell,
              'kwargs': {'num_units': 512},
              'num_layers': 1,
              'dropout': {'input_keep_prob': 1.0,
              'output_keep_prob': 0.5,
              'state_keep_prob': 1,
              'variational_recurrent': False,
              'input_size': [emb_hparams['dim']],
              '@no_typecheck': ['input_keep_prob',
              'output_keep_prob',
              'state_keep_prob']},
              'residual': False,
              'highway': False,
              '@no_typecheck': ['type']},

        "output_layer": {
            "num_layers": 1,
            "layer_size": 1,
            "activation": "identity",
            "final_layer_activation": None,
            "other_dense_kwargs": None,
            "dropout_layer_ids": [],
            "dropout_rate": 0.5,
            "variational_dropout": True
        },
        'name' : 'classifier',

    }
}

clas_crit_hparams = {
    'units':1,
    'activation':'linear'
}


g_opt_mle_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.001
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':5}
    },
    "gradient_noise_scale": None,
    "name": None
}

g_opt_pg_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.00005
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':5}
    },
    "gradient_noise_scale": None,
    "name": None
}

c_opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.01
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':5}
    },
    "gradient_noise_scale": None,
    "name": None
}

d_opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.001
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':50}
    },
    "gradient_noise_scale": None,
    "name": None
}

div_opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.001
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':50}
    },
    "gradient_noise_scale": None,
    "name": None
}

d_crit_opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.001
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':5}
    },
    "gradient_noise_scale": None,
    "name": None
}
c_crit_opt_hparams = {
    "optimizer": {
        "type": "AdamOptimizer",
        "kwargs": {
            "learning_rate": 0.001
        }
    },
    "learning_rate_decay": {
        "type": "",
        "kwargs": {},
        "min_learning_rate": 0.0,
        "start_decay_step": 0,
        "end_decay_step": 1e10
    },
    "gradient_clip": {
        "type": tensorflow.clip_by_global_norm,
        "kwargs": {'clip_norm':5}
    },
    "gradient_noise_scale": None,
    "name": None
}
