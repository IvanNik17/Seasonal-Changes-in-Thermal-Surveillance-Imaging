from argparse import Namespace

# Large
hparams = Namespace(**{# data
                       'model': 'VQVAE2',
                       'dataset': 'day',
                       'season': 'feb',
                       'img_dir': '/home/aau/github/data/thermal/sensor_paper',
                       'image_width': 384,
                       'image_height': 288,
                       'train_selection': None,
                       'test_selection': None,
                       'get_metadata': False,
                       # model
                       'in_channels':              1,
                       'hidden_channels':          128,
                       'res_channels':             32,
                       'nb_res_layers':            2,
                       'embed_dim':                64,
                       'nb_entries':               512,
                       'nb_levels':                3,
                       'scaling_rates':            [4, 2, 2],
                       # training
                       'log_dir': 'lightning_logs',
                       'gpus': 1,
                       'max_epochs': 200,
                       'learning_rate': 1e-4,
                       'beta': 0.25,
                       'beta1': 0.9,
                       'beta2': 0.999,
                       'batch_size': 128,
                       'num_workers':12})

