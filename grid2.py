import os
import sys
from utils.mosei import get_mosei_parser
from slp.config.config_parser import make_cli_parser, parse_config
from slp.plbind.dm import PLDataModuleFromDatasets
import ruamel.yaml
from sklearn.model_selection import ParameterGrid 
from datetime import datetime


if __name__=='__main__':
    config_path = '/home/poulinakis/Multimodal-Barlow-Twins/configs/my-config_2.yml'
    # load the config
    yaml = ruamel.yaml.YAML()
    # yaml.preserve_quotes = True
    with open(config_path) as fp:
        data = yaml.load(fp)

    experiment_name = 'SSL PERFORMANCE PERCENT' #'SSL MASKING NEW'

    # Define a search space for the parameters
    params = {'hidden_size': [100], # model
              'num_layers': [1],
              'bidirectional': [False],
              'dropout': [0.2],
              'weight_decay_ssl': [0.0],   # optimization
              'weight_decay': [0.0],
              'lr_ssl': [3e-6],
              'lr': [ 3e-4],
              'freeze_grads': [False],
              'alpha': [2e-2], #5e-2, 1e-1],

              # SSL transformations 
              'gauss_noise_p': [[0.0, 0.0]], # [[0.0, 0.0], [0.5, 0.5], [0.7, 0.2], [0.9, 0.1]],
              'gauss_noise_std': [[0, 0]], # [[1.0, 1.0], [2.0, 2.0], [0.01, 0.01], [0.2, 0.2]],
              'masking_p' :    [[0.0, 0.0]],
              'mask_percentage': [[0.0, 0.0]],#[[0.2, 0.0], [0.5, 0.0], [0.7, 0.0]],  #[[0.2, 0.2], [0.5, 0.5], [0.7, 0.7], [0.5, 0.2], [0.7, 0.2], [1.0, 0.2], [1.0, 0.7]], #
              'masking_mode': ['timestep'],  #timestep and feature
               
              'batch_size': [32, ],
              'batch_size_ssl': [170], # [10, 25, 50, 80, 120, 145, 170],

              'max_epochs': [100],    # epochs for fine tuning                                                   # Standard is 100 with early stopping
              'max_epochs_ssl':  [20], #[2, 5, 20], #[20], #,   # no early stopping is executed       # Standard is 20
           
              'data_percentage': [0.01, 0.1, 0.3, 0.6, 0.8, -1], # -1 for full dataset
              'data_percentage_ssl': [-1], #[0.001],   # -1 for full , 0.001 for "nothing"

              'transformation_order': [ ['noise'] ]
           }

    
    data['trainer']['experiment_name'] = experiment_name
    data['trainer_ssl']['experiment_name'] = experiment_name

    for i, parameters in enumerate(list(ParameterGrid(params))):
        fr = parameters['freeze_grads']
        data['run_name'] = f'freeze_grads={fr}_{str(datetime.now())}'
        data['tune']['freeze_grads'] = parameters['freeze_grads']
        data['trainer']['max_epochs'] = parameters['max_epochs']
        data['trainer_ssl']['max_epochs'] = parameters['max_epochs_ssl']

        # change the configuration file's values
        data['model']['hidden_size'] = parameters['hidden_size']
        data['model']['num_layers'] = parameters['num_layers']
        data['model']['bidirectional'] = parameters['bidirectional']
        data['model']['dropout'] = parameters['dropout']

        data['ssl_optimization']['optim']['weight_decay'] = parameters['weight_decay_ssl']
        data['ssl_optimization']['optim']['lr'] = parameters['lr_ssl']
        data['optimization']['optim']['weight_decay'] = parameters['weight_decay']
        data['optimization']['optim']['lr'] = parameters['lr']
        data['optimization']['optim']['lr'] = parameters['lr']
        
        data['barlow_twins']['alpha'] = parameters['alpha']

        # batch size
        data['data']['batch_size'] = parameters['batch_size']
        data['data']['batch_size_eval'] = parameters['batch_size']
        data['data']['data_percentage'] = parameters['data_percentage']
        data['data_ssl']['batch_size'] = parameters['batch_size_ssl']
        data['data_ssl']['batch_size_eval'] = parameters['batch_size_ssl']
        data['data_ssl']['data_percentage'] = parameters['data_percentage_ssl']

        # SSL transformations
        data['transformations']['order'] = parameters['transformation_order']
        data['transformations']['gauss_noise_p'] = parameters['gauss_noise_p']
        data['transformations']['gauss_noise_std'] = parameters['gauss_noise_std']
        data['transformations']['masking_p'] = parameters['masking_p']
        data['transformations']['mask_percentage'] = parameters['mask_percentage']
        data['transformations']['masking_mode'] = parameters['masking_mode']

        # save new config values 
        with open(config_path, 'w') as fp:
            yaml.dump(data, fp)

        # Parse updated config file
        parser = get_mosei_parser()
        parser = make_cli_parser(parser, PLDataModuleFromDatasets)
        config = parse_config(parser, config_path)

        # Call training script
        command = f'python3 experiments/mm.py --config {config_path} --gpus 1 '
        os.system(command)
