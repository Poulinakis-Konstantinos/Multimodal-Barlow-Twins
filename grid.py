import os
import sys
from utils.mosei import get_mosei_parser
from slp.config.config_parser import make_cli_parser, parse_config
from slp.plbind.dm import PLDataModuleFromDatasets
import ruamel.yaml
from sklearn.model_selection import ParameterGrid 
from datetime import datetime


if __name__=='__main__':
    config_path = '/home/poulinakis/Multimodal-Barlow-Twins/configs/my-config.yml'
    # load the config
    yaml = ruamel.yaml.YAML()
    # yaml.preserve_quotes = True
    with open(config_path) as fp:
        data = yaml.load(fp)

    experiment_name = 'Testing checkpoints'

    # Define a search space for the parameters
    params = {'hidden_size': [100], # model
              'num_layers': [1],
              'bidirectional': [False],
              'dropout': [0.2],
              'weight_decay_ssl': [0.0],   # optimization
              'weight_decay': [0.0],
              'lr_ssl': [5e-4],
              'lr': [5e-3],
              'freeze_grads': [False],
              'gauss_noise_p': [[0.5, 0.5], [0.7, 0.2]], #[[0.0, 0.0], [0.5, 0.5], , [0.9, 0.1]],
              
              'batch_size': [256],
              'batch_size_ssl': [178],

              'max_epochs': [1],    # epochs for fine tuning
              'max_epochs_ssl': [2], #[2, 5, 10, 30, 40 ,50, 60 , 100],   # patience is set at 100 so all epochs are executed.
           
              'data_percentage': [10] # provide absolute value of samples (for now)
           }

    
    data['trainer']['experiment_name'] = experiment_name
    data['trainer_ssl']['experiment_name'] = experiment_name

    for i, parameters in enumerate(list(ParameterGrid(params))):
        fr = parameters['freeze_grads']
        data['run_name'] = f'freeze_grads={fr}_{str(datetime.now())}'
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
        
        data['tune']['freeze_grads'] = parameters['freeze_grads']

        # batch size
        data['data']['batch_size'] = parameters['batch_size']
        data['data']['batch_size_eval'] = parameters['batch_size']
        data['data']['data_percentage'] = parameters['data_percentage']
        data['data_ssl']['batch_size'] = parameters['batch_size_ssl']
        data['data_ssl']['batch_size_eval'] = parameters['batch_size_ssl']

        # SSL transformations
        data['transformations']['gauss_noise_p'] = parameters['gauss_noise_p']

        # save new config values 
        with open(config_path, 'w') as fp:
            yaml.dump(data, fp)

        # Parse updated config file
        parser = get_mosei_parser()
        parser = make_cli_parser(parser, PLDataModuleFromDatasets)
        config = parse_config(parser, config_path)

        # Call training script
        command = f'python3 experiments/mm.py --config {config_path} --gpus 1 --offline'
        os.system(command)
    #subprocess.call(["python", "experiments/mm.py", "--config configs/my-config.yml", "--offline"])