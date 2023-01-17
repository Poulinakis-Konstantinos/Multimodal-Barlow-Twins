import os
import sys
from utils.mosei import get_mosei_parser
from slp.config.config_parser import make_cli_parser, parse_config
from slp.plbind.dm import PLDataModuleFromDatasets
import ruamel.yaml
from sklearn.model_selection import ParameterGrid 

if __name__=='__main__':
    config_path = '/home/poulinakis/Multimodal-Barlow-Twins/configs/my-config.yml'
    # load the config
    yaml = ruamel.yaml.YAML()
    # yaml.preserve_quotes = True
    with open(config_path) as fp:
        data = yaml.load(fp)

    # Define a search space for the parameters
    params = {'hidden_size': [50, 70], # model
              'num_layers': [1, 4],
              'bidirectional': [False, True],
              'dropout': [0.2, 0.5],
              'weight_decay_ssl': [0.0, 0.3],   # optimization
              'weight_decay': [0.0, 0.5],
              'lr_ssl': [5e-2, 5e-4],
              'lr': [5e-3],

              'freeze_grads': [True]
            }

    for i, parameters in enumerate(list(ParameterGrid(params))):
        data['trainer']['experiment_name'] = 'Gauss_Noise-Grid_Model_params'
        fr = parameters['freeze_grads']

        # change the configuration file's values
        data['model']['hidden_size'] = parameters['hidden_size']
        data['model']['num_layers'] = parameters['num_layers']
        data['model']['bidirectional'] = parameters['bidirectional']
        data['model']['dropout'] = parameters['dropout']

        data['ssl_optimization']['optim']['weight_decay'] = parameters['weight_decay_ssl']
        data['ssl_optimization']['optim']['lr'] = parameters['lr_ssl']
        data['optimization']['optim']['weight_decay'] = parameters['weight_decay']
        data['optimization']['optim']['lr'] = parameters['lr']

        data['run_name'] = f'grid_model_{i}_freeze_grads={fr}'
        data['tune']['freeze_grads'] = parameters['freeze_grads']

        # save new config values 
        with open(config_path, 'w') as fp:
            yaml.dump(data, fp)

        # Parse updated config file
        parser = get_mosei_parser()
        parser = make_cli_parser(parser, PLDataModuleFromDatasets)
        config = parse_config(parser, config_path)

        # Call training script
        command = f'python3 experiments/mm.py --config configs/my-config.yml --gpus 1 '
        os.system(command)
    #subprocess.call(["python", "experiments/mm.py", "--config configs/my-config.yml", "--offline"])