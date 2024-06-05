import yaml
import argparse

def parse_args():
    '''
    Parses command line arguments for specifying the config file.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_train.yaml", help="config file to run(default: config_train.yml)")
    return parser.parse_args()

def load_config(config_file):
    '''
    Loads the configuration from a YAML file.
    '''
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config