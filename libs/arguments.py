import argparse
import yaml


def save_arguments_to_yaml(args, filepath):
    with open(filepath, 'w') as file:
        yaml.dump(vars(args), file)


def load_arguments_from_yaml(filepath):
    with open(filepath, 'r') as file:
        args_dict = yaml.safe_load(file)
    return argparse.Namespace(**args_dict)


def merge_args_with_yaml(args, yaml_args):
    # Convert Namespace object to dictionary
    args_dict = vars(args)

    # Merge YAML args into the command-line args dictionary
    args_dict.update(vars(yaml_args))

    # Convert the dictionary back to a Namespace object
    merged_args = argparse.Namespace(**args_dict)

    return merged_args


def parse_arguments():
    parser = argparse.ArgumentParser(description='Argument Controller')
    parser.add_argument('--control_yaml', type=str, default='configs/base_control.yaml',
                        help='yaml path to load configs')
    parser.add_argument('--train_yaml', type=str, default='configs/base_rno.yaml',
                        help='yaml path to load configs')
    parser.add_argument('--set_re', type=int, default=-1, help='reynolds number to generate data')
    parser.add_argument('--set_epoch', type=int, default=-1, help='set training epochs')
    parser.add_argument('--force_close_wandb', action='store_true', help='close wandb log.')
    # Add to your argument parser
    parser.add_argument('--load_observer', action='store_true', help='Load a pre-trained observer model')
    parser.add_argument('--observer_model_path', type=str, default='./outputs/observer_model.pth', help='Path to pre-trained observer model')
    parser.add_argument('--load_policy', action='store_true', help='Load a pre-trained policy model')
    parser.add_argument('--policy_model_path', type=str, default='./outputs/policy_model.pth', help='Path to pre-trained policy model')
    args = parser.parse_args()
    return args
