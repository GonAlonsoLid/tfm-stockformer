"""Centralized configuration loading for all Stockformer scripts.

Reads an INI config file and returns an argparse.Namespace with all
parameters from [train], [data], [param], and [file] sections.

Usage:
    from lib.config import load_config
    args = load_config("config/Multitask_Stock_SP500.conf")
"""

import argparse
import configparser


def load_config(config_path: str, extra_args: list[str] | None = None) -> argparse.Namespace:
    """Load config from INI file and return as argparse.Namespace.

    Parameters
    ----------
    config_path : str
        Path to the .conf INI file.
    extra_args : list[str] | None
        Additional CLI arguments to parse (e.g. ['--checkpoint', 'path']).

    Returns
    -------
    argparse.Namespace
        Namespace with all config values as attributes.
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    parser = argparse.ArgumentParser(add_help=False)

    # [train] section
    parser.add_argument('--cuda', type=str, default=config['train']['cuda'])
    parser.add_argument('--seed', type=int, default=config['train']['seed'])
    parser.add_argument('--batch_size', type=int, default=config['train']['batch_size'])
    parser.add_argument('--max_epoch', type=int, default=config['train']['max_epoch'])
    parser.add_argument('--learning_rate', type=float, default=config['train']['learning_rate'])

    # [data] section
    parser.add_argument('--Dataset', default=config['data']['dataset'])
    parser.add_argument('--T1', type=int, default=config['data']['T1'])
    parser.add_argument('--T2', type=int, default=config['data']['T2'])
    parser.add_argument('--train_ratio', type=float, default=config['data']['train_ratio'])
    parser.add_argument('--val_ratio', type=float, default=config['data']['val_ratio'])
    parser.add_argument('--test_ratio', type=float, default=config['data']['test_ratio'])

    # [param] section
    parser.add_argument('--L', type=int, default=config['param']['layers'])
    parser.add_argument('--h', type=int, default=config['param']['heads'])
    parser.add_argument('--d', type=int, default=config['param']['dims'])
    parser.add_argument('--j', type=int, default=config['param']['level'])
    parser.add_argument('--s', type=float, default=config['param']['samples'])
    parser.add_argument('--w', default=config['param']['wave'])
    parser.add_argument('--max_features', type=int, default=config['param'].get('max_features', '0'))
    parser.add_argument('--decomposition', default=config['param'].get('decomposition', 'dwt'))
    parser.add_argument('--stl_period', type=int, default=config['param'].get('stl_period', '5'))
    parser.add_argument('--graph_type', default=config['param'].get('graph_type', 'static'))

    # [file] section
    parser.add_argument('--traffic_file', default=config['file']['traffic'])
    parser.add_argument('--indicator_file', default=config['file']['indicator'])
    parser.add_argument('--adj_file', default=config['file'].get('adj', ''))
    parser.add_argument('--adjgat_file', default=config['file']['adjgat'])
    parser.add_argument('--model_file', default=config['file']['model'])
    parser.add_argument('--log_file', default=config['file'].get('log', ''))
    parser.add_argument('--alpha_360_dir', default=config['file']['alpha_360_dir'])
    parser.add_argument('--output_dir', default=config['file']['output_dir'])
    parser.add_argument('--tensorboard_dir', default=config['file'].get('tensorboard_dir', ''))

    args = parser.parse_args(extra_args or [])
    args.config = config_path
    return args
