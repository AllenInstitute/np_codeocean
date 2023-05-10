from __future__ import annotations

import configparser
import json
import os
import pathlib
from typing import Literal

import np_config

CONFIG = np_config.fetch('/projects/np_codeocean')
"""Config for this project"""

AWS_CONFIG: dict[Literal['aws_access_key_id', 'aws_secret_access_key'], str] = np_config.fetch('/projects/np_codeocean/aws')['config']
"""Config for connecting to AWS/S3 via awscli/boto3"""

AWS_CREDENTIALS: dict[Literal['domain', 'token'], str]  = np_config.fetch('/projects/np_codeocean/aws')['credentials']
"""Config for connecting to AWS/S3 via awscli/boto3"""

CODEOCEAN_CONFIG: dict[Literal['region'], str] = np_config.fetch('/projects/np_codeocean/codeocean')['credentials']
"""Config for connecting to CodeOcean via http API"""


def get_home() -> pathlib.Path:
    if os.name == 'nt':
        return pathlib.Path(os.environ['USERPROFILE'])
    return pathlib.Path(os.environ['HOME'])

def get_aws_files() -> dict[Literal['config', 'credentials'], pathlib.Path]:
    return {
        'config': get_home() / '.aws' / 'config',
        'credentials': get_home() / '.aws' / 'credentials',
    }

def get_codeocean_files() -> dict[Literal['credentials'], pathlib.Path]:
    return {
        'credentials': get_home() / '.codeocean' / 'credentials.json',
    }

def verify_ini_config(path: pathlib.Path, contents: dict, profile: str = 'default') -> None:
    config = configparser.ConfigParser()
    if path.exists():
        config.read(path)
    if not all(k in config[profile] for k in contents):
        raise ValueError(f'Profile {profile} in {path} exists but is missing some keys required for codeocean or s3 access.')
    
def write_or_verify_ini_config(path: pathlib.Path, contents: dict, profile: str = 'default') -> None:
    config = configparser.ConfigParser()
    if path.exists():
        config.read(path)
        try:    
            verify_ini_config(path, contents, profile)
        except ValueError:
            pass
        else:   
            return
    config[profile] = contents
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with path.open('w') as f:
        config.write(f)
    verify_ini_config(path, contents, profile)

def verify_json_config(path: pathlib.Path, contents: dict) -> None:
    config = json.loads(path.read_text())
    if not all(k in config for k in contents):
        raise ValueError(f'{path} exists but is missing some keys required for codeocean or s3 access.')
    
def write_or_verify_json_config(path: pathlib.Path, contents: dict) -> None:
    if path.exists():
        try:
            verify_json_config(path, contents)
        except ValueError:
            contents = np_config.merge(json.loads(path.read_text()), contents)
        else:   
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    path.write_text(json.dumps(contents, indent=4))
    
def ensure_credentials() -> None:
    for file, contents in (
        (get_aws_files()['config'], AWS_CONFIG),
        (get_aws_files()['credentials'], AWS_CREDENTIALS),
    ):
        write_or_verify_ini_config(file, contents, profile='default')
    
    for file, contents in (
        (get_codeocean_files()['credentials'], CODEOCEAN_CONFIG),
    ):
        write_or_verify_json_config(file, contents)

if __name__ == '__main__':
    ensure_credentials()