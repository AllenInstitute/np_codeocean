# np_codeocean
Tools for uploading Mindscope Neuropixels experiments to S3 (for Code Ocean).

Requires running as admin on Windows in order to create remote-to-remote symlinks
on the Isilon.

## Development workflow
- clone the repo, or pull with rebase
- sync the development environment using `uv` (see below)
- push changes to main (always commit changes to `uv.lock`)
- github action formats and publishes a new version
- pull the bump commit

## Install 
Setup/sync the development environment for working on a specific project:
```shell
uv sync --extra <dynamicrouting|openscope>
```
This ensures all developers are using the same package versions.

## Add dependencies
For shared utilities:
```shell
uv add <package-name>
```

For project-specific utilities (added to optional dependency groups):
```shell
uv add <package-name> --optional <dynamicrouting|openscope>
```

## Update dependencies
All:
```shell
uv lock --upgrade
```

Single package:
```shell
uv lock --upgrade-package  <package-name>
```

## Usage 
- `upload` CLI tool is provided, which uses the
  [`np_session`](https://github.com/AllenInstitute/np_session) interface to find
  and upload
  raw data for one ecephys session:

    ```
    pip install np_codeocean
    upload <session-id>
    ```
 
    where session-id is any valid input to `np_session.Session()`, e.g.: 
    - a lims ID (`1333741475`) 
    - a workgroups foldername (`DRPilot_366122_20230101`) 
    - a path to a session folder (    `\\allen\programs\mindscope\workgroups\np-exp\1333741475_719667_20240227`)
    
- a folder of symlinks pointing to the raw data is created, with a new structure suitable for the KS2.5 sorting pipeline on Code Ocean
- the symlink folder, plus metadata, are entered into a csv file, which is
  submitted to [`http://aind-data-transfer-service`](http://aind-data-transfer-service), which in turn runs the
  [`aind-data-transfer`](https://github.com/AllenNeuralDynamics/aind-data-transfer)
  tool on the HPC, which follows the symlinks to the original data,
  median-subtracts/scales/compresses ephys data, then uploads with the AWS CLI tool
- all compression/zipping acts on copies in temporary folders: the original raw data is not altered in anyway 
