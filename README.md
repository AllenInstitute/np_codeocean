# np_codeocean
Tools for uploading Mindscope Neuropixels experiments to S3 (for Code Ocean).

Requires running as admin on Windows in order to create remote-to-remote symlinks
on the Isilon.

## Install 
For Dynamic Routing:
```shell
uv sync --extra dynamicrouting
```

For OpenScope:
```shell
uv sync --extra openscope
```

## Add dependencies
For shared utilities:
```shell
uv add <package-name>
```

For Dynamic Routing or OpenScope specific utilities:
```shell
uv add <package-name> --optional <dynamicrouting|openscope>
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
