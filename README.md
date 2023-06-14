# np_codeocean
Tools for uploading and interacting with Mindscope Neuropixels experiments on Code Ocean

Currently requires the AWS CLI tool to be installed (not the Python package).

- `upload()` is provided, which uses the [`np_session`](https://github.com/AllenInstitute/np_session) interface to find raw data for an ecephys session
- a folder of symlinks pointing to the raw data is created, with a new structure suitable for the KS2.5 sorting pipeline on Code Ocean
- the symlink folder, plus metadata, is fed into the [`aind-data-transfer`](https://github.com/AllenNeuralDynamics/aind-data-transfer) tool, which follows the symlinks to the original data, compresses ephys data and zips video data, then uploads via the AWS CLI tool
- all compression/zipping acts on new files in temporary folders: the original raw data is not altered in anyway 
