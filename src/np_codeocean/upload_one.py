from __future__ import annotations

import csv
import datetime
import pathlib
import sys
from pathlib import Path
from typing import ClassVar, NamedTuple

from aind_data_transfer.jobs.s3_upload_job import GenericS3UploadJobList
from aind_data_schema.data_description import ExperimentType
from aind_data_transfer.config_loader.base_config import BasicJobEndpoints
from aind_data_transfer.jobs.basic_job import BasicJob, BasicUploadJobConfigs
from aind_data_transfer.util.s3_utils import upload_to_s3, copy_to_s3
import np_config
import np_logging
import np_session
import np_tools

import np_codeocean.utils as utils 

logger = np_logging.get_logger(__name__)

CONFIG = np_config.fetch('/projects/np_codeocean')
JOB_LIST_FILENAME_PREFIX = 'codeocean_upload'

class ConfigWithExtendedS3Prefix(BasicUploadJobConfigs):
    
    _extension: ClassVar[str]
    """Some tag to add to the end of the bucket name, e.g.
    `_curated_<datetime>`"""
    
    def __init__(self, s3_bucket_name_extension: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__class__._extension = s3_bucket_name_extension

    @property
    def s3_prefix(self) -> str:
        extension = f'_{self._extension}' if self._extension[0] != '_' else self._extension
        return super().s3_prefix + extension
    
    
class EcephysJobListWithoutRawData(GenericS3UploadJobList):
    
    def __init__(self, s3_bucket_name_extension: str, *args):
        self.extension = s3_bucket_name_extension
        super().__init__(*args)
        
    def _create_job_config_list(self) -> list[BasicJob]:
        """Reads in the csv file and outputs a list of Job Configs."""
        job_list = list()
        param_store_name = None
        job_endpoints = {}
        with open(self.configs.jobs_csv_file, newline="") as csvfile:
            reader = csv.DictReader(csvfile, skipinitialspace=True)
            for row in reader:
                cleaned_row = {
                    k.strip().replace("-", "_"): self._clean_csv_entry(
                        k.strip().replace("-", "_"), v
                    )
                    for k, v in row.items()
                }
                cleaned_row["acq_date"] = BasicUploadJobConfigs.parse_date(
                    cleaned_row["acq_date"]
                )
                cleaned_row["acq_time"] = BasicUploadJobConfigs.parse_time(
                    cleaned_row["acq_time"]
                )
                # Override with flags set in command line
                if self.configs.dry_run is True:
                    cleaned_row["dry_run"] = True
                if self.configs.compress_raw_data is True:
                    cleaned_row["compress_raw_data"] = True
                # Avoid downloading endpoints from aws multiple times
                if cleaned_row.get("aws_param_store_name") is not None:
                    # Check if param store is defined in previous row
                    if cleaned_row["aws_param_store_name"] == param_store_name:
                        cleaned_row.update(job_endpoints)
                    # Otherwise, download it from aws
                    else:
                        job_endpoints = BasicJobEndpoints(
                            aws_param_store_name=(
                                cleaned_row["aws_param_store_name"]
                            )
                        ).dict()
                        cleaned_row.update(job_endpoints)
                        param_store_name = cleaned_row["aws_param_store_name"]
                    del cleaned_row["aws_param_store_name"]

                #! the original method switches here to create an EcephysJob:
                # that isn't appropriate for non-raw data, so we just create
                # the default general-purpose BasicJob instead
                configs_from_row = ConfigWithExtendedS3Prefix(self.extension, **cleaned_row)
                new_job = BasicJob(job_configs=configs_from_row)
                job_list.append(new_job)
        return job_list


class OneOffCodeOceanUpload(NamedTuple):
    """Objects required for uploading data associated with a Mindscope Neuropixels to CodeOcean.
    
    `source` can be a file or folder of data to upload.
    """
    session: np_session.Session
    """Session object that the paths belong to."""
    
    source: Path
    """Path to file or directory to upload."""
    
    job: Path
    """File containing job parameters for `aind-data-transfer`"""


def get_ephys_upload_csv_for_session(session: np_session.Session, source: Path) -> dict[str, str | int]:
    return {
        'data-source': np_config.normalize_path(source).as_posix(),
        's3-bucket': CONFIG['s3-bucket'],
        'subject-id': str(session.mouse),
        'experiment-type': 'ecephys',
        'modality': 'ECEPHYS',
        'acq-date': f'{session.date:%Y-%m-%d}',
        'acq-time': f'{session.start:%H-%M-%S}',
        'aws-param-store-name': CONFIG['aws-param-store-name'],
    } # type: ignore


def create_upload_job(session: np_session.Session, job: Path, source: Path) -> None:
    logger.info(f'Creating upload job file {job} for session {session}...')
    _csv = get_ephys_upload_csv_for_session(session, source)
    with open(job, 'w') as f:
        w = csv.writer(f)
        w.writerow(_csv.keys())
        w.writerow(_csv.values())    


def create_codeocean_upload(source: str | Path) -> OneOffCodeOceanUpload:
    """Create upload object
    - job file for feeding into `aind-data-transfer`
    """
    
    session = np_session.Session(source)
    
    upload = OneOffCodeOceanUpload(
        session = session, 
        source = np_config.normalize_path(source),
        job = np_config.normalize_path(source) / get_new_job_list_filename(),
        )
    create_upload_job(upload.session, upload.job, upload.source)    
    return upload

def get_new_job_list_filename() -> str:
     return f'{JOB_LIST_FILENAME_PREFIX}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.csv'

def get_timestamp_from_job_list_filename(filename: str) -> str:
    return filename.split(f'{JOB_LIST_FILENAME_PREFIX}_')[-1]

def get_s3_prefix(upload: OneOffCodeOceanUpload) -> str:
    d = get_ephys_upload_csv_for_session(upload.session, upload.source)
    return f"{d['experiment-type']}_{d['subject-id']}_{d['acq-date']}_{d['acq-time']}"
    
def upload(source: str | pathlib.Path, tag: str) -> None:
    utils.ensure_credentials()
    upload = create_codeocean_upload(source)
    tag = f"{tag}{'_' if tag[-1] != '_' else ''}{get_timestamp_from_job_list_filename(upload.job.stem)}"
    np_logging.web('np_codeocean').info(f'Uploading {upload.source}')
    # EcephysJobListWithoutRawData(tag, ["--jobs-csv-file", upload.job.as_posix()]).run_job()
    if upload.source.is_dir():
        fn = upload_to_s3
    else: 
        fn = copy_to_s3
    fn(
        source,
        s3_bucket=get_ephys_upload_csv_for_session(upload.session, upload.source)['s3-bucket'],
        s3_prefix=f"{get_s3_prefix(upload)}_{tag}", 
        dryrun=False,
    )
    np_logging.web('np_codeocean').info(f'Finished uploading {upload.source}')
    
def main() -> None:
    upload(*sys.argv[1:])
    
if __name__ == '__main__':
    main()