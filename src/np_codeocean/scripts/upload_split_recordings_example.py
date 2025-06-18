"""
Linked to a .exe file in the virtual env, which can be run as admin to get around
sylink-creation permissions issues.

- just edit this file, then run the `upload_sessions.exe` as admin (~/.venv/scripts/upload_sessions.exe) 
"""

import np_codeocean

split_recordings: dict[str, tuple[str, ...]] = {
    "//allen/programs/mindscope/workgroups/templeton/TTOC/2022-09-20_13-21-35_628801": (),
    "//allen/programs/mindscope/workgroups/templeton/TTOC/2022-09-20_14-10-18_628801": (),
    "//allen/programs/mindscope/workgroups/templeton/TTOC/2023-07-20_12-21-41_670181": (),
    "//allen/programs/mindscope/workgroups/templeton/TTOC/2023-07-25_09-47-29_670180": (),
    "//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_681532_20231019": ('recording1', 'recording2'),
    "//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_686176_20231206": ('recording1', 'recording2'), 
}
split_recording_folders: set[str] = set(split_recordings.keys())

session_folders_to_upload: set[str] = set([
    
])

def main() -> None:
    for session_folder in session_folders_to_upload - split_recording_folders:
        np_codeocean.upload_session_v2(session_folder)
        
    for session_folder, recording_dir_names in split_recordings.items():
        if recording_dir_names:
            np_codeocean.upload_session_v2(session_folder, recording_dir_names)
        
if __name__ == '__main__':
    main()    