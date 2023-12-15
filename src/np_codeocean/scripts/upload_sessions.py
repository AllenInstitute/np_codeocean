"""
Linked to a .exe file in the virtual env, which can be run as admin to get around
sylink-creation permissions issues.

- just edit this file, then run the `upload_sessions.exe` as admin (~/.venv/scripts/upload_sessions.exe) 
"""

import np_codeocean

split_recording_folders: set[str] = set([
    "//allen/programs/mindscope/workgroups/templeton/TTOC/2022-09-20_13-21-35_628801",
    "//allen/programs/mindscope/workgroups/templeton/TTOC/2022-09-20_14-10-18_628801",
    "//allen/programs/mindscope/workgroups/templeton/TTOC/2023-07-20_12-21-41_670181",
    "//allen/programs/mindscope/workgroups/templeton/TTOC/2023-07-25_09-47-29_670180",
    "//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_681532_20231019",
    "//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_686176_20231206", 
])

session_folders_to_upload: set[str] = set([
    
])

def main() -> None:
    for session_folder in session_folders_to_upload - split_recording_folders:
        np_codeocean.upload_session(session_folder)

if __name__ == '__main__':
    main()    