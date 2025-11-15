import os
import json
import pandas as pd
from datetime import datetime, timezone


def normalize_timestamp(timestamp):
    try:
        # Case 1: Check if it's a Unix timestamp in milliseconds or seconds      
        if isinstance(timestamp, (int, float)) or (isinstance(timestamp, str) and timestamp.isdigit()):
            timestamp = int(timestamp)  # Ensure it's an integer
            if len(str(timestamp)) == 13:  # Milliseconds
                return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            elif len(str(timestamp)) == 10:  # Seconds
                return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        
        # Case 2: Check if it's already in a recognizable datetime format
        elif isinstance(timestamp, str):
            # Try parsing with known formats
            for fmt in ['%Y-%m-%d %H:%M', '%Y-%m-%d %H:%M:%S']:
                try:
                    return datetime.strptime(timestamp, fmt).strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    continue
        
        # If none of the above cases match, raise an error
        raise ValueError(f"Unrecognized timestamp format: {timestamp}")
    
    except Exception as e:
        print(f"Error normalizing timestamp: {e}")
        return None


def parse_labels(json_file, folder_path):
    with open(os.path.join(folder_path, json_file), 'r') as file:
        data = json.load(file)

    folder_dict = {}

    for item in data:
        # Determine format
        if 'annotations' in item:
            file_name = item['data']['csv']
            # Remove hash: split by '-' and take from second part
            file_name_clean = '-'.join(file_name.split('-')[1:]) if '-' in file_name else file_name
            for annotation in item['annotations']:
                for result in annotation['result']:
                    label_list = result['value']['timeserieslabels']
                    # Convert list to string (take first if single, or join)
                    label = label_list[0] if len(label_list) == 1 else ' '.join(label_list)
                    start_time = result['value']['start']
                    end_time = result['value']['end']
                    if file_name_clean not in folder_dict:
                        folder_dict[file_name_clean] = []
                    folder_dict[file_name_clean].append({
                    'label': label,
                    'start': normalize_timestamp(start_time),
                    'end': normalize_timestamp(end_time)
                    })
        elif 'label' in item:
            file_name = item['csv']
            # Remove hash: split by '-' and take from second part
            file_name_clean = '-'.join(file_name.split('-')[1:]) if '-' in file_name else file_name
            for label_entry in item['label']:
                label_list = label_entry['timeserieslabels']
                # Convert list to string (take first if single, or join)
                label = label_list[0] if len(label_list) == 1 else ' '.join(label_list)
                start_time = label_entry['start']
                end_time = label_entry['end']
                if file_name_clean not in folder_dict:
                    folder_dict[file_name_clean] = []
                folder_dict[file_name_clean].append({
                    'label': label,
                    'start': normalize_timestamp(start_time),
                    'end': normalize_timestamp(end_time)
                })
        else:
            print(f"Unknown format for item ID: {item.get('id', 'N/A')}")
    
    return folder_dict


def get_data_with_labels(data_path):
    folders = os.listdir(data_path)
    folders.remove('consensus')
    folders.remove('sample')

    all_labels = {}
    for folder in folders:
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

            if not json_files:
                print(f'No JSON file found in folder: {folder}')
                continue
            print(f'Folder: {folder}, JSON files: {json_files}')

            folder_dict = {}
            for json_file in json_files:
                parsed_labels = parse_labels(json_file, folder_path)
                for file_name, labels in parsed_labels.items():
                    if file_name not in folder_dict:
                        folder_dict[file_name] = []
                    folder_dict[file_name].extend(labels)

            # Deduplicate labels within each file
            for file_name, labels in folder_dict.items():
                unique_labels = {frozenset(label.items()): label for label in labels}.values()
                folder_dict[file_name] = list(unique_labels)

            # Add the folder's data to the main dictionary
            all_labels[folder] = folder_dict

    data_with_labels = {}

    for folder, files in all_labels.items():
        folder_path = os.path.join(data_path, folder)
        for file_name, labels in files.items():
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                # Read the data file
                data = pd.read_csv(file_path)
                # Attach the labels to the data
                data_with_labels[file_name] = {
                    'data': data,
                    'labels': labels
                }

    return data_with_labels
