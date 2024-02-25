import os
import json
import random


# Split the prepcrocessed data into training and testing datasets

def read_dataset(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_to_folders(dataset, base_path):
    locations = ['home', 'work', 'public']
    time_categories = ['wd-evening', 'wd-morning', 'we-anytime', 'wd-anytime',
                       'we-evening', 'we-morning', 'wd-afternoon']
    
    # Create base directories for each location and time category
    for location in locations:
        for time_category in time_categories:
            os.makedirs(os.path.join(base_path, 'training', location, time_category), exist_ok=True)
            os.makedirs(os.path.join(base_path, 'testing', location, time_category), exist_ok=True)

    random.shuffle(dataset)  # Shuffle the dataset randomly
    split_index = int(len(dataset) * 0.8)  # 80% for training, 20% for testing

    # Function to save tasks in respective folders
    def save_tasks(tasks, split_type):
        for task in tasks:
            for location in task['location']:
                if location in locations:
                    for time in task['time']:
                        if time in time_categories:
                            folder_path = os.path.join(base_path, split_type, location, time)
                            file_name = f"{task['task'].replace(' ', '_')}_{location}_{time}.json"
                            file_path = os.path.join(folder_path, file_name)
                            with open(file_path, 'w') as f:
                                json.dump(task, f)

    save_tasks(dataset[:split_index], 'training')
    save_tasks(dataset[split_index:], 'testing')

dataset_file_path = 'Eventify/dataset/preprocessed_data.json'
dataset = read_dataset(dataset_file_path)
base_path = './Eventify/dataset/'  

save_to_folders(dataset, base_path)
