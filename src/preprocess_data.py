import json

# Take the ms-latte dataset and preprocess it to extract the task, location and time information

def preprocess_entry(entry):
    task = entry["TaskTitle"].lower()
    locations_set = set()
    for loc in entry["LocJudgements"]:
        if loc["Known"] == "yes" and loc["Locations"]:
            locations_set.update([location.strip().lower() for location in loc["Locations"].split(',')])

    times_set = set()
    for time in entry["TimeJudgements"]:
        if time["Known"] == "yes" and time["Times"]:
            times_set.update([time.strip().lower() for time in time["Times"].split(',')])
    
    return {"task": task, "location": list(locations_set), "time": list(times_set)}

def process_dataset(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    preprocessed_data = [preprocess_entry(entry) for entry in data]
    
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(preprocessed_data, file, indent=2)

input_file = './dataset/MS-LaTTE.json' 
output_file = './dataset/preprocessed_data.json'
process_dataset(input_file, output_file)

print("Preprocessing complete. Output saved to:", output_file)
