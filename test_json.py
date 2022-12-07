"""Test loading the DC blocker JSON file"""

import json

# Open the JSON file
directory = "./test/data/"
file_name = "b4gic_blocker_placement_results.json"
path = directory + file_name
dc_blocker_file = open(path)

# Load the data as a dictionary
dc_placement = json.load(dc_blocker_file)

print(list(dc_placement["input"]["gmd_bus"].values()))