# Download the human player scores and save them to a file called "human_scores.csv"

import json
import csv
import urllib.request

# URL of the JSON file
url = "https://planted.ipk-gatersleben.de/highscores/highscores.json"

# Load the JSON data from the URL
with urllib.request.urlopen(url) as response:
    data = json.loads(response.read().decode())

# Filter out entries where "timepoint" is smaller than 100
filtered_data = [entry for entry in data if entry.get("datetime_added", 0) >= 1715896799]

# Print the filtered data
print(filtered_data)

# Define the CSV columns
csv_columns = ["id", "name", "score", "datetime_added"]

# Write the filtered data to the CSV file
with open("human_scores.csv", mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    writer.writeheader()
    for entry in filtered_data:
        writer.writerow({
            "id": entry.get("id"),
            "name": entry.get("name"),
            "score": entry.get("score"),
            "datetime_added": entry.get("datetime_added")
        })