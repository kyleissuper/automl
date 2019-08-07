import json
import csv

with open("sales_scrape.json", "r") as sales_messages_json:
    messages = json.loads(sales_messages_json.read())

with open("messages.csv", "w") as messages_csv:
    writer = csv.writer(messages_csv)
    for message in messages:
        writer.writerow([message])
