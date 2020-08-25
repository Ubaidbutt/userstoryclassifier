import json
from pymongo import MongoClient

config_data = {}
with open('config.json') as config_file:
    config_data = json.load(config_file)

client = MongoClient(config_data['mongodb_url'])

db = client['EdwardsStories']
story_collection = db.stories


def get_story():
    results = story_collection.find_one({}, {"_id": 0})
    return results
