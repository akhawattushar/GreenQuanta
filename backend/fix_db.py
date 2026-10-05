import os
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()
client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('MONGODB_DB')]
res = db.voyages.update_many({'vessel_type': 'Oil Tanker'}, {'$set': {'vessel_type': 'Tanker Ship'}})
print(f'Updated {res.modified_count} records')

