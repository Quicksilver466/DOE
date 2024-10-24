import sys
from tqdm import tqdm

sys.path.append("/home/ketkar/Public/Python_programs/DOE")

from src.utils.db_utils import PgDb
import os
from datasets import Dataset

db = PgDb(connection_args={"url": os.getenv("DATABASE_URL")})
query = "SELECT id, text_chunk FROM public.embeddings ORDER BY id"
fetcher = db.fetch_records_batch(query=query, batch_size=10000)

hf_dict = {
    "id": [],
    "text": []
}

for fetch in tqdm(fetcher):
    hf_dict["id"].append(fetch["id"])
    hf_dict["text"].append(fetch["text"])

hf_dataset = Dataset.from_dict(hf_dict)
hf_dataset.save_to_disk("/data/Datasets-LLMS/CPA_DB_TEXT")