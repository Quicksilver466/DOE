import sys

sys.path.append("/home/ketkar/Public/Python_programs/DOE")

from src.utils.db_utils import PgDb
import os
from datasets import IterableDataset, Dataset

db = PgDb(connection_args={"url": os.getenv("DATABASE_URL")})
query = "SELECT id, text_chunk FROM public.embeddings ORDER BY id"
fetcher = db.fetch_records_batch(query=query, batch_size=10000)

#for fetch in fetcher:
#    print(fetch)
#    break