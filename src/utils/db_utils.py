import psycopg2

class PgDb:
    def __init__(self, connection_args: dict) -> None:
        self.connection_args = connection_args
        self.conn = psycopg2.connect(self.connection_args.get("url"))

    def fetch_records_batch(self, query: str, batch_size: int):
        with self.conn.cursor(name='BrightAssistCursor') as cursor:
            cursor.itersize = batch_size

            cursor.execute(query=query)
            for row in cursor:
                yield {"id":  row[0], "text": row[1]}