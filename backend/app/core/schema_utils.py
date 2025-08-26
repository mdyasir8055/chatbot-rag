from sqlalchemy import inspect
from app.core.db import engine

def get_schema():
    inspector = inspect(engine)
    schema_info = {}
    for table in inspector.get_table_names():
        schema_info[table] = [col["name"] for col in inspector.get_columns(table)]
    return schema_info
