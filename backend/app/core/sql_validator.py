def is_safe_sql(query: str) -> bool:
    forbidden = ["drop", "delete", "truncate", "alter"]
    return not any(word in query.lower() for word in forbidden)
