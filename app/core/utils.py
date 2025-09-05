def ok(data):
    return {"status":"ok","data":data}

def sanitize_text(txt: str) -> str:
    return " ".join((txt or "").split())
