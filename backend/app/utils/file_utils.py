import os

def save_file(content: bytes, filename: str, folder: str = "uploads"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "wb") as f:
        f.write(content)
    return path
