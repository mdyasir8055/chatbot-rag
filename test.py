import os
import requests

def test_gse():
    api_key = os.getenv("GOOGLE_API_KEY")
    cx_id = os.getenv("GOOGLE_CX")

    if not api_key or not cx_id:
        print("❌ Missing GOOGLE_API_KEY or GOOGLE_CX in environment.")
        return

    try:
        query = "KZ EDX Pro price site:headphonezone.in"
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": api_key, "cx": cx_id, "q": query, "num": 3}

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        items = data.get("items", [])
        if not items:
            print("⚠️ No results found. Check if your CX is restricted.")
        else:
            print("✅ GSE API key works! Here are some results:")
            for i, item in enumerate(items, 1):
                print(f"{i}. {item.get('title')} — {item.get('link')}")

    except Exception as e:
        print("❌ GSE test failed:", str(e))


if __name__ == "__main__":
    test_gse()
