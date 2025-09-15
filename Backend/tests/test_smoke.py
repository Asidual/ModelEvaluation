import requests
url = "http://localhost:8000/v1/datasets"
with open(r"data\raw\crocodile_dataset.csv", "rb") as f:
    files = {"file": ("crocodiles.csv", f, "text/csv")}
    data = {
        "delimiter": ",",
        "decimal": ".",
        "dayfirst": "true",
        "threshold_cat": "20",
        "max_cat_values": "10",
    }
    r = requests.post(url, files=files, data=data, timeout=60)
print(r.status_code)
print(r.json())
