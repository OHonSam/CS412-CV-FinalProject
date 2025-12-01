import os
import requests
import dotenv

# ========= CONFIG =========
config = dotenv.dotenv_values(".env")
API_KEY = config.get("DRIVE_API_KEY", "YOUR_DRIVE_API_KEY_HERE")
FOLDER_ID = "177LF1vytnslJ5NFd1kkmsUuvxVEEYhqj"
DEST_DIR = "./SUTD/questions/"   # downloaded files go here
EXTENSION = ".jsonl"
# ==========================


def list_all_files(folder_id, api_key):
    """Return a list of metadata for every file in a public Drive folder."""
    url = "https://www.googleapis.com/drive/v3/files"
    params = {
        "q": f"'{folder_id}' in parents and trashed=false",
        "fields": "nextPageToken, files(id, name, mimeType)",
        "pageSize": 1000,
        "key": api_key
    }

    all_files = []

    while True:
        response = requests.get(url, params=params)
        data = response.json()

        files = data.get("files", [])
        all_files.extend(files)

        if "nextPageToken" not in data:
            break

        params["pageToken"] = data["nextPageToken"]

    return all_files


def download_file(file_id, filename, api_key, dest_dir):
    """Download a Google Drive file using only an API key."""
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"[OK] {filename}")


if __name__ == "__main__":
    print("Listing files in folder...")
    files = list_all_files(FOLDER_ID, API_KEY)

    print(f"Total files found: {len(files)}")
    specific_type_files = [f for f in files if f["name"].lower().endswith(EXTENSION)]
    print(f"Total files: {len(specific_type_files)}\n")

    for f in specific_type_files:
        print("Downloading:", f["name"])
        try:
            download_file(f["id"], f["name"], API_KEY, DEST_DIR)
        except Exception as e:
            print(f"[ERROR] Failed to download {f['name']}: {e}")
