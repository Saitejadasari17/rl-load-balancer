import gdown
import os

os.makedirs("data", exist_ok=True)

file_id = "14oLQcGKLTrtsWcS27FmmhtFcM08uO25x"

url = f"https://drive.google.com/uc?id={file_id}"

output = "data/azurefunctions-dataset2019.tar.xz"

if not os.path.exists(output):
    print("Downloading dataset from Google Drive...")
    gdown.download(url, output, quiet=False)
    print("Dataset downloaded successfully!")
else:
    print("Dataset already exists.")