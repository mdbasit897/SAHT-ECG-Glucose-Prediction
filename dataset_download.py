import requests
import zipfile
import os
import mimetypes

# Cuff-Less Blood Pressure Estimation download URL
url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/9c47vwvtss-4.zip"
zip_path = 'Dataset on electrocardiograph, sleep and metabolic function of male type 2 diabetes mellitus.zip'
extract_dir = 'Dataset_on_electrocardiograph'

# Set headers to mimic browser request and ensure file download
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/zip'
}

# Download with streaming for large files
print("Redownloading Dataset on electrocardiograph, sleep and metabolic function of male type 2 diabetes mellitus.zip from Mendeley Data")
response = requests.get(url, headers=headers, stream=True)
response.raise_for_status()  # Check for HTTP errors

# Save the file
with open(zip_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
print(f"Downloaded {zip_path}, size: {os.path.getsize(zip_path)} bytes")

# Verify file type
mime_type, _ = mimetypes.guess_type(zip_path)
print(f"Detected MIME type: {mime_type}")

if os.path.getsize(zip_path) > 10_000_000:  # Expect >10 MB
    if mime_type == 'application/zip':
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("Extraction successful. Contents:", os.listdir(extract_dir))
        except zipfile.BadZipFile:
            print("File is corrupted despite correct size. Inspect manually.")
    else:
        with open(zip_path, 'rb') as f:
            print("First 10 bytes:", f.read(10))
        print("Not a ZIP file. Likely an HTML page or error response.")
else:
    print("File size too small (<10 MB). Redownload failed. Try manual download from Zenodo.")

# Manual download fallback (instructions)
print("\nIf automated download fails, manually download from:")
print("1. Visit: https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/9c47vwvtss-4.zip")
print("2. Click 'Download' on Dataset on electrocardiograph, sleep and metabolic function of male type 2 diabetes mellitus.zip")
print("3. Place the file in the notebook directory as 'Dataset on electrocardiograph, sleep and metabolic function of male type 2 diabetes mellitus.zip' and rerun extraction.")