import gdown
import zipfile
import os

def download_and_extract_models():
    # Google Drive file ID (replace this with your actual ID)
    file_id = "15hl5tqOLrm86R8aYmSmw5YJ7Jl9A7k_T"
    url = f"https://drive.google.com/uc?id={file_id}"

    # Create models directory if not exists
    if not os.path.exists('models'):
        os.makedirs('models')

    # Download models.zip
    output = 'models.zip'
    print("Downloading models.zip...")
    gdown.download(url, output, quiet=False)

    # Extract models.zip
    print("Extracting models.zip...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('models')

    # Remove the zip file after extraction
    os.remove(output)
    print("✅ Models downloaded and extracted to /models/")

if __name__ == "__main__":
    download_and_extract_models()
