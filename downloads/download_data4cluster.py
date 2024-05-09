import requests
import os
from tqdm import tqdm
import tarfile

def download_file(url, save_path, chunk_size=1024):
    """ Download file with progress bar using requests """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(save_path, 'wb') as file:
        for data in response.iter_content(chunk_size=chunk_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Something went wrong with the download")
        return False
    return True

def extract_tar_file(tar_path, extract_to='.'):
    """ Extract tar.gz file """
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)

# URL of the CIFAR-10 dataset
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

# Define the path to save the downloaded file
save_path = 'data/cifar-10/cifar-10-python.tar.gz'

# Download the file
print("Downloading CIFAR-10 dataset...")
success = download_file(url, save_path)

if success:
    print("Download complete. Extracting files...")
    extract_tar_file(save_path)
    print("Extraction complete.")
    # Optionally, remove the tar file after extraction
    # os.remove(save_path)
else:
    print("Failed to download the file.")