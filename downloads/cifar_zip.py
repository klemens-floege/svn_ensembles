import tarfile

# Path to your CIFAR-10 .tar.gz file
cifar10_path = 'data/cifar-10/cifar-10-python.tar.gz'

# Open the .tar.gz file
with tarfile.open(cifar10_path, "r:gz") as tar:
    # Extract all contents into a directory
    tar.extractall(path="path/to/extract/")
    print("Extraction completed!")