import tarfile
import gzip

# Path to your CIFAR-10 .tar.gz file
#cifar10_path = 'data/cifar-10/cifar-10-python.tar.gz'

#wine_path = 'data/wine/data.csv.gz'

# Open the .tar.gz file
#with tarfile.open(cifar10_path, "r:gz") as tar:
#    # Extract all contents into a directory
#    tar.extractall(path="path/to/extract/")
#    print("Extraction completed!")




wine_path = 'data/wine/data.csv.gz'
#wine_path = 'data/wine/test_mask.csv.gz'

# Open the .gz file
with gzip.open(wine_path, 'rt') as gz:
    # Read the contents of the file
    file_content = gz.read()
    print("Extraction completed!")

# Optional: If you want to save the extracted content to a new file
output_path = 'data/wine/data.csv'
with open(output_path, 'w') as output_file:
    output_file.write(file_content)

print(f"File extracted and saved to {output_path}!")