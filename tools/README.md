# Tools

These are tools that can be used for various tasks related to machine learning, mainly for
cleaning dataset and extracting knowledge fastly.

Usage:
```bash
chmod +x <program>
./<program> <args>
```

## generate_blob.py
This application generates the blob (the average image) of a collection of images.
Two arguments must be given:
* folder_path: the path to the folder containing the dataset
* output_path: the path to the folder in which the blob will be saved

The file will always be named "blob.png".