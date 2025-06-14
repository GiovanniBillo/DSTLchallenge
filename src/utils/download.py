import os
from huggingface_hub import hf_hub_download
import zipfile

def download_from_huggingface(repo_id="piolla/dstl_challenge_dataset", filename="dstl_dataset.zip", output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {filename} from {repo_id} ...")
    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=output_dir,
        local_dir_use_symlinks=False  # ensure copy rather than symlink
    )

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    print(f"✅ Done. Dataset extracted to: {output_dir}")

if __name__ == "__main__":
    download_from_huggingface()

