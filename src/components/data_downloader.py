import os
import opendatasets as od
from src.logger import logger

# Dataset URL
DATASET_URL = "https://www.kaggle.com/datasets/spscientist/students-performance-in-exams"

# Download directory
DOWNLOAD_DIR = "data/"

def download_dataset(url=DATASET_URL, download_dir=DOWNLOAD_DIR):
    """
    Download dataset from Kaggle using opendatasets
    """
    try:
        os.makedirs(download_dir, exist_ok=True)
        logger.info(f"📥 Starting download from {url} ...")
        od.download(url, data_dir=download_dir)
        logger.info(f"✅ Dataset downloaded to {download_dir}")
    except Exception as e:
        logger.error(f"❌ Failed to download dataset: {e}")
        raise e

if __name__ == "__main__":
    download_dataset()
