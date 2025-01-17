from google.cloud.storage import Client
from pathlib import Path
from google.cloud.storage import Client, transfer_manager
import glob
import os
from tqdm import tqdm

def upload_local_directory_to_gcs(storage_client, bucket, local_path, gcs_path):
    assert os.path.isdir(local_path)
    for local_file in tqdm(glob.glob(local_path + '/**')):
        if not os.path.isfile(local_file):
            print(f'currently on directory: {local_file}')
            upload_local_directory_to_gcs(storage_client, bucket, local_file, gcs_path + "/" + os.path.basename(local_file))
        else:
            remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):]).replace('\\','/')
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)


if __name__ == "__main__":

    storage_client = Client.from_service_account_json(
        r"C:\Users\HP-VICTUS\Documents\Masters\WQF7023 AI Project\pure-night-447115-e5-36ba8058f52c.json"
    )
    dataset_path = r"C:\Users\HP-VICTUS\Documents\Masters\WQF7023 AI Project\dataset"
    bucket_name = "isic2019_derm_dataset"
    bucket = storage_client.bucket(bucket_name)

    upload_local_directory_to_gcs(storage_client, bucket, dataset_path, "dataset")