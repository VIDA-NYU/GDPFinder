import paramiko
import numpy as np
import os
from tqdm import tqdm
import handle_tif_images
import data

if __name__ == "__main__":
    vidagpu_username = ...
    vidagpu_password = ...
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('vidagpu.poly.edu', username=vidagpu_username, password=vidagpu_password)
    sftp = ssh.open_sftp()
    sftp.chdir('/vida/work/GDPFinder/GDPFinder/data/output/unzipped_files')
    dir_contents = sftp.listdir()
    # keep only tif files
    dir_contents = [f for f in dir_contents if f.endswith('.tif')]
    print(f"Total of {len(dir_contents)} tif files")
    
    dir_contents.sort()
    n = 1000
    for file in tqdm(dir_contents[:n]):
        # download
        sftp.get(file, "../data/output/unzipped_files/" + file)

        df = handle_tif_images.create_files_df()
        df.to_file("../data/output/downloaded_scenes_metadata.geojson")

        data.save_samples_patch(
            output_dir = "small_patches",
            size = 112
        )

        os.remove("../data/output/unzipped_files/" + file)






    
    sftp.close()
    ssh.close()