import paramiko
import geopandas as gpd
import os
import shutil
from tqdm import tqdm

import handle_tif_images
import data


def download_patches():
    vidagpu_username = input()
    vidagpu_password = input()
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        "vidagpu.poly.edu", username=vidagpu_username, password=vidagpu_password
    )
    sftp = ssh.open_sftp()
    sftp.chdir("/vida/work/GDPFinder/GDPFinder/data/output/unzipped_files")
    dir_contents = sftp.listdir()
    # keep only tif files
    dir_contents = [f for f in dir_contents if f.endswith(".tif")]
    print(f"Total of {len(dir_contents)} tif files")

    dir_contents.sort()
    n = 1000
    for file in tqdm(dir_contents[:n]):
        # download
        sftp.get(file, "../data/output/unzipped_files/" + file)

        df = handle_tif_images.create_files_df()
        df.to_file("../data/output/downloaded_scenes_metadata.geojson")

        data.save_samples_patch(output_dir="old_patches", size=224)

        os.remove("../data/output/unzipped_files/" + file)

    sftp.close()
    ssh.close()


def separate_patches_by_city_state():
    city_scenes_geojson = os.listdir("../data/scenes_metadata")
    city_scenes_geojson = [s for s in city_scenes_geojson if s.endswith(".geojson")]
    city_scenes_geojson.sort()
    patches_files = os.listdir("../data/output/old_patches")

    for city_scene in tqdm(city_scenes_geojson):
        city_df = gpd.read_file("../data/scenes_metadata/" + city_scene)
        city_state = city_scene.replace("_last_scenes.geojson", "")
        if not os.path.exists("../data/output/patches/" + city_state):
            os.mkdir("../data/output/patches/" + city_state)

        for i, row in city_df.iterrows():
            scene_id = row["entity_id"]
            patches_files_scene = [
                p for p in patches_files if p.split("_")[0] == scene_id
            ]
            patches_files_scene = [
                p
                for p in patches_files_scene
                if os.path.exists("../data/output/old_patches/" + p)
            ]
            for patches_file in patches_files_scene:
                shutil.move(
                    "../data/output/old_patches/" + patches_file,
                    f"../data/output/patches/{city_state}/{patches_file}",
                )


if __name__ == "__main__":
    # download_patches()
    separate_patches_by_city_state()
