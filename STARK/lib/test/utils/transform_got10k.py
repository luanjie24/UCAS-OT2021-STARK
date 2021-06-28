import numpy as np
import os
import zipfile
import shutil
from lib.test.evaluation.environment import env_settings


def transform_got10k(tracker_name, cfg_name):

    env = env_settings()
    result_dir = env.results_path
     

    zip_dir = os.path.join(result_dir, "%s/%s/got10k.zip" % (tracker_name, cfg_name))
    src_dir = os.path.join(result_dir, "%s/%s/got10k" % (tracker_name, cfg_name))
    if not os.path.exists(src_dir):
        os.mkdir(src_dir)
    z = zipfile.ZipFile(zip_dir, 'r')
    z.extractall(path=src_dir)
    z.close()

    dest_dir = os.path.join(result_dir, "%s/%s/got10k_submit/" % (tracker_name, cfg_name))
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    items = os.listdir(src_dir)
    for item in items:
        if "all" in item:
            continue
        src_path = os.path.join(src_dir, item)
        if "time" not in item:
            seq_name = item.replace(".txt", '')
            seq_dir = os.path.join(dest_dir, seq_name)
            if not os.path.exists(seq_dir):
                os.makedirs(seq_dir)
            new_item = item.replace(".txt", '_001.txt')
            dest_path = os.path.join(seq_dir, new_item)
            bbox_arr = np.loadtxt(src_path, dtype=np.int, delimiter='\t')
            np.savetxt(dest_path, bbox_arr, fmt='%d', delimiter=',')
        else:
            seq_name = item.replace("_time.txt", '')
            seq_dir = os.path.join(dest_dir, seq_name)
            if not os.path.exists(seq_dir):
                os.makedirs(seq_dir)
            dest_path = os.path.join(seq_dir, item)
            #os.system("copy %s %s" % (src_path, dest_path))
            shutil.copyfile(src_path,dest_path)

    # make zip archive
    shutil.make_archive(src_dir, "zip", src_dir)
    shutil.make_archive(dest_dir, "zip", dest_dir)
    # Remove the original files
    shutil.rmtree(src_dir)
    shutil.rmtree(dest_dir)


if __name__ == "__main__":

    transform_got10k("stark_s", "baseline_got10k_only")


