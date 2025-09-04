import shutil



THRESHOLD_GiB = 5 # stop when less than this much space remains


def get_free_space_gb(path):
    total, used, free = shutil.disk_usage(path)
    return free // (1024**3)

def is_there_still_space_for_data_collection_and_transfer(path, initial_free):
    current_free = get_free_space_gb(path)
    return ((initial_free - current_free)*2 + THRESHOLD_GiB) < initial_free


