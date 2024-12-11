import os
import shutil

def cleanup_directory(directory, required_free_space_gb=14):
    required_free_space_bytes = required_free_space_gb * (1024**3)

    def get_free_space(directory):
        statvfs = os.statvfs(directory)
        return statvfs.f_frsize * statvfs.f_bavail

    def get_files_and_dirs(directory):
        items = []
        for root, dirs, files in os.walk(directory):
            for name in files:
                file_path = os.path.join(root, name)
                items.append((file_path, os.path.getatime(file_path)))
            for name in dirs:
                dir_path = os.path.join(root, name)
                items.append((dir_path, os.path.getatime(dir_path)))
        return items

    def delete_item(path):
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
            return True
        except Exception as e:
            print(f"Failed to delete {path}: {e}")
            return False

    free_space = get_free_space(directory)
    if free_space >= required_free_space_bytes:
        print(f"Sufficient free space available: {free_space / (1024**3):.2f} GB in {directory}")
        return

    items = get_files_and_dirs(directory)
    items.sort(key=lambda x: x[1])

    for path, atime in items:
        if delete_item(path):
            free_space = get_free_space(directory)
            if free_space >= required_free_space_bytes:
                print(f"Freed up enough space. Available free space: {free_space / (1024**3):.2f} GB in {directory}")
                return

    print("Finished cleanup, but still insufficient free space.")

if __name__ == "__main__":
    import sys
    cleanup_directory(sys.argv[1])