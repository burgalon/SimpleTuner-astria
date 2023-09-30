from helpers.training.state_tracker import StateTracker
from helpers.multiaspect.image import MultiaspectImage
from helpers.data_backend.base import BaseDataBackend
from pathlib import Path
import json, logging, os
from multiprocessing import Manager
from tqdm import tqdm
from multiprocessing import Process, Queue
import numpy as np
from math import floor

logger = logging.getLogger("BucketManager")
target_level = os.environ.get("SIMPLETUNER_LOG_LEVEL", "WARNING")
logger.setLevel(target_level)


class BucketManager:
    def __init__(
        self,
        instance_data_root: str,
        cache_file: str,
        metadata_file: str,
        data_backend: BaseDataBackend,
        accelerator,
        batch_size: int,
        resolution: float,
        resolution_type: str,
        apply_dataset_padding: bool = False,
    ):
        self.accelerator = accelerator
        self.data_backend = data_backend
        self.apply_dataset_padding = apply_dataset_padding
        self.batch_size = batch_size
        self.instance_data_root = Path(instance_data_root)
        self.cache_file = Path(cache_file)
        self.metadata_file = Path(metadata_file)
        self.aspect_ratio_bucket_indices = {}
        self.image_metadata = {}  # Store image metadata
        self.instance_images_path = set()
        # Initialize a multiprocessing.Manager dict for seen_images
        manager = Manager()
        self.seen_images = manager.dict()
        self.reload_cache()
        self.resolution = resolution
        self.resolution_type = resolution_type

    def __len__(self):
        """
        Returns:
            int: The number of batches in the dataset, rounded down to account for likely-discarded images.
        """
        return floor(
            sum(
                [
                    (len(bucket) // self.batch_size) * self.batch_size
                    for bucket in self.aspect_ratio_bucket_indices.values()
                    if len(bucket) >= self.batch_size
                ]
            )
            / self.batch_size
        )

    def _discover_new_files(self, for_metadata: bool = False):
        """
        Discover new files that have not been processed yet.

        Returns:
            list: A list of new files.
        """
        all_image_files = (
            StateTracker.get_image_files()
            or StateTracker.set_image_files(
                self.data_backend.list_files(
                    instance_data_root=self.instance_data_root,
                    str_pattern="*.[jJpP][pPnN][gG]",
                )
            )
        )
        # Extract only the files from the data
        if for_metadata:
            return [
                file
                for file in all_image_files
                if self.get_metadata_by_filepath(file) is None
            ]
        return [
            file
            for file in all_image_files
            if str(file) not in self.instance_images_path
        ]

    def reload_cache(self):
        """
        Load cache data from file.

        Returns:
            dict: The cache data.
        """
        # Query our DataBackend to see whether the cache file exists.
        if self.data_backend.exists(self.cache_file):
            try:
                # Use our DataBackend to actually read the cache file.
                cache_data_raw = self.data_backend.read(self.cache_file)
                cache_data = json.loads(cache_data_raw)
            except Exception as e:
                logger.warning(
                    f"Error loading aspect bucket cache, creating new one: {e}"
                )
                cache_data = {}
            self.aspect_ratio_bucket_indices = cache_data.get(
                "aspect_ratio_bucket_indices", {}
            )
            self.instance_images_path = set(cache_data.get("instance_images_path", []))

    def _save_cache(self):
        """
        Save cache data to file.
        """
        # Prune any buckets that have fewer samples than batch_size
        self._enforce_min_bucket_size()
        # Convert any non-strings into strings as we save the index.
        aspect_ratio_bucket_indices_str = {
            key: [str(path) for path in value]
            for key, value in self.aspect_ratio_bucket_indices.items()
        }
        # Encode the cache as JSON.
        cache_data = {
            "aspect_ratio_bucket_indices": aspect_ratio_bucket_indices_str,
            "instance_images_path": [str(path) for path in self.instance_images_path],
        }
        cache_data_str = json.dumps(cache_data)
        # Use our DataBackend to write the cache file.
        self.data_backend.write(self.cache_file, cache_data_str)

    def _bucket_worker(
        self,
        tqdm_queue,
        files,
        aspect_ratio_bucket_indices_queue,
        metadata_updates_queue,
        existing_files_set,
        data_backend,
    ):
        """
        A worker function to bucket a list of files.

        Args:
            tqdm_queue (Queue): A queue to report progress to.
            files (list): A list of files to bucket.
            aspect_ratio_bucket_indices_queue (Queue): A queue to report the bucket indices to.
            existing_files_set (set): A set of existing files.

        Returns:
            dict: The bucket indices.
        """
        local_aspect_ratio_bucket_indices = {}
        local_metadata_updates = {}
        for file in files:
            if str(file) not in existing_files_set:
                local_aspect_ratio_bucket_indices = MultiaspectImage.process_for_bucket(
                    data_backend,
                    self,
                    file,
                    local_aspect_ratio_bucket_indices,
                    metadata_updates=local_metadata_updates,
                )
            tqdm_queue.put(1)
        if aspect_ratio_bucket_indices_queue is not None:
            aspect_ratio_bucket_indices_queue.put(local_aspect_ratio_bucket_indices)
        metadata_updates_queue.put(local_metadata_updates)

    def compute_aspect_ratio_bucket_indices(self):
        """
        Compute the aspect ratio bucket indices. The workhorse of this class.

        Returns:
            dict: The aspect ratio bucket indices.
        """
        logger.info("Discovering new files...")
        new_files = self._discover_new_files()

        if not new_files:
            logger.info("No new files discovered. Exiting.")
            return

        existing_files_set = set().union(*self.aspect_ratio_bucket_indices.values())

        num_cpus = 8  # Using a fixed number for better control and predictability
        files_split = np.array_split(new_files, num_cpus)

        metadata_updates_queue = Queue()
        tqdm_queue = Queue()
        aspect_ratio_bucket_indices_queue = Queue()
        self.load_image_metadata()

        workers = [
            Process(
                target=self._bucket_worker,
                args=(
                    tqdm_queue,
                    file_shard,
                    aspect_ratio_bucket_indices_queue,
                    metadata_updates_queue,
                    existing_files_set,
                    self.data_backend,
                ),
            )
            for file_shard in files_split
        ]

        for worker in workers:
            worker.start()

        with tqdm(total=len(new_files)) as pbar:
            while any(worker.is_alive() for worker in workers):
                while not tqdm_queue.empty():
                    pbar.update(tqdm_queue.get())
                while not aspect_ratio_bucket_indices_queue.empty():
                    aspect_ratio_bucket_indices_update = (
                        aspect_ratio_bucket_indices_queue.get()
                    )
                    for key, value in aspect_ratio_bucket_indices_update.items():
                        self.aspect_ratio_bucket_indices.setdefault(key, []).extend(
                            value
                        )
                # Now, pull metadata updates from the queue
                while not metadata_updates_queue.empty():
                    metadata_update = metadata_updates_queue.get()
                    for filepath, meta in metadata_update.items():
                        self.set_metadata_by_filepath(
                            filepath=filepath, metadata=meta, update_json=False
                        )

        for worker in workers:
            worker.join()

        self.instance_images_path.update(new_files)
        self._save_cache()
        self.save_image_metadata()
        logger.info("Completed aspect bucket update.")

    def split_buckets_between_processes(self):
        """
        Splits the contents of each bucket in aspect_ratio_bucket_indices between the available processes.
        """
        new_aspect_ratio_bucket_indices = {}
        for bucket, images in self.aspect_ratio_bucket_indices.items():
            with self.accelerator.split_between_processes(
                images, apply_padding=self.apply_dataset_padding
            ) as images_split:
                # Now images_split contains only the part of the images list that this process should handle
                new_aspect_ratio_bucket_indices[bucket] = images_split

        # Replace the original aspect_ratio_bucket_indices with the new one containing only this process's share
        self.aspect_ratio_bucket_indices = new_aspect_ratio_bucket_indices

    def mark_as_seen(self, image_path):
        """Mark an image as seen."""
        self.seen_images[image_path] = True  # This will be shared across all processes

    def is_seen(self, image_path):
        """Check if an image is seen."""
        return self.seen_images.get(image_path, False)

    def reset_seen_images(self):
        """Reset the seen images."""
        self.seen_images.clear()

    def remove_image(self, image_path, bucket):
        """
        Used by other classes to reliably remove images from a bucket.

        Args:
            image_path (str): The path to the image to remove.
            bucket (str): The bucket to remove the image from.

        Returns:
            dict: The aspect ratio bucket indices.
        """
        if image_path in self.aspect_ratio_bucket_indices[bucket]:
            self.aspect_ratio_bucket_indices[bucket].remove(image_path)

    def update_buckets_with_existing_files(self, existing_files: set):
        """
        Update bucket indices to remove entries that no longer exist.

        Args:
            existing_files (set): A set of existing files.
        """
        for bucket, images in self.aspect_ratio_bucket_indices.items():
            self.aspect_ratio_bucket_indices[bucket] = [
                img for img in images if img in existing_files
            ]
        # Save the updated cache
        self._save_cache()

    def refresh_buckets(self, rank: int = None):
        """
        Discover new files and remove images that no longer exist.
        """
        logger.debug(f"{rank} Computing new file aspect bucket indices")
        # Discover new files and update bucket indices
        self.compute_aspect_ratio_bucket_indices()

        # Get the list of existing files
        existing_files = StateTracker.get_image_files()
        logger.debug(
            f"{rank} Discovering existing files for refresh_buckets, so that we can remove files from the aspect bucket cache if they no longer exist"
        )

        # Update bucket indices to remove entries that no longer exist
        logger.debug(f"{rank} Finally, we can update the bucket index")
        self.update_buckets_with_existing_files(existing_files)
        logger.debug(f"{rank} Done updating bucket index, continuing.")
        return

    def _enforce_min_bucket_size(self):
        """
        Remove buckets that have fewer samples than batch_size.
        """
        for bucket, images in list(
            self.aspect_ratio_bucket_indices.items()
        ):  # Make a list of items to iterate
            if len(images) < self.batch_size:
                del self.aspect_ratio_bucket_indices[bucket]
                logger.warning(f"Removed bucket {bucket} due to insufficient samples.")

    def handle_incorrect_bucket(self, image_path: str, bucket: str, actual_bucket: str):
        """
        Used by other classes to move images between buckets, when mis-detected.

        Args:
            image_path (str): The path to the image to move.
            bucket (str): The bucket to move the image from.
            actual_bucket (str): The bucket to move the image to.
        """
        logger.warning(
            f"Found an image in bucket {bucket} it doesn't belong in, when actually it is: {actual_bucket}"
        )
        self.remove_image(image_path, bucket)
        if actual_bucket in self.aspect_ratio_bucket_indices:
            logger.warning(f"Moved image to bucket, it already existed.")
            self.aspect_ratio_bucket_indices[actual_bucket].append(image_path)
        else:
            logger.warning(f"Created new bucket for that pesky image.")
            self.aspect_ratio_bucket_indices[actual_bucket] = [image_path]
        self._save_cache()

    def handle_small_image(
        self, image_path: str, bucket: str, delete_unwanted_images: bool
    ):
        """
        Used by other classes to remove an image, or DELETE it from disk, depending on parameters.

        Args:
            image_path (str): The path to the image to remove.
            bucket (str): The bucket to remove the image from.
            delete_unwanted_images (bool): Whether to delete the image from disk.
        """
        if delete_unwanted_images:
            try:
                logger.warning(
                    f"Image {image_path} too small: DELETING image and continuing search."
                )
                self.data_backend.remove(image_path)
            except Exception as e:
                logger.debug(
                    f"Image {image_path} was already deleted. Another GPU must have gotten to it."
                )
        else:
            logger.warning(
                f"Image {image_path} too small, but --delete_unwanted_images is not provided, so we simply ignore and remove from bucket."
            )
        self.remove_image(image_path, bucket)

    def read_cache(self):
        """
        Read the entire bucket cache.
        """
        return self.aspect_ratio_bucket_indices

    def get_metadata_attribute_by_filepath(self, filepath: str, attribute: str):
        """Use get_metadata_by_filepath to return a specific attribute.

        Args:
            filepath (str): The complete path from the aspect bucket list.
            attribute (str): The attribute you are seeking.

        Returns:
            any type: The attribute value, or None.
        """
        metadata = self.get_metadata_by_filepath(filepath)
        if metadata:
            return metadata.get(attribute, None)
        else:
            return None

    def set_metadata_attribute_by_filepath(
        self, filepath: str, attribute: str, value: any, update_json: bool = True
    ):
        """Use set_metadata_by_filepath to update the contents of a specific attribute.

        Args:
            filepath (str): The complete path from the aspect bucket list.
            attribute (str): The attribute you are updating.
            value (any type): The value to set.
        """
        metadata = self.get_metadata_by_filepath(filepath) or {}
        metadata[attribute] = value
        return self.set_metadata_by_filepath(filepath, metadata, update_json)

    def set_metadata_by_filepath(
        self, filepath: str, metadata: dict, update_json: bool = True
    ):
        """Set metadata for a given image file path.

        Args:
            filepath (str): The complete path from the aspect bucket list.
        """
        logger.debug(f"Setting metadata for {filepath} to {metadata}.")
        self.image_metadata[filepath] = metadata
        if update_json:
            self.save_image_metadata()

    def get_metadata_by_filepath(self, filepath: str):
        """Retrieve metadata for a given image file path.

        Args:
            filepath (str): The complete path from the aspect bucket list.

        Returns:
            dict: Metadata for the image. Returns None if not found.
        """
        return self.image_metadata.get(filepath, None)

    def load_image_metadata(self):
        """Load image metadata from a JSON file."""
        self.image_metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                data = json.load(f)
                self.image_metadata = data.get("image_metadata", {})

    def save_image_metadata(self):
        """Save image metadata to a JSON file."""
        self.data_backend.write(self.metadata_file, json.dumps(self.image_metadata))

    def scan_for_metadata(self):
        """
        Update the metadata without modifying the bucket indices.
        """
        logger.info("Discovering new images for metadata scan...")
        new_files = self._discover_new_files(for_metadata=True)
        if not new_files:
            logger.info("No new files discovered. Exiting.")
            return

        existing_files_set = self.image_metadata.keys()

        num_cpus = 8  # Using a fixed number for better control and predictability
        files_split = np.array_split(new_files, num_cpus)

        metadata_updates_queue = Queue()
        tqdm_queue = Queue()
        self.load_image_metadata()

        workers = [
            Process(
                target=self._bucket_worker,
                args=(
                    tqdm_queue,
                    file_shard,
                    None,  # Passing None to indicate we don't want to update the buckets
                    metadata_updates_queue,
                    existing_files_set,
                    self.data_backend,
                ),
            )
            for file_shard in files_split
        ]

        for worker in workers:
            worker.start()

        with tqdm(total=len(new_files)) as pbar:
            while any(worker.is_alive() for worker in workers):
                while not tqdm_queue.empty():
                    pbar.update(tqdm_queue.get())

                # Only update the metadata
                while not metadata_updates_queue.empty():
                    metadata_update = metadata_updates_queue.get()
                    for filepath, meta in metadata_update.items():
                        self.set_metadata_by_filepath(
                            filepath=filepath, metadata=meta, update_json=False
                        )

        for worker in workers:
            worker.join()

        self._save_cache()
        self.save_image_metadata()
        logger.info("Completed metadata update.")
