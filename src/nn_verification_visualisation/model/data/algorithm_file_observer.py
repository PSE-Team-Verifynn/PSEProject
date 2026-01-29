import os
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Assuming this is your storage import
from nn_verification_visualisation.model.data.storage import Storage
from nn_verification_visualisation.model.data_loader.algorithm_loader import AlgorithmLoader


class AlgorithmFileObserver(FileSystemEventHandler):
    """
    Handles the algorithm file changes. This allows the program to dynamically react to the user editing the 'algorithms' directory.
    """

    ALLOWED_EXTENSIONS = (".py",)

    def __init__(self):
        # figuring out the algorithms directory
        current_dir = Path(__file__).parent.resolve()
        self.watch_dir = (current_dir.parents[4] / "algorithms")

        if not self.watch_dir.exists():
            raise FileNotFoundError(f"Could not find algorithm directory at: {self.watch_dir}")

        self.observer = Observer()
        self.observer.schedule(self, str(self.watch_dir), recursive=True)
        self.observer.start()
        print(f"Observer started on: {self.watch_dir}")

    def __process_event(self, event, action_type):
        """
        Helper method to handle the logic for all event types.
        """
        if event.is_directory:
            return

        if not event.src_path.endswith(self.ALLOWED_EXTENSIONS):
            return

        file_path = Path(event.src_path)
        algo_name = file_path.stem

        print(f"Detected {action_type} on algorithm: {algo_name}")

        # Creating the storage if it doesn't exist yet
        storage = Storage()

        # Example logic:
        if action_type == "deleted":
            storage.remove_algorithm(algo_name)
            return

        new_algorithm = AlgorithmLoader.load_algorithm(event.src_path)
        if action_type == "modified":
            storage.modify_algorithm(algo_name, new_algorithm)
        else:
            storage.add_algorithm(new_algorithm)


    def on_modified(self, event):
        self.__process_event(event, "modified")

    def on_created(self, event):
        self.__process_event(event, "created")

    def on_deleted(self, event):
        self.__process_event(event, "deleted")

    def stop(self):
        self.observer.stop()
        self.observer.join()