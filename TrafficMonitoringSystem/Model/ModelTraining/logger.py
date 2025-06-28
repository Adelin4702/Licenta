import sys
import logging
from datetime import datetime
import os


class TeeLogger:
    """
    Logger class that duplicates all stdout and stderr to a log file while
    maintaining output to the console.
    """

    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        self.file = open(log_file, 'w', encoding='utf-8')

        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file)
            ]
        )
        self.logger = logging.getLogger()

        print(f"Logging to file: {log_file}")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()

    def __del__(self):
        self.close()