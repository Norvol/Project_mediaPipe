import os
from ftplib import FTP, error_perm
import logging
from functools import wraps
import time
import configparser
import socket

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Read configuration from a file
config = configparser.ConfigParser()
config.read('config.ini')

FTP_SERVER = config.get('FTP', 'SERVER')
FTP_PORT = config.getint('FTP', 'PORT')
FTP_USERNAME = config.get('FTP', 'USERNAME')
FTP_PASSWORD = config.get('FTP', 'PASSWORD')
FTP_ALARM_PATH = config.get('FTP', 'ALARM_PATH')

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
CONNECTION_TIMEOUT = 30  # seconds

def retry(max_attempts, delay=RETRY_DELAY):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except (socket.error, IOError, error_perm) as e:
                    attempts += 1
                    logging.warning(f"Attempt {attempts} failed: {str(e)}")
                    if attempts == max_attempts:
                        logging.error(f"Max attempts reached. Function {func.__name__} failed.")
                        raise
                    time.sleep(delay)
        return wrapper
    return decorator

class FTPHandler:
    def __init__(self):
        self.ftp = None

    @retry(MAX_RETRIES)
    def connect(self):
        self.ftp = FTP()
        self.ftp.connect(FTP_SERVER, FTP_PORT, timeout=CONNECTION_TIMEOUT)
        self.ftp.login(FTP_USERNAME, FTP_PASSWORD)
        logging.info("Successfully connected to FTP server")

    def disconnect(self):
        if self.ftp:
            try:
                self.ftp.quit()
            except:
                self.ftp.close()
            finally:
                self.ftp = None
                logging.info("Disconnected from FTP server")

    @retry(MAX_RETRIES)
    def send_file(self, local_path, remote_path):
        if not self.ftp:
            self.connect()
        
        try:
            with open(local_path, 'rb') as file:
                self.ftp.storbinary(f'STOR {remote_path}', file)
            logging.info(f"File sent successfully: {remote_path}")
        except error_perm as e:
            if str(e).startswith('550'):
                logging.error(f"Permission denied or directory doesn't exist: {remote_path}")
                self.create_remote_dirs(os.path.dirname(remote_path))
                self.send_file(local_path, remote_path)  # Retry after creating directories
            else:
                raise

    def create_remote_dirs(self, remote_dir):
        """Create directories recursively on the FTP server."""
        directories = remote_dir.split('/')
        path = ''
        for directory in directories:
            if not directory:
                continue
            path = os.path.join(path, directory)
            try:
                self.ftp.cwd(path)
            except error_perm:
                try:
                    self.ftp.mkd(path)
                    self.ftp.cwd(path)
                    logging.info(f"Created remote directory: {path}")
                except error_perm as e:
                    logging.error(f"Failed to create directory '{path}': {str(e)}")
                    raise

def send_alarm_to_ftp(timestamp):
    alarm_file_name = f'boxing_alarm_{timestamp}.txt'
    local_path = os.path.join(os.getcwd(), alarm_file_name)
    remote_path = os.path.join(FTP_ALARM_PATH, alarm_file_name).replace('\\', '/')
    alarm_content = f"Boxing detected at {timestamp}"

    try:
        # Write alarm content to a local file
        with open(local_path, 'w') as f:
            f.write(alarm_content)

        # Send file to FTP
        ftp_handler = FTPHandler()
        try:
            ftp_handler.connect()
            ftp_handler.send_file(local_path, remote_path)
        finally:
            ftp_handler.disconnect()

        logging.info(f"Alarm sent to FTP: {remote_path}")

    except Exception as e:
        logging.error(f"Failed to send alarm to FTP: {str(e)}")
    finally:
        # Remove the local file after sending
        if os.path.exists(local_path):
            os.remove(local_path)
            logging.info(f"Removed local file: {local_path}")

# Example usage
if __name__ == "__main__":
    send_alarm_to_ftp(time.strftime("%Y%m%d-%H%M%S"))
