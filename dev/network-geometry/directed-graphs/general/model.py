import os
import sys
import logging

"""
Generic Model Class.
"""

class Model:
    def __init__(self, log_file_path: str = "logs/default.log", verbose: bool = False, **kwargs):
        """
        Initialize the Model.
        """
        self.logger = self._setup_logging(log_file_path)
        self.verbose = verbose
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def _setup_logging(self, log_file_path: str) -> logging.Logger:
        """
        Setup logging with the given log file path.
        
        Parameters
        ----------
        log_file_path : str
            Path to the log file.
        
        Returns
        -------
        logging.Logger
            Logger
        """
        for i in range(1, len(log_file_path.split("/"))):
            if not os.path.exists("/".join(log_file_path.split("/")[:i])):
                os.mkdir("/".join(log_file_path.split("/")[:i]))
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        if log_file_path:
            handler = logging.FileHandler(log_file_path, mode='w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler) 
        return logger
        
    def set_params(self, **kwargs):
        """
        Set the Ensemble Analyser parameters.
        
        Parameters
        ----------
        **kwargs : dict
            Parameters.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def modify_log_file_path(self, log_file_path: str) -> None:
        """
        Modify the log file path for the logger.

        Args:
            log_file_path (str): The new log file path.
        """
        #create directory if doesn't exist
        for i in range(1, len(log_file_path.split("/"))):
            if not os.path.exists("/".join(log_file_path.split("/")[:i])):
                os.mkdir("/".join(log_file_path.split("/")[:i]))
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
        handler = logging.StreamHandler(sys.stdout)
        if log_file_path:
            handler = logging.FileHandler(log_file_path, mode='w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)