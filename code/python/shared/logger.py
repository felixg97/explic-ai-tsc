"""
Logger
"""
import datetime
import os
import sys

from enum import Enum
from utils import find_path

TS_FORMAT = '%Y-%m-%d %H:%M:%S'

class Level(Enum):
    ERROR = 'ERROR'
    WARN = 'WARN'
    INFO = 'INFO'
    DEBUG = 'DEBUG'

class Logger():
    levels = [ 'ERROR', 'WARN', 'INFO', 'DEBUG' ]

    def __init__(self, level: str = 'ERROR',
            curr_path: str = None,
            file: str = './../../../output/logs/my_log.log', 
            persist: bool = False):  
        self.timestamp = datetime.datetime
        self.persist = persist       

        if persist: 
            self.set_file(curr_path, file)
        
        self.set_level(level)

    def set_level(self, level: str):
        if level.upper() in self.levels:    
            self.level = level
        else:
            raise Exception('Not a valid log level!')

    def set_file(self, curr_path: str, file: str):
        if curr_path is not None:
            self.file = open(find_path(curr_path, file), 'a')
        else:
            self.file = open(find_path(os.getcwd(), file), 'a')   

    def log(self, content):
        """
        Logging to file
        """
        print(content)
        if self.persist: self.file.write(f'\n{content}')

    def error(self, message):
        """ 
        One or more functionalities are not working, 
        preventing some functionalities from working correctly. 
        """
        self.log(f'[{self.timestamp.now()}][ERROR]: { message }')

    def warn(self, message):
        """
        Unexpected behavior happened inside the application, 
        but it is continuing its work and the key business features 
        are operating as expected.
        """
        self.log(f'[{self.timestamp.now()}][WARN]: { message }')

    def info(self, message):
        """
        An event happened, the event is purely informative and 
        can be ignored during normal operations.
        """
        self.log(f'[{self.timestamp.now()}][INFO]: { message }')

    def debug(self, message):
        """ Events useful for software debugging """
        self.log(f'[{self.timestamp.now()}][DEBUG]: { message }')



#### TEST
# log = Logger(level='info', persist=True)

# for i in range(0, 30):
#     log.info('Fuck me daddy')


    
