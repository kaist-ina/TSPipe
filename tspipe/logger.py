import os
from enum import Enum
from functools import total_ordering


@total_ordering
class LogLevel(Enum):
    DEBUG = 1
    VERBOSE = 2
    INFO = 3
    WARN = 4
    ERROR = 5
    FATAL = 6

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.value == other.value
        return NotImplemented


current_loglevel = LogLevel.WARN
FORCE_FLUSH_LOG = True


class Log:
    @staticmethod
    def v(*msg):
        """Verbose log"""
        if current_loglevel <= LogLevel.VERBOSE:
            print(f'[{os.getpid()}][V] ', *msg, flush=FORCE_FLUSH_LOG)

    @staticmethod
    def d(*msg):
        """Debug log"""
        if current_loglevel <= LogLevel.DEBUG:
            print(f'[{os.getpid()}][D] ', *msg, flush=FORCE_FLUSH_LOG)

    @staticmethod
    def i(*msg):
        """Info log"""
        if current_loglevel <= LogLevel.INFO:
            print(f'[{os.getpid()}][I] ', *msg, flush=FORCE_FLUSH_LOG)

    @staticmethod
    def w(*msg):
        """Warn log"""
        if current_loglevel <= LogLevel.WARN:
            print(f'[{os.getpid()}][W] ', *msg, flush=FORCE_FLUSH_LOG)

    @staticmethod
    def e(*msg):
        """Error log"""
        if current_loglevel <= LogLevel.ERROR:
            print(f'[{os.getpid()}][E] ', *msg, flush=FORCE_FLUSH_LOG)
