from colorama import Fore, Back, Style, init
import logging
from sys import stdout

# NOT important to review any of the following functions


########################################################################
# CODING PATTERS
########################################################################

# SINGLETON

class SingletonMeta(type):
    """
    Implementation of the Singleton pattern as Python metaclass.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


###########################################################################
# LOGGER
###########################################################################

class Logger(logging.Logger, metaclass=SingletonMeta):
    def __init__(self):
        fmt: str = "%(asctime)s %(levelname)8s >> %(message)s"
        # fmt: str = "%(asctime)s %(levelname)8s: %(name)18s >> %(message)s"
        date_format: str = "%m/%d/%Y %I:%M:%S %p"
        formatter = ColoredFormatter(fmt, datefmt=date_format)

        # now we set the main / base logger called 'parent'
        super().__init__('parent', logging.DEBUG)

        # now we add a handler to print all the debug shit in the terminal it
        handler = logging.StreamHandler(stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        self.addHandler(handler)

        # and another one where to save all the info in a file...
        f_handler = logging.FileHandler('info.log')
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(formatter)
        self.addHandler(f_handler)

    def _log(self, level: int, msg: object, args, **kwargs) -> None:
        """
        Log a message at a given level and always including the module's name
        from where the call was made.
        """
        # module_name = sys.modules[__name__].__spec__.name
        # msg = f'{self.module_name}: {msg}'
        logging.Logger.__dict__['_log'](self, level, msg, args, **kwargs)


# Prepare the colored formatter
init(autoreset=True)
colors = {"DEBUG": Fore.BLUE,  # "INFO": Fore.CYAN,
          "WARNING": Fore.YELLOW, "ERROR": Fore.RED,
          "CRITICAL": Fore.WHITE + Back.RED + Style.BRIGHT}


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        if record.levelname in colors:
            msg = colors[record.levelname] + msg + \
                  Fore.RESET + Back.RESET + Style.RESET_ALL
        return msg
