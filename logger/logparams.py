import logging
import datetime
class Debugger:

    def __init__(self, filename="test_log"):
        """
        initialize debugger with filename
        :param filename:
        """
        hanlder = logging.FileHandler(filename="%s_%s.log" % (filename, datetime.datetime.now().strftime("%d_%b")),
                                      mode='w', encoding="UTF-8")
        logging.basicConfig(level=Debug_param.debug_scope(),
                            handlers=[hanlder])
        self.logger = logging.getLogger("Debugger")
        # logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("\n\n%(asctime)s - %(name)s - %(levelname)s:\n\n%(message)s")
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(formatter)
        verbose = False
        # if (logger.hasHandlers()):
        #     logger.handlers.clear()
        # logger.addHandler(file_handler)
        if verbose:
            self.logger.addHandler(self.stream_handler)
        self.logger.debug("\n{}\nScript is runing\n{}".format("=" * 50, "=" * 50))


class Debug_param:

    @staticmethod
    def debug_scope():
        return logging.DEBUG





