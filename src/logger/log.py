from logging import getLogger, StreamHandler, INFO, DEBUG, Formatter, FileHandler


def debug_logger(log_dir):
    logger = getLogger('train')
    logger.setLevel(DEBUG)

    fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')

    sh = StreamHandler()
    sh.setLevel(INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = FileHandler(filename=log_dir.joinpath('debug.txt'), mode='w')
    fh.setLevel(DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
