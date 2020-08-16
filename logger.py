import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler
import time
import os

def setup_logging(config, default_level=logging.INFO):
    try: 
        logging.config.dictConfig(config)
    except Exception as e:
        print(e)
        print('Error in Logging Configuration. Using default configs')
        logging.basicConfig(level=default_level)
        
class CustomTimedRotatingHandler(TimedRotatingFileHandler):
    """
    Custom rotating file handler inheriting logging.handlers.TimedRotatingFileHandler
    Changes rollover filename to YYYY-MM-DD.log (subject to 'when' and 'interval' arguments)
    Newest log filename (the file being written to) is from config
    """
    def __init__(self, filename, when='h', interval=1, backupCount=0, encoding=None):
        TimedRotatingFileHandler.__init__(
            self, 
            filename=filename, 
            when=when, 
            interval=interval, 
            backupCount=backupCount, 
            encoding=encoding)
    
    def doRollover(self):
        """
        Overrides TimedRotatingFileHandler.doRollover()
        Changes rotation filename, which also makes getFilesToDelete() useless
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        # get the time that this sequence started at and make it a TimeTuple
        currentTime = int(time.time())
        dstNow = time.localtime(currentTime)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
            dstThen = timeTuple[-1]
            if dstNow != dstThen:
                if dstNow:
                    addend = 3600
                else:
                    addend = -3600
                timeTuple = time.localtime(t + addend)
        ################# Changes from here ################
        dirName, _ = os.path.split(self.baseFilename)
        dfn = self.rotation_filename(os.path.join(
            dirName, time.strftime(self.suffix, timeTuple) + '.log'))
        ################# Changes end here ###################
        if os.path.exists(dfn):
            os.remove(dfn)
        self.rotate(self.baseFilename, dfn)
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
        if not self.delay:
            self.stream = self._open()
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        #If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                    addend = -3600
                else:           # DST bows out before next rollover, so we need to add an hour
                    addend = 3600
                newRolloverAt += addend
        self.rolloverAt = newRolloverAt