import logging 
import os
from datetime import datetime

#creating as logger file
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

#creating specified log to file 
Log_path=os.path.join(os.getcwd(),"logs",LOG_FILE)

#creating log directory
os.makedirs(Log_path,exist_ok=True)

#creating log file path to insert log to the directory
Log_File_Path=os.path.join(Log_path,LOG_FILE)


#creating logger basic format
logging.basicConfig(
    filename=Log_File_Path,
    format="[%(asctime)s]%(lineno)d%(name)s-%(levelname)s-%(message)s",
    level=logging.INFO,
)

#for testing logger
if __name__=="__main__":
    logging.info("Logging has started")