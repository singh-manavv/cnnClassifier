from cnnClassifier import logger
from cnnClassifier.pipeline import DataIngestionTrainingPipeline


STAGE_NAME = 'Data Ingestion'
try:
    logger.info(f'>>>>>>>>>>> {STAGE_NAME} stage started<<<<<')
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f'>>>>>>>>>>> {STAGE_NAME} stage finished<<<<<\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
except Exception as e:
    logger.exception(e)
    raise e