from cnnClassifier import logger
from cnnClassifier.pipeline import (DataIngestionTrainingPipeline, PrepareBaseModelPipeline, PrepareCallbacksPipeline)


STAGE_NAME = 'Data Ingestion'
try:
    logger.info(f'>>>>>>>>>>> {STAGE_NAME} stage started<<<<<')
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f'>>>>>>>>>>> {STAGE_NAME} stage finished<<<<<\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = 'Base Model Preparation'
try:
    logger.info(f'>>>>>>>>>>> {STAGE_NAME} stage started<<<<<')
    data_ingestion = PrepareBaseModelPipeline()
    data_ingestion.main()
    logger.info(f'>>>>>>>>>>> {STAGE_NAME} stage finished<<<<<\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = 'Prepare Callbacks'
try:
    logger.info(f'>>>>> {STAGE_NAME} stage started<<<<<')
    obj = PrepareCallbacksPipeline()
    obj.main()
    logger.info(f'>>>>> {STAGE_NAME} stage finished<<<<<')
except Exception as e:
    logger.exception(e)
    raise e