from cnnClassifier import logger
from cnnClassifier.pipeline import (DataIngestionTrainingPipeline, 
                                    PrepareBaseModelPipeline, 
                                    ModelTrainingPipeline,
                                    EvaluationPipeline)


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
    prepare_base_model = PrepareBaseModelPipeline()
    prepare_base_model.main()
    logger.info(f'>>>>>>>>>>> {STAGE_NAME} stage finished<<<<<\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = 'Model Training'
try:
    logger.info(f'>>>>> {STAGE_NAME} stage started<<<<<')
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f'>>>>> {STAGE_NAME} stage finished<<<<<')
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = 'Model Evaluation'
try:
    logger.info(f'>>>>> {STAGE_NAME} stage started<<<<<')
    model_evaluation = EvaluationPipeline()
    model_evaluation.main()
    logger.info(f'>>>>> {STAGE_NAME} stage finished<<<<<')
except Exception as e:
    logger.exception(e)
    raise e