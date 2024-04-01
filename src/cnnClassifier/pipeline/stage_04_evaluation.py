from cnnClassifier.config import *
from cnnClassifier import logger
from cnnClassifier.components import *


STAGE_NAME = "Model Evaluation"

class EvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_validation_config()
        evaluation = Evaluation(config=evaluation_config)
        evaluation.evaluation()
        evaluation.save_score()
        
if __name__ == '__main__':
    try:
        logger.info(f'>>>>> {STAGE_NAME} stage started<<<<<')
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f'>>>>> {STAGE_NAME} stage finished<<<<<')
    except Exception as e:
        logger.exception(e)
        raise e