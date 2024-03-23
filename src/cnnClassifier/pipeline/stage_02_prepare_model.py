from cnnClassifier.config import *
from cnnClassifier import logger
from cnnClassifier.components import *


STAGE_NAME = "Base Model Preparation"

class PrepareBaseModelPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()
        
if __name__ == '__main__':
    try:
        logger.info(f'>>>>> {STAGE_NAME} stage started<<<<<')
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f'>>>>> {STAGE_NAME} stage finished<<<<<')
    except Exception as e:
        logger.exception(e)
        raise e