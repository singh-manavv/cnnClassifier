from cnnClassifier.config import *
from cnnClassifier import logger
from cnnClassifier.components import *


STAGE_NAME = "Prepare Callbacks"

class PrepareCallbacksPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_callbacks_config()
        prepare_base_model = PrepareCallback(config=prepare_base_model_config)
        prepare_base_model.get_tb_ckpt_callbacks()
        
if __name__ == '__main__':
    try:
        logger.info(f'>>>>> {STAGE_NAME} stage started<<<<<')
        obj = PrepareCallbacksPipeline()
        obj.main()
        logger.info(f'>>>>> {STAGE_NAME} stage finished<<<<<')
    except Exception as e:
        logger.exception(e)
        raise e