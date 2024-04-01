import tensorflow as tf
from cnnClassifier.entity import *
from cnnClassifier.utils import *
from pathlib import Path



class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def _valid_generator(self):
        
        datagenerator_kwargs = dict(
            rescale= 1./255,
            validation_split = 0.20
        )
        
        dataflow_kwargs = dict(
            target_size = self.config.params_image_size[:-1],
            batch_size = self.config.params_batch_size,
            interpolation = 'bilinear'
        )
        
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory= self.config.training_data,
            subset='validation',
            shuffle= False,
            **dataflow_kwargs
        )
        
    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def evaluation(self):
        model = self.load_model(self.config.model_path)
        self._valid_generator()
        self.score = model.evaluate(self.valid_generator)
        
    
    def save_score(self):
        scores = {'Loss': self.score[0], 'Accuracy': self.score[1]}
        save_json(path=Path('scores.json'),data=scores)