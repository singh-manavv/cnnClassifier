import os
from cnnClassifier.entity import *
from cnnClassifier.utils import *
from cnnClassifier import logger
from pathlib import Path
import tensorflow as tf


class PrepareBaseModel:
    def __init__(self,config: PrepareBaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        logger.info("Preparing base model")
        self.model = tf.keras.applications.VGG16(
                        include_top = self.config.params_include_top,
                        weights = self.config.params_weights,
                        input_shape = self.config.params_image_size
                    )
        logger.info("Saving base model")
        self.save_model(path=self.config.base_model_path,model=self.model)
        
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False
        
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units = classes,
            activation = "softmax"
        )(flatten_in)
        
        full_model = tf.keras.models.Model(
            inputs = model.input,
            outputs = prediction
        )
        
        full_model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics = ['accuracy']
        )
        full_model.summary()
        
        return full_model
    
    def update_base_model(self):
        logger.info("Updating base model")
        self.full_model = self._prepare_full_model(
                            model = self.model,
                            classes = self.config.params_classes,
                            freeze_all = True,
                            freeze_till = None,
                            learning_rate= self.config.params_learning_rate
                        )
        logger.info("Saving updated model")
        self.save_model(path=self.config.updated_base_model_path,model=self.full_model)
        
    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)