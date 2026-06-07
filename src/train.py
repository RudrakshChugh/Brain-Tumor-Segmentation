import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics

try:
    from model import create_unet
except ImportError:
    from .model import create_unet

class DiceScore(tf.keras.metrics.Metric):
    """Mean Dice score metric for binary segmentation masks."""
    def __init__(self, name="DiceScore", **kwargs):
        super(DiceScore, self).__init__(name=name, **kwargs)
        self.mean_dice = metrics.Mean(name="DiceScore_Tracker")
    
    def update_state(self, real, pred, sample_weight=None):
        numerator = 2. * tf.reduce_sum(tf.multiply(real, pred), axis=[1,2])
        denominator = tf.reduce_sum(tf.add(real, pred), axis=[1,2])
        x = tf.math.divide_no_nan(numerator, denominator)
        x = tf.reduce_mean(x)
        self.mean_dice.update_state(x)
    
    def result(self):
        return self.mean_dice.result()

    def reset_state(self):
        self.mean_dice.reset_state()

def combined_loss():
    """Return a combined Dice loss and weighted BCE loss function."""
    def convert_to_logits(y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1-tf.keras.backend.epsilon())
        return tf.math.log(y_pred / (1 - y_pred))
    
    def loss(y_true, y_pred, beta):
        y_pred = convert_to_logits(y_pred)
        pos_weight = beta / (1-beta)
        loss = tf.nn.weighted_cross_entropy_with_logits(
            logits=y_pred,
            labels=y_true,
            pos_weight=pos_weight,
        )
        return tf.reduce_mean(loss * (1-beta))
    
    def dice_loss(y_true, y_pred):
        numerator = 2. * tf.reduce_sum(tf.multiply(y_true, y_pred), axis=[1,2])
        denominator = tf.reduce_sum(tf.add(y_true, y_pred), axis=[1,2])
        x = tf.math.divide_no_nan(numerator, denominator)
        return tf.reduce_mean(1 - x)
    
    def combined(y_true, y_pred,):
        return 0.5*dice_loss(y_true, y_pred) + 0.5*loss(y_true, y_pred, 0.8)
    
    return combined

def train_model(X_train, y_train, X_val, y_val, batch_size=4, n_epochs=40, learning_rate=1e-4):
    """Train the 2D U-Net model and return the model with its Keras history."""
    inputs = tf.keras.layers.Input((128, 128, 3))
    model = create_unet(inputs)
    train_steps = X_train.shape[0] // batch_size
    valid_steps = X_val.shape[0] // batch_size
    decay_steps = train_steps * int(0.85 * n_epochs)
    alpha = 1e-3
    lr = tf.keras.optimizers.schedules.CosineDecay(learning_rate, decay_steps=decay_steps, alpha=alpha)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr, decay=1e-5), loss=combined_loss(), metrics=[DiceScore()])
    cbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_DiceScore", patience=np.ceil(0.05*n_epochs).astype(int), mode="max", restore_best_weights=True)]
    model_history = model.fit(
        X_train, 
        y_train,
        validation_data=(X_val, y_val),
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        epochs=n_epochs,
        callbacks=cbacks
    )
    return model, model_history
