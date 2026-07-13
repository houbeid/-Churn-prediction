import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class MLPKerasModel:
    """
    Modele MLP avec Keras pour la prediction du churn bancaire.
    """

    def __init__(self, input_dim, random_state=42):
        self.input_dim    = input_dim
        self.random_state = random_state
        self.model        = None
        self.history      = None

        # Hyperparamètres
        self.params = {
            'layers'       : [128, 64],
            'dropout_rate' : 0.3,
            'learning_rate': 0.001,
            'epochs'       : 50,
            'batch_size'   : 512,
        }

    def _build_model(self, layers, dropout_rate, learning_rate):
        from tensorflow import keras
        import tensorflow as tf
        tf.random.set_seed(self.random_state)

        model = keras.Sequential()
        model.add(keras.layers.Dense(
            layers[0], activation='relu',
            input_shape=(self.input_dim,)
        ))
        model.add(keras.layers.Dropout(dropout_rate))

        for units in layers[1:]:
            model.add(keras.layers.Dense(units, activation='relu'))
            model.add(keras.layers.Dropout(dropout_rate))

        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
            loss      = 'binary_crossentropy',
            metrics   = ['AUC']
        )
        return model

    def train(self, X_train, y_train, X_val, y_val):
        from tensorflow import keras

        print('\n========== MLP KERAS — ENTRAINEMENT ==========')
        print(f'  Architecture  : Input({self.input_dim}) → '
              f'{" → ".join(str(l) for l in self.params["layers"])} → 1')
        print(f'  Dropout       : {self.params["dropout_rate"]}')
        print(f'  Learning rate : {self.params["learning_rate"]}')
        print(f'  Epochs        : {self.params["epochs"]}')
        print(f'  Batch size    : {self.params["batch_size"]}')

        self.model = self._build_model(
            layers        = self.params['layers'],
            dropout_rate  = self.params['dropout_rate'],
            learning_rate = self.params['learning_rate']
        )

        n_total   = len(y_train)
        n_churn   = int(y_train.sum())
        n_nochurn = n_total - n_churn
        class_weight = {
            0: n_total / (2 * n_nochurn),
            1: n_total / (2 * n_churn)
        }
        print(f'  Class weights : 0 → {class_weight[0]:.2f} | 1 → {class_weight[1]:.2f}')

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_auc', patience=5, mode='max',
                verbose=1, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_auc', factor=0.5, patience=3,
                mode='max', verbose=1
            )
        ]

        self.history = self.model.fit(
            X_train, y_train,
            validation_data = (X_val, y_val),
            epochs          = self.params['epochs'],
            batch_size      = self.params['batch_size'],
            class_weight    = class_weight,
            callbacks       = callbacks,
            verbose         = 1
        )
        print('  Modele entraine !')
        return self

    def evaluate(self, X_val, y_val):
        y_pred_proba = self.model.predict(X_val, verbose=0).flatten()
        y_pred       = (y_pred_proba >= 0.5).astype(int)

        accuracy = accuracy_score(y_val, y_pred)
        auc      = roc_auc_score(y_val, y_pred_proba)

        print('\n========== MLP KERAS — RESULTATS ==========')
        print(f'  Accuracy : {accuracy:.4f}')
        print(f'  AUC      : {auc:.4f}')

        if auc >= 0.85:
            print('  Bonus avance atteint ! (AUC >= 0.85)')
        elif auc >= 0.83:
            print('  Bonus atteint ! (AUC >= 0.83)')
        elif auc >= 0.8183:
            print('  Objectif minimum atteint ! (AUC >= 0.8183)')
        else:
            print('  Objectif non atteint (AUC < 0.8183)')

        return accuracy, auc

    def predict_proba(self, X):
        return self.model.predict(X, verbose=0).flatten()

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)