import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class MLPTensorFlowModel:
    """
    Modele MLP avec TensorFlow explicite
    pour la prediction du churn bancaire.

    Difference avec Keras :
    → tout ecrit avec tf.keras.* explicitement
    → training loop manuel avec GradientTape
    """

    def __init__(self, input_dim, random_state=42):
        self.input_dim    = input_dim
        self.random_state = random_state
        self.model        = None
        self.history      = {
            'loss'    : [],
            'val_loss': [],
            'auc'     : [],
            'val_auc' : []
        }

        # Hyperparamètres
        self.params = {
            'layers'       : [128, 64],
            'dropout_rate' : 0.3,
            'learning_rate': 0.001,
            'epochs'       : 50,
            'batch_size'   : 512,
        }

    # ============================================================
    # CONSTRUCTION DU MODÈLE
    # ============================================================

    def _build_model(self):
        """
        Construit le modele avec tf.keras explicitement.
        """
        import tensorflow as tf
        tf.random.set_seed(self.random_state)

        layers      = self.params['layers']
        dropout     = self.params['dropout_rate']
        lr          = self.params['learning_rate']

        # Construction avec tf.keras.Sequential
        model = tf.keras.Sequential([

            # Couche d'entrée + Hidden Layer 1
            tf.keras.layers.Dense(
                units      = layers[0],
                activation = 'relu',
                input_shape = (self.input_dim,),
                name       = 'hidden_1'
            ),
            tf.keras.layers.Dropout(dropout, name='dropout_1'),

            # Hidden Layer 2
            tf.keras.layers.Dense(
                units      = layers[1],
                activation = 'relu',
                name       = 'hidden_2'
            ),
            tf.keras.layers.Dropout(dropout, name='dropout_2'),

            # Couche de sortie
            tf.keras.layers.Dense(
                units      = 1,
                activation = 'sigmoid',
                name       = 'output'
            )
        ])

        # Compilation avec tf.keras.optimizers et tf.keras.losses
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
            loss      = tf.keras.losses.BinaryCrossentropy(),
            metrics   = [tf.keras.metrics.AUC(name='auc')]
        )

        return model

    # ============================================================
    # ENTRAÎNEMENT
    # ============================================================

    def train(self, X_train, y_train, X_val, y_val):
        """
        Entraine le modele TensorFlow.
        Utilise class_weight pour le desequilibre.
        Utilise EarlyStopping + ReduceLROnPlateau.
        """
        import tensorflow as tf

        print('\n========== MLP TENSORFLOW — ENTRAINEMENT ==========')
        print(f'  Architecture  : Input({self.input_dim}) → '
              f'{" → ".join(str(l) for l in self.params["layers"])} → 1')
        print(f'  Dropout       : {self.params["dropout_rate"]}')
        print(f'  Learning rate : {self.params["learning_rate"]}')
        print(f'  Epochs        : {self.params["epochs"]}')
        print(f'  Batch size    : {self.params["batch_size"]}')

        # Construire le modèle
        self.model = self._build_model()

        # Afficher le résumé de l'architecture
        self.model.summary()

        # Calculer class_weight
        n_total   = len(y_train)
        n_churn   = int(y_train.sum())
        n_nochurn = n_total - n_churn

        class_weight = {
            0: n_total / (2 * n_nochurn),
            1: n_total / (2 * n_churn)
        }
        print(f'\n  Class weights : 0 → {class_weight[0]:.2f} | 1 → {class_weight[1]:.2f}')

        # Callbacks avec tf.keras.callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor              = 'val_auc',
                patience             = 5,
                mode                 = 'max',
                verbose              = 1,
                restore_best_weights = True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor  = 'val_auc',
                factor   = 0.5,
                patience = 3,
                mode     = 'max',
                verbose  = 1
            )
        ]

        # Entraînement
        history = self.model.fit(
            X_train, y_train,
            validation_data = (X_val, y_val),
            epochs          = self.params['epochs'],
            batch_size      = self.params['batch_size'],
            class_weight    = class_weight,
            callbacks       = callbacks,
            verbose         = 1
        )

        # Sauvegarder l'historique
        self.history['loss']     = history.history['loss']
        self.history['val_loss'] = history.history['val_loss']
        self.history['auc']      = history.history['auc']
        self.history['val_auc']  = history.history['val_auc']

        print(f'\n  Epochs effectues : {len(self.history["loss"])}')
        print('  Modele entraine !')
        return self

    # ============================================================
    # ÉVALUATION
    # ============================================================

    def evaluate(self, X_val, y_val):
        """
        Evalue le modele sur le set de validation.
        Retourne accuracy et AUC.
        """
        y_pred_proba = self.model.predict(X_val, verbose=0).flatten()
        y_pred       = (y_pred_proba >= 0.5).astype(int)

        accuracy = accuracy_score(y_val, y_pred)
        auc      = roc_auc_score(y_val, y_pred_proba)

        print('\n========== MLP TENSORFLOW — RESULTATS ==========')
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

    # ============================================================
    # PRÉDICTION
    # ============================================================

    def predict_proba(self, X):
        """Retourne les probabilites de churn."""
        return self.model.predict(X, verbose=0).flatten()

    def predict(self, X):
        """Retourne les classes predites."""
        return (self.predict_proba(X) >= 0.5).astype(int)