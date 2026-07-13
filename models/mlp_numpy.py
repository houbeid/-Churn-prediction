import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# FONCTIONS D'ACTIVATION
# Adaptees de activation_functions.py + sigmoid pour binaire
# ============================================================

def sigmoid(Z):
    """Activation sigmoid pour classification binaire."""
    Z = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(A):
    """Derivee du sigmoid."""
    return A * (1 - A)

def relu(Z):
    """Activation ReLU."""
    return np.maximum(0, Z)

def relu_derivative(Z):
    """Derivee du ReLU."""
    return (Z > 0).astype(float)

def binary_cross_entropy(y_true, y_pred):
    """
    Binary Cross-Entropy loss pour classification binaire.
    Adaptee de cross_entropy() de ton code original.
    """
    eps    = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ============================================================
# DENSELAYER
# Adaptee de layers.py — ajout du Dropout
# ============================================================

class DenseLayer:
    """
    Couche dense adaptee de layers.py.
    Ajouts par rapport a l original :
    - Dropout
    - Support Adam optimizer
    - Sigmoid en plus de relu/softmax
    """

    def __init__(self, units, activation='relu',
                 weights_initializer='heUniform', dropout_rate=0.0):
        self.units               = units
        self.activation          = activation
        self.weights_initializer = weights_initializer
        self.dropout_rate        = dropout_rate

        # Poids et biais
        self.weights = None
        self.bias    = None

        # Cache pour forward/backward
        self.Z     = None
        self.A     = None
        self.input = None
        self.mask  = None   # masque dropout

        # Gradients
        self.dW = None
        self.db = None

        # Moments Adam (m = 1er moment, v = 2eme moment)
        self.m_W = None
        self.v_W = None
        self.m_b = None
        self.v_b = None

    def initialize_weights(self, input_size):
        """
        Initialise les poids.
        Meme logique que dans layers.py original.
        """
        if self.weights_initializer == 'heUniform':
            limit        = np.sqrt(6 / input_size)
            self.weights = np.random.uniform(-limit, limit, (input_size, self.units))
        elif self.weights_initializer == 'xavierUniform':
            limit        = np.sqrt(6 / (input_size + self.units))
            self.weights = np.random.uniform(-limit, limit, (input_size, self.units))
        else:
            self.weights = np.random.randn(input_size, self.units) * 0.01

        self.bias = np.zeros((1, self.units))

        # Initialiser les moments Adam a zero
        self.m_W = np.zeros_like(self.weights)
        self.v_W = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)

    def forward(self, X, training=True):
        """
        Propagation avant.
        Adaptee de layers.py — ajout sigmoid et dropout.
        """
        self.input = X
        self.Z     = np.dot(X, self.weights) + self.bias

        # Activation
        if self.activation == 'relu':
            self.A = relu(self.Z)
        elif self.activation == 'sigmoid':
            self.A = sigmoid(self.Z)
        else:
            self.A = self.Z   # lineaire

        # Dropout — seulement pendant l entraînement
        if training and self.dropout_rate > 0:
            self.mask = (np.random.rand(*self.A.shape) > self.dropout_rate)
            self.A    = self.A * self.mask / (1 - self.dropout_rate)
        else:
            self.mask = None

        return self.A


# ============================================================
# MLPNUMPYMODEL
# Adaptee de model.py (NeuralNetwork) pour le churn
# ============================================================

class MLPNumpyModel:
    """
    MLP from scratch avec NumPy pour la prediction du churn bancaire.
    Adapte de model.py (NeuralNetwork) avec :
    - Sigmoid + Binary Cross-Entropy (binaire au lieu de multi-classe)
    - Dropout
    - Adam optimizer (au lieu de SGD)
    - class_weight pour desequilibre
    - AUC comme metrique principale
    """

    def __init__(self, input_dim, random_state=42):
        self.input_dim    = input_dim
        self.random_state = random_state
        self.layers       = []

        # Historique d entraînement (meme structure que model.py)
        self.history = {
            'train_loss': [],
            'val_loss'  : [],
            'train_auc' : [],
            'val_auc'   : []
        }

        # Hyperparamètres
        self.params = {
            'layers'       : [128, 64],
            'dropout_rate' : 0.3,
            'learning_rate': 0.001,
            'epochs'       : 100,
            'batch_size'   : 512,
            'beta1'        : 0.9,     # Adam — 1er moment
            'beta2'        : 0.999,   # Adam — 2eme moment
            'epsilon'      : 1e-8,    # Adam — stabilite numerique
        }

    # ============================================================
    # CONSTRUCTION DU RESEAU
    # Adaptee de createNetwork() dans model.py
    # ============================================================

    def _build_network(self):
        """
        Construit l architecture du reseau.
        Meme logique que createNetwork() dans model.py.
        """
        layers       = self.params['layers']
        dropout_rate = self.params['dropout_rate']

        self.layers = []

        # Hidden layers avec ReLU + Dropout
        for units in layers:
            self.layers.append(DenseLayer(
                units              = units,
                activation         = 'relu',
                weights_initializer = 'heUniform',
                dropout_rate       = dropout_rate
            ))

        # Couche de sortie — Sigmoid pour binaire
        # (remplace Softmax de ton code original)
        self.layers.append(DenseLayer(
            units              = 1,
            activation         = 'sigmoid',
            weights_initializer = 'xavierUniform',
            dropout_rate       = 0.0   # pas de dropout sur la sortie
        ))

    def _initialize_network(self):
        """
        Initialise les poids.
        Meme logique que _initialize_network() dans model.py.
        """
        prev_units = self.input_dim
        for layer in self.layers:
            layer.initialize_weights(prev_units)
            prev_units = layer.units

    # ============================================================
    # FORWARD PROPAGATION
    # Adaptee de _forward_propagation() dans model.py
    # ============================================================

    def _forward(self, X, training=True):
        """
        Propagation avant.
        Meme logique que _forward_propagation() dans model.py.
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A, training=training)
        return A.flatten()

    # ============================================================
    # BACKPROPAGATION
    # Adaptee de _backward_propagation() dans model.py
    # Modification : sigmoid + BCE au lieu de softmax + CE
    # ============================================================

    def _backward(self, X, y_true, sample_weights=None):
        """
        Retropropagation.
        Adaptee de _backward_propagation() dans model.py.
        Changement principal : gradient sigmoid+BCE au lieu de softmax+CE.
        """
        m = X.shape[0]

        # Gradient de la derniere couche (sigmoid + BCE)
        # dL/dZ = (y_pred - y_true) avec poids eventuels
        A_last = self.layers[-1].A   # shape (m, 1)
        delta  = A_last - y_true.reshape(-1, 1)

        # Appliquer les sample_weights
        if sample_weights is not None:
            delta = delta * sample_weights.reshape(-1, 1)

        # Parcourir les couches en sens inverse
        # Meme logique que _backward_propagation() dans model.py
        for i in range(len(self.layers) - 1, -1, -1):
            layer  = self.layers[i]
            A_prev = X if i == 0 else self.layers[i-1].A

            # Calculer les gradients
            dW = np.dot(A_prev.T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            # Gradient clipping pour stabilite (meme que model.py)
            dW = np.clip(dW, -5, 5)
            db = np.clip(db, -5, 5)

            layer.dW = dW
            layer.db = db

            # Gradient pour la couche precedente
            if i > 0:
                delta = np.dot(delta, layer.weights.T)
                prev_layer = self.layers[i-1]

                # Appliquer derivee selon activation
                if prev_layer.activation == 'relu':
                    delta *= relu_derivative(prev_layer.Z)
                elif prev_layer.activation == 'sigmoid':
                    delta *= sigmoid_derivative(prev_layer.A)

                # Appliquer le masque dropout si present
                if prev_layer.mask is not None:
                    delta *= prev_layer.mask / (1 - prev_layer.dropout_rate)

    # ============================================================
    # ADAM OPTIMIZER
    # Remplace le SGD de model.py
    # ============================================================

    def _adam_update(self, t):
        """
        Mise a jour des poids avec Adam optimizer.
        Remplace layer.weights -= lr * dW du model.py original.
        """
        lr      = self.params['learning_rate']
        beta1   = self.params['beta1']
        beta2   = self.params['beta2']
        epsilon = self.params['epsilon']

        for layer in self.layers:
            # Mettre a jour les moments
            layer.m_W = beta1 * layer.m_W + (1 - beta1) * layer.dW
            layer.v_W = beta2 * layer.v_W + (1 - beta2) * layer.dW ** 2
            layer.m_b = beta1 * layer.m_b + (1 - beta1) * layer.db
            layer.v_b = beta2 * layer.v_b + (1 - beta2) * layer.db ** 2

            # Correction du biais
            m_W_corr = layer.m_W / (1 - beta1 ** t)
            v_W_corr = layer.v_W / (1 - beta2 ** t)
            m_b_corr = layer.m_b / (1 - beta1 ** t)
            v_b_corr = layer.v_b / (1 - beta2 ** t)

            # Mise a jour des poids
            layer.weights -= lr * m_W_corr / (np.sqrt(v_W_corr) + epsilon)
            layer.bias    -= lr * m_b_corr / (np.sqrt(v_b_corr) + epsilon)

    # ============================================================
    # ENTRAÎNEMENT
    # Adaptee de fit() dans model.py
    # ============================================================

    def train(self, X_train, y_train, X_val, y_val):
        """
        Entraîne le modele.
        Adaptee de fit() dans model.py avec :
        - class_weight pour desequilibre
        - Adam au lieu de SGD
        - AUC comme metrique
        - Sigmoid + BCE
        """
        np.random.seed(self.random_state)

        print('\n========== MLP NUMPY — ENTRAINEMENT ==========')
        print(f'  Architecture  : Input({self.input_dim}) → '
              f'{" → ".join(str(l) for l in self.params["layers"])} → 1')
        print(f'  Dropout       : {self.params["dropout_rate"]}')
        print(f'  Learning rate : {self.params["learning_rate"]} (Adam)')
        print(f'  Epochs        : {self.params["epochs"]}')
        print(f'  Batch size    : {self.params["batch_size"]}')

        # Construire et initialiser le reseau
        self._build_network()
        self._initialize_network()

        # Calculer class_weight pour desequilibre
        n_total   = len(y_train)
        n_churn   = int(y_train.sum())
        n_nochurn = n_total - n_churn
        w_churn   = n_total / (2 * n_churn)
        w_nochurn = n_total / (2 * n_nochurn)

        # Sample weights pour chaque exemple
        sample_weights = np.where(y_train == 1, w_churn, w_nochurn)

        print(f'  Class weights : 0 → {w_nochurn:.2f} | 1 → {w_churn:.2f}')

        # Early stopping (meme logique que model.py)
        best_val_loss = float('inf')
        patience      = 10
        wait          = 0
        best_weights  = None

        t = 0   # compteur Adam

        for epoch in range(self.params['epochs']):

            # Melanger les donnees (meme que model.py)
            indices            = np.random.permutation(X_train.shape[0])
            X_shuffled         = X_train[indices]
            y_shuffled         = y_train[indices]
            sw_shuffled        = sample_weights[indices]

            # Mini-batch training (meme que model.py)
            batch_size = self.params['batch_size']
            for i in range(0, X_train.shape[0], batch_size):
                X_batch  = X_shuffled[i:i+batch_size]
                y_batch  = y_shuffled[i:i+batch_size]
                sw_batch = sw_shuffled[i:i+batch_size]

                t += 1
                self._forward(X_batch, training=True)
                self._backward(X_batch, y_batch, sw_batch)
                self._adam_update(t)

            # Metriques sur train et val
            train_proba = self._forward(X_train, training=False)
            val_proba   = self._forward(X_val,   training=False)

            train_loss = binary_cross_entropy(y_train, train_proba)
            val_loss   = binary_cross_entropy(y_val,   val_proba)
            train_auc  = roc_auc_score(y_train, train_proba)
            val_auc    = roc_auc_score(y_val,   val_proba)

            # Sauvegarder historique (meme que model.py)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)

            # Afficher progression
            if (epoch + 1) % 10 == 0:
                print(f'  Epoch {epoch+1:3d}/{self.params["epochs"]} | '
                      f'Loss: {train_loss:.4f} | AUC train: {train_auc:.4f} | '
                      f'AUC val: {val_auc:.4f}')

            # Early stopping sur val_loss (meme que model.py)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait          = 0
                # Sauvegarder les meilleurs poids
                best_weights = [
                    (l.weights.copy(), l.bias.copy())
                    for l in self.layers
                ]
            else:
                wait += 1
                if wait >= patience:
                    print(f'\n  Early stopping a l epoch {epoch+1}')
                    break

        # Restaurer les meilleurs poids
        if best_weights is not None:
            for layer, (w, b) in zip(self.layers, best_weights):
                layer.weights = w
                layer.bias    = b

        print('\n  Modele entraine !')
        return self

    # ============================================================
    # EVALUATION
    # ============================================================

    def evaluate(self, X_val, y_val):
        """Evalue sur le set de validation."""
        y_pred_proba = self._forward(X_val, training=False)
        y_pred       = (y_pred_proba >= 0.5).astype(int)

        accuracy = accuracy_score(y_val, y_pred)
        auc      = roc_auc_score(y_val, y_pred_proba)

        print('\n========== MLP NUMPY — RESULTATS ==========')
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
        return self._forward(X, training=False)

    def predict(self, X):
        """Retourne les classes predites."""
        return (self.predict_proba(X) >= 0.5).astype(int)