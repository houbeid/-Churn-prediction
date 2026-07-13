import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings('ignore')


class MLPSklearnModel:
    """
    Modele MLP avec Scikit-learn MLPClassifier
    pour la prediction du churn bancaire.

    Note : sample_weight n est pas supporte par MLPClassifier
    dans les nouvelles versions de Scikit-learn.
    On utilise class_weight via les donnees rééchantillonnées.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model        = None
        self.best_params  = None

        # Grille d hyperparamètres
        self.param_grid = {
            'hidden_layer_sizes' : [(128, 64), (256, 128), (128, 64, 32)],
            'activation'         : ['relu', 'tanh'],
            'alpha'              : [0.0001, 0.001, 0.01],
            'learning_rate_init' : [0.001, 0.0001],
            'max_iter'           : [200],
            'early_stopping'     : [True],
            'validation_fraction': [0.1],
        }

    def _get_balanced_data(self, X_train, y_train):
        """
        Cree un dataset équilibré par oversampling de la classe minoritaire.
        Remplace sample_weight qui n est plus supporte.
        """
        idx_churn    = np.where(y_train == 1)[0]
        idx_nochurn  = np.where(y_train == 0)[0]

        # Répliquer les churners pour équilibrer
        n_repeat     = len(idx_nochurn) // len(idx_churn)
        idx_churn_up = np.tile(idx_churn, n_repeat)

        # Combiner et mélanger
        idx_balanced = np.concatenate([idx_nochurn, idx_churn_up])
        np.random.seed(self.random_state)
        np.random.shuffle(idx_balanced)

        X_balanced = X_train[idx_balanced]
        y_balanced = y_train[idx_balanced]

        print(f'  Dataset équilibré : {len(y_balanced)} exemples')
        print(f'  Churn : {y_balanced.mean()*100:.1f}%')

        return X_balanced, y_balanced

    # ============================================================
    # GRID SEARCH
    # ============================================================

    def search(self, X_train, y_train, n_iter=10, cv=3):
        """Recherche les meilleurs hyperparametres."""
        print('\n========== MLP SKLEARN — GRID SEARCH ==========')
        print(f'  Combinaisons testees : {n_iter}')
        print(f'  Cross-validation     : {cv} folds')
        print(f'  Taille X_train       : {X_train.shape}')
        print('  En cours...')

        base_model = MLPClassifier(random_state=self.random_state)

        search = RandomizedSearchCV(
            estimator           = base_model,
            param_distributions = self.param_grid,
            n_iter              = n_iter,
            cv                  = cv,
            scoring             = 'roc_auc',
            random_state        = self.random_state,
            n_jobs              = -1,
            verbose             = 1
        )

        # Pas de sample_weight — non supporte
        search.fit(X_train, y_train)
        self.best_params = search.best_params_

        print(f'\n  Meilleurs parametres trouves :')
        for param, value in self.best_params.items():
            print(f'     {param} : {value}')
        print(f'\n  AUC cross-validation : {search.best_score_:.4f}')

        return self

    # ============================================================
    # ENTRAÎNEMENT
    # ============================================================

    def train(self, X_train, y_train):
        """
        Entraine avec oversampling pour gerer le desequilibre.
        sample_weight supprime car non supporte dans Scikit-learn recent.
        """
        print('\n========== MLP SKLEARN — ENTRAINEMENT ==========')

        if self.best_params is None:
            raise ValueError("Appelle d'abord search() !")

        # Équilibrer les données par oversampling
        X_bal, y_bal = self._get_balanced_data(X_train, y_train)

        self.model = MLPClassifier(
            **self.best_params,
            random_state = self.random_state
        )

        print(f'  Hyperparametres :')
        for param, value in self.best_params.items():
            print(f'     {param} : {value}')

        # Entraîner sans sample_weight
        self.model.fit(X_bal, y_bal)
        print(f'  Iterations effectuees : {self.model.n_iter_}')
        print('  Modele entraine !')
        return self

    # ============================================================
    # ÉVALUATION
    # ============================================================

    def evaluate(self, X_val, y_val):
        """Evalue sur le set de validation."""
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        y_pred       = self.model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        auc      = roc_auc_score(y_val, y_pred_proba)

        print('\n========== MLP SKLEARN — RESULTATS ==========')
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
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)