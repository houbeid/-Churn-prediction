import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class RandomForestModel:
    """
    Modele Random Forest avec RandomizedSearchCV
    pour la prediction du churn bancaire.
    """

    def __init__(self, random_state=42, n_jobs=-1):
        self.random_state = random_state
        self.n_jobs       = n_jobs
        self.model        = None
        self.best_params  = None

        # Grille d'hyperparamètres
        self.param_grid = {
            'n_estimators'     : [100, 200, 300],
            'max_depth'        : [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf' : [1, 2, 4],
            'max_features'     : ['sqrt', 'log2'],
            'class_weight'     : ['balanced'],
        }

    # ============================================================
    # GRID SEARCH
    # ============================================================

    def search(self, X_train, y_train, n_iter=20, cv=3):
        """Recherche les meilleurs hyperparametres avec RandomizedSearchCV."""
        print('\n========== RANDOM FOREST — GRID SEARCH ==========')
        print(f'  Combinaisons testees : {n_iter}')
        print(f'  Cross-validation     : {cv} folds')
        print(f'  Taille X_train       : {X_train.shape}')
        print('  En cours... (peut prendre 20-40 minutes)')

        base_model = RandomForestClassifier(
            random_state = self.random_state,
            n_jobs       = self.n_jobs
        )

        search = RandomizedSearchCV(
            estimator           = base_model,
            param_distributions = self.param_grid,
            n_iter              = n_iter,
            cv                  = cv,
            scoring             = 'roc_auc',
            random_state        = self.random_state,
            n_jobs              = self.n_jobs,
            verbose             = 1
        )

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
        """Entraine avec les meilleurs hyperparametres trouves par search()."""
        print('\n========== RANDOM FOREST — ENTRAINEMENT ==========')

        if self.best_params is None:
            raise ValueError("Appelle d'abord search() !")

        self.model = RandomForestClassifier(
            **self.best_params,
            random_state = self.random_state,
            n_jobs       = self.n_jobs
        )

        print(f'  Hyperparametres :')
        for param, value in self.best_params.items():
            print(f'     {param} : {value}')

        self.model.fit(X_train, y_train)
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

        print('\n========== RANDOM FOREST — RESULTATS ==========')
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

    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================

    def feature_importance(self, feature_names, top_n=20):
        """Affiche les features les plus importantes."""
        import pandas as pd
        importances = self.model.feature_importances_
        df = pd.DataFrame({
            'feature'   : feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        print(f'\n========== TOP {top_n} FEATURES IMPORTANTES ==========')
        for _, row in df.iterrows():
            bar = '|' * int(row['importance'] * 200)
            print(f'  {row["feature"]:<40} {row["importance"]:.4f} {bar}')

        return df