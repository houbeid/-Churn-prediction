import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Classe de preprocessing pour le projet Churn Prediction.
    Gère les valeurs manquantes, l'encodage, la normalisation et le split.
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size    = test_size
        self.random_state = random_state

        self.cols_to_drop   = []
        self.cols_to_flag   = []
        self.cols_to_impute = []
        self.cols_important = [
            'SUM_TRAN_CLO_TENDENCY1M', 'CNT_TRAN_CLO_TENDENCY1M',
            'SUM_TRAN_MED_TENDENCY1M', 'CNT_TRAN_MED_TENDENCY1M',
            'SUM_TRAN_AUT_TENDENCY1M', 'CNT_TRAN_AUT_TENDENCY1M',
            'SUM_TRAN_AUT_TENDENCY3M', 'CNT_TRAN_AUT_TENDENCY3M',
            'CNT_TRAN_CLO_TENDENCY3M', 'SUM_TRAN_CLO_TENDENCY3M',
            'CNT_TRAN_MED_TENDENCY3M', 'SUM_TRAN_MED_TENDENCY3M',
            'PRC_ACCEPTS_A_EMAIL_LINK', 'PRC_ACCEPTS_A_POS',
            'PRC_ACCEPTS_A_AMOBILE',   'PRC_ACCEPTS_A_MTP',
            'PRC_ACCEPTS_A_TK',        'PRC_ACCEPTS_A_ATM',
            'PRC_ACCEPTS_TK',          'CNT_ACCEPTS_TK',
            'CNT_ACCEPTS_MTP',         'PRC_ACCEPTS_MTP',
        ]
        self.groupes_flags = {
            'TRAN_CLO_1M_missing' : ['SUM_TRAN_CLO_TENDENCY1M', 'CNT_TRAN_CLO_TENDENCY1M'],
            'TRAN_MED_1M_missing' : ['SUM_TRAN_MED_TENDENCY1M', 'CNT_TRAN_MED_TENDENCY1M'],
            'TRAN_AUT_1M_missing' : ['SUM_TRAN_AUT_TENDENCY1M', 'CNT_TRAN_AUT_TENDENCY1M'],
            'TRAN_AUT_3M_missing' : ['SUM_TRAN_AUT_TENDENCY3M', 'CNT_TRAN_AUT_TENDENCY3M'],
            'TRAN_CLO_3M_missing' : ['CNT_TRAN_CLO_TENDENCY3M', 'SUM_TRAN_CLO_TENDENCY3M'],
            'TRAN_MED_3M_missing' : ['CNT_TRAN_MED_TENDENCY3M', 'SUM_TRAN_MED_TENDENCY3M'],
            'ACCEPTS_missing'     : [
                'PRC_ACCEPTS_A_EMAIL_LINK', 'PRC_ACCEPTS_A_POS',
                'PRC_ACCEPTS_A_AMOBILE',    'PRC_ACCEPTS_A_MTP',
                'PRC_ACCEPTS_A_TK',         'PRC_ACCEPTS_A_ATM',
                'PRC_ACCEPTS_TK',           'CNT_ACCEPTS_TK',
                'CNT_ACCEPTS_MTP',          'PRC_ACCEPTS_MTP',
            ],
        }
        self.cols_to_remove  = ['CLNT_JOB_POSITION']
        self.imputer_num     = SimpleImputer(strategy='median')
        self.imputer_cat     = SimpleImputer(strategy='most_frequent')
        self.label_encoders  = {}
        self.scaler          = StandardScaler()
        self.feature_cols    = []   # colonnes features (sans ID et TARGET)

    # ============================================================
    # MISSING VALUES
    # ============================================================

    def _compute_missing(self, train):
        """Calcule les colonnes à supprimer, flaguer et imputer."""
        missing_pct = train.isnull().sum() / len(train) * 100

        self.cols_to_drop   = missing_pct[missing_pct > 50].index.tolist()
        self.cols_to_flag   = missing_pct[(missing_pct > 20) & (missing_pct <= 50)].index.tolist()
        self.cols_to_impute = missing_pct[(missing_pct > 0)  & (missing_pct <= 20)].index.tolist()

        self.cols_important = [c for c in self.cols_important if c in train.columns]
        self.cols_to_drop   = [c for c in self.cols_to_drop if c not in self.cols_important]

        print(f'   Colonnes supprimées   (> 50%)  : {len(self.cols_to_drop)}')
        print(f'   Flag + imputer        (20-50%) : {len(self.cols_to_flag)}')
        print(f'   Imputer only          (< 20%)  : {len(self.cols_to_impute)}')
        print(f'   Colonnes importantes sauvées    : {len(self.cols_important)}')

    def _create_flags(self, train, test):
        """Crée les colonnes binaires _missing."""
        flags_train = {}
        flags_test  = {}

        for flag_name, colonnes in self.groupes_flags.items():
            col_ref = colonnes[0]
            if col_ref in train.columns:
                flags_train[flag_name] = train[col_ref].isnull().astype(int)
                flags_test[flag_name]  = test[col_ref].isnull().astype(int)

        for col in self.cols_to_flag:
            flag_name = f'{col}_missing'
            flags_train[flag_name] = train[col].isnull().astype(int)
            flags_test[flag_name]  = test[col].isnull().astype(int)

        train = pd.concat([train, pd.DataFrame(flags_train, index=train.index)], axis=1)
        test  = pd.concat([test,  pd.DataFrame(flags_test,  index=test.index)],  axis=1)

        print(f'   Flags créés : {len(flags_train)}')
        return train, test

    def _impute(self, train, test):
        """Impute les NaN (médiane pour num, mode pour cat)."""
        all_cols = self.cols_important + self.cols_to_flag + self.cols_to_impute
        all_cols = [c for c in all_cols if c in train.columns]

        num_cols = [c for c in all_cols if train[c].dtype in
                    [np.float64, np.int64, np.float32, np.int32]]
        cat_cols = [c for c in all_cols if train[c].dtype == object]

        if num_cols:
            train[num_cols] = self.imputer_num.fit_transform(train[num_cols])
            test[num_cols]  = self.imputer_num.transform(test[num_cols])
            print(f'   {len(num_cols)} colonnes numériques imputées (médiane)')

        if cat_cols:
            train[cat_cols] = self.imputer_cat.fit_transform(train[cat_cols])
            test[cat_cols]  = self.imputer_cat.transform(test[cat_cols])
            print(f'   {len(cat_cols)} colonnes catégorielles imputées (mode)')

        return train, test

    def handle_missing(self, train, test):
        """Pipeline complet : compute → flags → drop → impute."""
        print('\n========== GESTION DES MANQUANTS ==========')
        self._compute_missing(train)
        train, test = self._create_flags(train, test)
        train.drop(columns=self.cols_to_drop, inplace=True)
        test.drop(columns=self.cols_to_drop,  inplace=True)
        print(f'    {len(self.cols_to_drop)} colonnes supprimées')
        train, test = self._impute(train, test)
        print(f'  Manquants restants train : {train.isnull().sum().sum()}')
        print(f'  Manquants restants test  : {test.isnull().sum().sum()}')
        print(' Manquants terminé !')
        return train, test

    # ============================================================
    # ENCODAGE
    # ============================================================

    def _remove_useless_cols(self, train, test):
        """Supprime les colonnes redondantes."""
        cols = [c for c in self.cols_to_remove if c in train.columns]
        train.drop(columns=cols, inplace=True)
        test.drop(columns=cols,  inplace=True)
        print(f'    Colonnes redondantes supprimées : {cols}')
        return train, test

    def _label_encode(self, train, test):
        """Label Encoding sur toutes les colonnes catégorielles."""
        cat_cols = train.select_dtypes(include=['object', 'str']).columns.tolist()
        cat_cols = [c for c in cat_cols if c not in ['ID']]

        for col in cat_cols:
            le         = LabelEncoder()
            all_values = pd.concat([train[col], test[col]], axis=0).astype(str)
            le.fit(all_values)
            train[col] = le.transform(train[col].astype(str))
            test[col]  = le.transform(test[col].astype(str))
            self.label_encoders[col] = le
            print(f'   Label Encoded : {col} ({len(le.classes_)} valeurs)')

        return train, test

    def encode(self, train, test):
        """Pipeline complet d'encodage."""
        print('\n========== ENCODAGE ==========')
        train, test = self._remove_useless_cols(train, test)
        train, test = self._label_encode(train, test)

        remaining  = train.select_dtypes(include=['object', 'str']).columns.tolist()
        remaining  = [c for c in remaining if c not in ['ID']]
        cols_train = set(train.columns)
        cols_test  = set(test.columns)

        if len(remaining) == 0:
            print('   Toutes les colonnes catégorielles sont encodées !')
        else:
            print(f'    Colonnes texte restantes : {remaining}')

        if cols_train == cols_test:
            print('   Train et test ont les mêmes colonnes !')
        else:
            print(f'    Diff train/test : {cols_train - cols_test}')

        print(' Encodage terminé !')
        return train, test

    # ============================================================
    # NORMALISATION
    # ============================================================

    def normalize(self, train, test):
        """
        Normalise les features numériques avec StandardScaler.
        fit sur train uniquement → transform sur train et test.
        Les flags (0/1) et ID/TARGET sont exclus.
        """
        print('\n========== NORMALISATION ==========')

        # Colonnes à normaliser : numériques sauf ID, TARGET et flags _missing
        self.feature_cols = [
            c for c in train.columns
            if c not in ['ID', 'TARGET']
            and not c.endswith('_missing')
            and train[c].dtype in [np.float64, np.int64, np.float32, np.int32]
        ]

        # fit sur train uniquement
        train[self.feature_cols] = self.scaler.fit_transform(train[self.feature_cols])
        test[self.feature_cols]  = self.scaler.transform(test[self.feature_cols])

        print(f'   {len(self.feature_cols)} colonnes normalisées (StandardScaler)')
        print(f'  Moyenne train (vérif) ≈ {train[self.feature_cols].mean().mean():.6f} (doit être ≈ 0)')
        print(f'  Std    train (vérif) ≈ {train[self.feature_cols].std().mean():.6f}  (doit être ≈ 1)')
        print(' Normalisation terminée !')
        return train, test

    # ============================================================
    # SPLIT TRAIN / VALIDATION
    # ============================================================

    def split(self, train):
        """
        Split stratifié 80/20 sur les données train.
        Stratifié pour conserver les proportions du churn (8.1% / 91.9%).
        Retourne X_train, X_val, y_train, y_val
        """
        print('\n========== SPLIT TRAIN / VALIDATION ==========')

        feature_cols = [c for c in train.columns if c not in ['ID', 'TARGET']]
        X = train[feature_cols].values
        y = train['TARGET'].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size    = self.test_size,
            random_state = self.random_state,
            stratify     = y        # stratifié sur TARGET
        )

        print(f'  X_train : {X_train.shape}  |  y_train : {y_train.shape}')
        print(f'  X_val   : {X_val.shape}    |  y_val   : {y_val.shape}')
        print(f'  Churn dans train : {y_train.mean()*100:.2f}%')
        print(f'  Churn dans val   : {y_val.mean()*100:.2f}%')
        print(' Split terminé !')
        return X_train, X_val, y_train, y_val

    # ============================================================
    # PIPELINE COMPLET
    # ============================================================

    def fit_transform(self, train, test):
        """
        Lance le preprocessing complet dans l'ordre :
        1. Gestion des manquants
        2. Encodage
        3. Normalisation
        4. Split train / validation
        Retourne : X_train, X_val, y_train, y_val, X_test
        """
        print('\n' + '=' * 50)
        print('       PREPROCESSING COMPLET')
        print('=' * 50)

        train, test              = self.handle_missing(train, test)
        train, test              = self.encode(train, test)
        train, test              = self.normalize(train, test)
        X_train, X_val, y_train, y_val = self.split(train)

        # Préparer X_test (sans ID et TARGET)
        feature_cols = [c for c in test.columns if c not in ['ID', 'TARGET']]
        X_test       = test[feature_cols].values

        print('\n' + '=' * 50)
        print(' PREPROCESSING TERMINÉ !')
        print(f'   X_train : {X_train.shape}')
        print(f'   X_val   : {X_val.shape}')
        print(f'   X_test  : {X_test.shape}')
        print('=' * 50)

        return X_train, X_val, y_train, y_val, X_test