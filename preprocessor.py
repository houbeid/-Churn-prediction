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
    Gere les valeurs manquantes, l encodage, la normalisation et le split.
    Compatible Pandas 2 et Pandas 3.
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
        self.cols_to_remove = ['CLNT_JOB_POSITION']
        self.label_encoders = {}
        self.scaler         = StandardScaler()
        self.feature_cols   = []

    # ============================================================
    # UTILITAIRE — Detection des colonnes categorielles
    # ============================================================

    def _get_cat_cols(self, df, exclude=None):
        """
        Retourne les colonnes categorielles.
        Compatible Pandas 2 (dtype object) et Pandas 3 (dtype string).
        """
        if exclude is None:
            exclude = ['ID', 'TARGET']
        cat_cols = []
        for col in df.columns:
            if col in exclude:
                continue
            dtype_str = str(df[col].dtype)
            # Pandas 2 → object / Pandas 3 → string ou StringDtype
            if dtype_str == 'object' or 'string' in dtype_str.lower():
                cat_cols.append(col)
        return cat_cols

    # ============================================================
    # MISSING VALUES
    # ============================================================

    def _compute_missing(self, train):
        """Calcule les colonnes a supprimer, flaguer et imputer."""
        missing_pct = train.isnull().sum() / len(train) * 100

        self.cols_to_drop   = missing_pct[missing_pct > 50].index.tolist()
        self.cols_to_flag   = missing_pct[(missing_pct > 20) & (missing_pct <= 50)].index.tolist()
        self.cols_to_impute = missing_pct[(missing_pct > 0)  & (missing_pct <= 20)].index.tolist()

        self.cols_important = [c for c in self.cols_important if c in train.columns]
        self.cols_to_drop   = [c for c in self.cols_to_drop if c not in self.cols_important]

        print(f'  Colonnes supprimees   (> 50%)  : {len(self.cols_to_drop)}')
        print(f'  Flag + imputer        (20-50%) : {len(self.cols_to_flag)}')
        print(f'  Imputer only          (< 20%)  : {len(self.cols_to_impute)}')
        print(f'  Colonnes importantes sauvees    : {len(self.cols_important)}')

    def _create_flags(self, train, test):
        """Cree les colonnes binaires _missing."""
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

        print(f'  Flags crees : {len(flags_train)}')
        return train, test

    def _impute_all_remaining(self, train, test):
        """
        Imputation universelle — corrige TOUS les NaN.
        Numeriques  → mediane du train
        Categoriels → mode du train
        Applique sur train ET test — toutes les colonnes.
        """
        num_dtypes = [np.float64, np.int64, np.float32, np.int32,
                      np.float16, np.int16, np.int8]
        fixed = 0

        # Parcourir TOUTES les colonnes communes train et test
        all_cols = (set(train.columns) | set(test.columns)) - {'ID', 'TARGET'}

        for col in all_cols:
            # Calculer la valeur de remplacement depuis le train
            if col in train.columns:
                if train[col].dtype in num_dtypes:
                    fill_val = float(train[col].median())
                else:
                    mode     = train[col].mode()
                    fill_val = str(mode[0]) if len(mode) > 0 else 'unknown'
            else:
                fill_val = 0

            # Appliquer sur train si NaN
            if col in train.columns and train[col].isnull().any():
                train[col] = train[col].fillna(fill_val)
                fixed += 1

            # Appliquer sur test TOUJOURS
            if col in test.columns and test[col].isnull().any():
                test[col] = test[col].fillna(fill_val)

        print(f'  Colonnes corrigees : {fixed}')
        print(f'  NaN restants train : {train.isnull().sum().sum()}')
        print(f'  NaN restants test  : {test.isnull().sum().sum()}')
        return train, test

    def handle_missing(self, train, test):
        """Pipeline complet : compute → flags → drop → impute."""
        print('\n========== GESTION DES MANQUANTS ==========')
        self._compute_missing(train)
        train, test = self._create_flags(train, test)
        train.drop(columns=self.cols_to_drop, inplace=True)
        test.drop(columns=self.cols_to_drop,  inplace=True)
        print(f'  {len(self.cols_to_drop)} colonnes supprimees')
        print('  Imputation en cours...')
        train, test = self._impute_all_remaining(train, test)
        print('Manquants termine !')
        return train, test

    # ============================================================
    # ENCODAGE
    # ============================================================

    def _remove_useless_cols(self, train, test):
        """Supprime les colonnes redondantes."""
        cols = [c for c in self.cols_to_remove if c in train.columns]
        train.drop(columns=cols, inplace=True)
        test.drop(columns=cols,  inplace=True)
        print(f'  Colonnes redondantes supprimees : {cols}')
        return train, test

    def _label_encode(self, train, test):
        """Label Encoding sur toutes les colonnes categorielles."""
        # Utilise _get_cat_cols pour compatibilite Pandas 2 et 3
        cat_cols = self._get_cat_cols(train)

        for col in cat_cols:
            le         = LabelEncoder()
            all_values = pd.concat([train[col], test[col]], axis=0).astype(str)
            le.fit(all_values)
            train[col] = le.transform(train[col].astype(str))
            test[col]  = le.transform(test[col].astype(str))
            self.label_encoders[col] = le
            print(f'  Label Encoded : {col} ({len(le.classes_)} valeurs)')

        return train, test

    def encode(self, train, test):
        """Pipeline complet d encodage."""
        print('\n========== ENCODAGE ==========')
        train, test = self._remove_useless_cols(train, test)
        train, test = self._label_encode(train, test)

        remaining  = self._get_cat_cols(train)
        cols_train = set(train.columns)
        cols_test  = set(test.columns)

        if len(remaining) == 0:
            print('  Toutes les colonnes categorielles sont encodees !')
        else:
            print(f'  Colonnes texte restantes : {remaining}')

        if cols_train == cols_test:
            print('  Train et test ont les memes colonnes !')
        else:
            print(f'  Diff train/test : {cols_train - cols_test}')

        print('Encodage termine !')
        return train, test

    # ============================================================
    # NORMALISATION
    # ============================================================

    def normalize(self, train, test):
        """Normalise avec StandardScaler. fit sur train uniquement."""
        print('\n========== NORMALISATION ==========')

        self.feature_cols = [
            c for c in train.columns
            if c not in ['ID', 'TARGET']
            and not c.endswith('_missing')
            and train[c].dtype in [np.float64, np.int64,
                                    np.float32, np.int32]
        ]

        train[self.feature_cols] = self.scaler.fit_transform(train[self.feature_cols])
        test[self.feature_cols]  = self.scaler.transform(test[self.feature_cols])

        print(f'  {len(self.feature_cols)} colonnes normalisees (StandardScaler)')
        print(f'  Moyenne train ≈ {train[self.feature_cols].mean().mean():.6f} (doit etre ≈ 0)')
        print(f'  Std    train  ≈ {train[self.feature_cols].std().mean():.6f}  (doit etre ≈ 1)')
        print('Normalisation terminee !')
        return train, test

    # ============================================================
    # SPLIT TRAIN / VALIDATION
    # ============================================================

    def split(self, train):
        """Split stratifie 80/20."""
        print('\n========== SPLIT TRAIN / VALIDATION ==========')

        feature_cols = [c for c in train.columns if c not in ['ID', 'TARGET']]
        X = train[feature_cols].values
        y = train['TARGET'].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size    = self.test_size,
            random_state = self.random_state,
            stratify     = y
        )

        print(f'  X_train : {X_train.shape}  |  y_train : {y_train.shape}')
        print(f'  X_val   : {X_val.shape}    |  y_val   : {y_val.shape}')
        print(f'  Churn dans train : {y_train.mean()*100:.2f}%')
        print(f'  Churn dans val   : {y_val.mean()*100:.2f}%')
        print('Split termine !')
        return X_train, X_val, y_train, y_val

    # ============================================================
    # FEATURE ENGINEERING
    # ============================================================

    def _feature_engineering(self, df):
        """
        Cree de nouvelles features a partir des colonnes existantes.
        Doit etre appele AVANT handle_missing() pour avoir acces
        aux colonnes originales.
        """

        # ① Age en annees (AGE est en mois dans le dataset)
        if 'AGE' in df.columns:
            df['AGE_YEARS'] = df['AGE'] / 12

        # ② Nombre total de produits utilises
        #    Un client avec beaucoup de produits est plus attache
        prod_cols = [
            'CR_PROD_CNT_CC', 'CR_PROD_CNT_IL',
            'CR_PROD_CNT_PIL', 'CR_PROD_CNT_VCU',
            'CR_PROD_CNT_TOVR', 'CR_PROD_CNT_CCFP'
        ]
        prod_cols = [c for c in prod_cols if c in df.columns]
        if prod_cols:
            df['TOTAL_PRODUCTS'] = df[prod_cols].fillna(0).sum(axis=1)

        # ③ Ratio transactions / solde
        #    Client qui depense beaucoup par rapport a son solde → risque churn
        if 'TURNOVER_CC' in df.columns and 'REST_AVG_CUR' in df.columns:
            df['TRAN_TO_BALANCE'] = (
                df['TURNOVER_CC'] / (df['REST_AVG_CUR'].abs() + 1)
            )

        # ④ Tendance globale des transactions (1 mois)
        #    Transactions en baisse → signal de depart prochain
        trend_1m = [
            'CNT_TRAN_ATM_TENDENCY1M', 'CNT_TRAN_AUT_TENDENCY1M'
        ]
        trend_1m = [c for c in trend_1m if c in df.columns]
        if trend_1m:
            df['TRAN_GLOBAL_TREND_1M'] = df[trend_1m].fillna(0).mean(axis=1)

        # ⑤ Tendance globale des transactions (3 mois)
        trend_3m = [
            'CNT_TRAN_ATM_TENDENCY3M', 'CNT_TRAN_AUT_TENDENCY3M'
        ]
        trend_3m = [c for c in trend_3m if c in df.columns]
        if trend_3m:
            df['TRAN_GLOBAL_TREND_3M'] = df[trend_3m].fillna(0).mean(axis=1)

        # ⑥ Ratio turnover compte courant / salaire
        #    Indique l utilisation du compte par rapport au revenu
        if 'TURNOVER_CC' in df.columns and 'TURNOVER_PAYM' in df.columns:
            df['TURNOVER_RATIO'] = (
                df['TURNOVER_CC'] / (df['TURNOVER_PAYM'] + 1)
            )

        # ⑦ Solde moyen dynamique (evolution)
        if 'REST_DYNAMIC_CUR_3M' in df.columns and 'REST_DYNAMIC_CUR_1M' in df.columns:
            df['BALANCE_EVOLUTION'] = (
                df['REST_DYNAMIC_CUR_3M'] - df['REST_DYNAMIC_CUR_1M']
            )

        # ⑧ Ratio transactions ATM vs total
        if 'TRANS_COUNT_ATM_PRC' in df.columns:
            df['ATM_USAGE'] = df['TRANS_COUNT_ATM_PRC'].fillna(0)

        print(f'  Features creees : AGE_YEARS, TOTAL_PRODUCTS, TRAN_TO_BALANCE,')
        print(f'                    TRAN_GLOBAL_TREND_1M, TRAN_GLOBAL_TREND_3M,')
        print(f'                    TURNOVER_RATIO, BALANCE_EVOLUTION, ATM_USAGE')

        return df

    # ============================================================
    # PIPELINE COMPLET
    # ============================================================

    def fit_transform(self, train, test):
        """
        Pipeline complet :
        1. Feature Engineering
        2. Gestion des manquants
        3. Encodage
        4. Normalisation
        5. Split 80/20
        Retourne : X_train, X_val, y_train, y_val, X_test
        """
        print('\n' + '=' * 50)
        print('       PREPROCESSING COMPLET')
        print('=' * 50)

        # NOUVEAU — Feature Engineering avant tout
        print('\n========== FEATURE ENGINEERING ==========')
        train = self._feature_engineering(train)
        test  = self._feature_engineering(test)
        print(f'  Shape train apres FE : {train.shape}')
        print(f'  Shape test  apres FE : {test.shape}')

        train, test = self.handle_missing(train, test)
        train, test = self.encode(train, test)
        train, test = self.normalize(train, test)

        X_train, X_val, y_train, y_val = self.split(train)

        feature_cols = [c for c in test.columns if c not in ['ID', 'TARGET']]
        X_test       = test[feature_cols].values

        print('\n' + '=' * 50)
        print('PREPROCESSING TERMINE !')
        print(f'   X_train : {X_train.shape}')
        print(f'   X_val   : {X_val.shape}')
        print(f'   X_test  : {X_test.shape}')
        print('=' * 50)

        return X_train, X_val, y_train, y_val, X_test