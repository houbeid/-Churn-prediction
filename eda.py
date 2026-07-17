import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='husl')
plt.rcParams['figure.figsize'] = (12, 6)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# Créer le dossier figures s'il n'existe pas
os.makedirs('figures', exist_ok=True)

print(' Imports OK')

# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================
train = pd.read_csv('data/bank_data_train.csv')
test  = pd.read_csv('data/bank_data_test.csv')

print(f'\nTrain : {train.shape[0]} lignes, {train.shape[1]} colonnes')
print(f'Test  : {test.shape[0]} lignes,  {test.shape[1]} colonnes')

# ============================================================
# 2. APERÇU GÉNÉRAL
# ============================================================
print('\n--- Premières lignes ---')
print(train.head())

print('\n--- Types + valeurs manquantes ---')
print(train.info())

print('\n--- Statistiques descriptives ---')
print(train.describe().T)

# ============================================================
# 3. DISTRIBUTION DE LA TARGET (churn)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

counts = train['TARGET'].value_counts()
axes[0].bar(['Reste (0)', 'Churn (1)'], counts.values,
            color=['#2ecc71', '#e74c3c'], edgecolor='black')
axes[0].set_title('Distribution de la TARGET', fontsize=14)
axes[0].set_ylabel('Nombre de clients')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

pcts = train['TARGET'].value_counts(normalize=True) * 100
axes[1].pie(pcts.values, labels=['Reste (0)', 'Churn (1)'],
            autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
            startangle=90, explode=(0, 0.05))
axes[1].set_title('Proportion du churn', fontsize=14)

plt.tight_layout()
plt.savefig('figures/01_target_distribution.png', dpi=80, bbox_inches='tight')
plt.close()

print(f'\nClasse 0 (reste) : {counts[0]} ({pcts[0]:.2f}%)')
print(f'Classe 1 (churn) : {counts[1]} ({pcts[1]:.2f}%)')
print(f'\n  Ratio déséquilibre : 1:{counts[0]//counts[1]}')

# ============================================================
# 4. VALEURS MANQUANTES
# ============================================================
missing     = train.isnull().sum()
missing_pct = (missing / len(train)) * 100
missing_df  = pd.DataFrame({
    'Colonne'         : missing.index,
    'Manquants'       : missing.values,
    'Pourcentage (%)' : missing_pct.values
}).sort_values('Pourcentage (%)', ascending=False)

missing_df = missing_df[missing_df['Manquants'] > 0]
print(f'\nColonnes avec valeurs manquantes : {len(missing_df)}')
print(missing_df.head(20).to_string())

# Répartition par seuil
sup_50 = missing_pct[missing_pct > 50]
entre  = missing_pct[(missing_pct > 20) & (missing_pct <= 50)]
inf_20 = missing_pct[(missing_pct > 0)  & (missing_pct <= 20)]
print(f'\n À supprimer  (> 50%)  : {len(sup_50)} colonnes')
print(f' Flag+imputer (20-50%) : {len(entre)} colonnes')
print(f' Imputer only (< 20%)  : {len(inf_20)} colonnes')

# Visualisation
if len(missing_df) > 0:
    top_missing = missing_df.head(30)
    plt.figure(figsize=(14, 8))
    plt.barh(top_missing['Colonne'], top_missing['Pourcentage (%)'],
             color=plt.cm.RdYlGn_r(top_missing['Pourcentage (%)'] / 100))
    plt.axvline(x=50, color='red',    linestyle='--', label='Seuil 50%')
    plt.axvline(x=20, color='orange', linestyle='--', label='Seuil 20%')
    plt.xlabel('Pourcentage de valeurs manquantes (%)')
    plt.title('Top colonnes avec valeurs manquantes', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/02_missing_values.png', dpi=80, bbox_inches='tight')
    plt.close()

# ============================================================
# 5. TYPES DE COLONNES
# ============================================================
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train.select_dtypes(include=['object', 'str']).columns.tolist()
num_cols = [c for c in num_cols if c not in ['ID', 'TARGET']]
cat_cols = [c for c in cat_cols if c not in ['ID', 'TARGET']]

print(f'\nColonnes numériques    : {len(num_cols)}')
print(f'Colonnes catégorielles : {len(cat_cols)}')
print(f'Catégorielles : {cat_cols}')

# ============================================================
# 6. DISTRIBUTION DES VARIABLES NUMÉRIQUES (histogrammes)
# ============================================================
cols_to_plot = num_cols[:20]
n_cols_plot  = 4
n_rows_plot  = (len(cols_to_plot) + n_cols_plot - 1) // n_cols_plot

fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                         figsize=(16, n_rows_plot * 3))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot):
    axes[i].hist(train[col].dropna(), bins=20,
                 color='steelblue', edgecolor='white', alpha=0.8)
    axes[i].set_title(col, fontsize=10)

for j in range(len(cols_to_plot), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Distribution des variables numériques (20 premières)',
             fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('figures/03_histograms.png', dpi=80, bbox_inches='tight')
plt.close()

# ============================================================
# 7. BOXPLOTS — DÉTECTION DES OUTLIERS
# ============================================================
fig, axes = plt.subplots(n_rows_plot, n_cols_plot,
                         figsize=(16, n_rows_plot * 3))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot):
    axes[i].boxplot(train[col].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='navy'),
                    medianprops=dict(color='red', linewidth=2))
    axes[i].set_title(col, fontsize=10)

for j in range(len(cols_to_plot), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Boxplots — Détection des outliers (20 premières)',
             fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('figures/04_boxplots.png', dpi=80, bbox_inches='tight')
plt.close()

# ============================================================
# 8. CORRÉLATION AVEC LA TARGET
# ============================================================
corr_target = train[num_cols + ['TARGET']].corr()['TARGET'].drop('TARGET')
corr_target = corr_target.abs().sort_values(ascending=False)

plt.figure(figsize=(14, 8))
colors = ['#e74c3c' if v > 0 else '#3498db'
          for v in train[num_cols + ['TARGET']].corr()['TARGET']
          .drop('TARGET').loc[corr_target.index]]
plt.barh(corr_target.index[:30], corr_target.values[:30], color=colors[:30])
plt.xlabel('Corrélation absolue avec TARGET')
plt.title('Top 30 features les plus corrélées avec le churn', fontsize=14)
plt.tight_layout()
plt.savefig('figures/05_correlation_target.png', dpi=80, bbox_inches='tight')
plt.close()

# ============================================================
# 9. HEATMAP DE CORRÉLATION
# ============================================================
top_features  = corr_target.index[:20].tolist() + ['TARGET']
corr_matrix   = train[top_features].corr()

plt.figure(figsize=(14, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Heatmap de corrélation — Top 20 features', fontsize=14)
plt.tight_layout()
plt.savefig('figures/06_heatmap.png', dpi=80, bbox_inches='tight')
plt.close()

# ============================================================
# 10. VARIANCE DES FEATURES
# ============================================================
variance     = train[num_cols].var().sort_values(ascending=False)
low_variance = variance[variance < 0.01]

print(f'\nFeatures à variance quasi-nulle (< 0.01) : {len(low_variance)}')
print(low_variance)

plt.figure(figsize=(14, 6))
plt.plot(range(len(variance)), variance.values,
         color='steelblue', linewidth=1.5)
plt.axhline(y=0.01, color='red', linestyle='--', label='Seuil 0.01')
plt.yscale('log')
plt.xlabel('Features (triées par variance)')
plt.ylabel('Variance (échelle log)')
plt.title('Variance de toutes les features numériques', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('figures/07_variance.png', dpi=80, bbox_inches='tight')
plt.close()

# ============================================================
# 11. VARIABLES CATÉGORIELLES
# ============================================================
if len(cat_cols) > 0:
    n_rows_cat = (len(cat_cols) + 2) // 3
    fig, axes  = plt.subplots(n_rows_cat, 3,
                               figsize=(15, n_rows_cat * 4))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        # IMPORTANT : limiter à TOP 15 valeurs maximum
        # → évite de tracer 19 588 barres pour CLNT_JOB_POSITION
        n_unique      = train[col].nunique()
        value_counts  = train[col].value_counts().head(15)

        axes[i].bar(value_counts.index.astype(str), value_counts.values,
                    color='coral', edgecolor='black')

        # Titre indique si on affiche seulement le top 15
        if n_unique > 15:
            axes[i].set_title(
                f'{col}\n({n_unique} valeurs — top 15 affichées)', fontsize=9)
        else:
            axes[i].set_title(
                f'{col} ({n_unique} valeurs)', fontsize=10)

        axes[i].tick_params(axis='x', rotation=45)

    for j in range(len(cat_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Distribution des variables catégorielles',
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/08_categorical.png', dpi=80, bbox_inches='tight')
    plt.close()

# ============================================================
# 12. FEATURES vs TARGET (churn vs non-churn)
# ============================================================
top12 = corr_target.index[:12].tolist()

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(top12):
    churn_0 = train[train['TARGET'] == 0][col].dropna()
    churn_1 = train[train['TARGET'] == 1][col].dropna()
    axes[i].hist(churn_0, bins=20, alpha=0.6,
                 label='Reste (0)', color='#2ecc71', density=True)
    axes[i].hist(churn_1, bins=20, alpha=0.6,
                 label='Churn (1)', color='#e74c3c', density=True)
    axes[i].set_title(col, fontsize=10)
    axes[i].legend(fontsize=8)

plt.suptitle('Distribution des top features par classe TARGET',
             fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('figures/09_features_vs_target.png', dpi=80, bbox_inches='tight')
plt.close()

# ============================================================
# 13. CORRÉLATION DES FLAGS AVEC LA TARGET
# ============================================================
print('\n--- Corrélation des flags manquants avec TARGET ---')
cols_sup50 = missing_pct[missing_pct > 50].index.tolist()

for col in cols_sup50:
    if col in train.columns:
        flag = train[col].isnull().astype(int)
        corr = flag.corr(train['TARGET'])
        if abs(corr) > 0.05:
            print(f'  {col} — corrélation flag/TARGET : {corr:.4f}')

# ============================================================
# 14. RÉSUMÉ FINAL
# ============================================================
print('\n' + '=' * 60)
print('           RÉSUMÉ DE L\'ANALYSE EXPLORATOIRE')
print('=' * 60)
print(f'\n Dataset train : {train.shape[0]} lignes x {train.shape[1]} colonnes')
print(f' Dataset test  : {test.shape[0]} lignes x {test.shape[1]} colonnes')
print(f'\n Classe 0 (reste) : {counts[0]} ({pcts[0]:.1f}%)')
print(f' Classe 1 (churn) : {counts[1]} ({pcts[1]:.1f}%)')
print(f'\n Features numériques    : {len(num_cols)}')
print(f'  Features catégorielles : {len(cat_cols)}')
print(f'\n Colonnes avec manquants : {len(missing_df)}')
print(f' Features à faible variance : {len(low_variance)}')
print(f'\n Top 5 features corrélées avec TARGET :')
for feat, val in corr_target.head(5).items():
    print(f'   - {feat} : {val:.4f}')
print('\n' + '=' * 60)
print(' EDA terminée — prêt pour le preprocessing')
print('=' * 60)