import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, norm
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint


# Загрузка данных
data = pd.read_excel("data/dataset_final.xlsx")
status = data["Status"]
predictors = data.iloc[:, :-2]

# Разделение данных
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    predictors, status, test_size=0.2, random_state=20, stratify=status
)

group_normal = X_train_full[y_train_full == 0]
group_patology = X_train_full[y_train_full == 1]

# Быстрый отбор значимых на train
def quick_screening(group1, group2, alpha=0.05):
    results = {}
    for feature in group1.columns:
        try:
            _, p = mannwhitneyu(group1[feature].dropna(), group2[feature].dropna())
            results[feature] = p
        except:
            results[feature] = 1.0
    p_values = pd.Series(results)
    _, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    significant = p_values.index[p_corrected < alpha]
    return significant.tolist()

print("Предварительный скрининг (только на train)...")
significant_features = quick_screening(group_normal, group_patology)
print(f"Отобрано {len(significant_features)} значимых признаков из {predictors.shape[1]}")


# Фильтрация по иерархической корреляции на train
def filter_correlated_hierarchical(df, features, group1, group2, threshold=0.4):
    corr_matrix = df[features].corr(method='spearman').abs()
    distance_matrix = 1 - corr_matrix
    condensed_dist = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method='average')
    clusters = fcluster(linkage_matrix, threshold, criterion='distance')
    
    cluster_dict = {}
    for feat, clust in zip(features, clusters):
        cluster_dict.setdefault(clust, []).append(feat)
    
    # Выбор с наименьшим p-value
    kept = []
    for cluster in cluster_dict.values():
        best = None
        best_p = 1.0
        for feat in cluster:
            try:
                _, p = mannwhitneyu(
                    group1[feat].dropna(),
                    group2[feat].dropna()
                )
                if p < best_p:
                    best_p = p
                    best = feat
            except:
                pass
        kept.append(best if best else cluster[0])
    
    # Диагностика
    print(f"Порог расстояния: {threshold}")
    print(f"Всего признаков на входе: {len(features)}")
    print(f"Всего кластеров: {len(cluster_dict)}")
    print(f"Признаков осталось: {len(kept)}")
    
    return kept

filtered_features = filter_correlated_hierarchical(
    X_train_full, significant_features, group_normal, group_patology, threshold=0.2
)
print(f"После корреляционной фильтрации: {len(filtered_features)} признаков")


class EffectSizeTopKSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=20):
        self.k = k
        self.selected_features_ = None
    
    def fit(self, X, y):
        df = pd.DataFrame(X)
        df['target'] = y.values
        group0 = df[df['target'] == 0].drop('target', axis=1)
        group1 = df[df['target'] == 1].drop('target', axis=1)
        
        effect_sizes = {}
        N = len(df)
        for col in group0.columns:
            try:
                _, p = mannwhitneyu(group0[col], group1[col])
                Z = stats.norm.ppf(1 - p/2)
                r = Z / np.sqrt(N)
                effect_sizes[col] = abs(r)
            except:
                effect_sizes[col] = 0
        
        self.selected_features_ = sorted(effect_sizes, key=effect_sizes.get, reverse=True)[:self.k]
        return self
    
    def transform(self, X):
        return X[self.selected_features_]


# Отбор топ-5 по effect size на train
selector = EffectSizeTopKSelector(k=5)
selector.fit(X_train_full[filtered_features], y_train_full)
final_features = selector.selected_features_
print(f"Отобрано топ-{len(final_features)} по effect size: {final_features}")

# Финальная модель на отобранных признаках
X_train = X_train_full[final_features]
X_test = X_test_full[final_features]
y_train = y_train_full
y_test = y_test_full

# Пайплайн с вложенным отбором
pipeline = Pipeline([
    ('selector', EffectSizeTopKSelector(k=2)),
    ('scaler', RobustScaler()),
    ('lr', LogisticRegression(random_state=42))
])

# Внешняя кросс-валидация
print("\nПРАВИЛЬНАЯ ОЦЕНКА (отбор внутри CV)")

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(
    pipeline, X_train, y_train, 
    cv=outer_cv, 
    scoring=['roc_auc', 'accuracy'],
    return_train_score=True,
    return_estimator=True
)

print(f"ROC-AUC (CV): {cv_results['test_roc_auc'].mean():.3f} ± {cv_results['test_roc_auc'].std():.3f}")
print(f"Train ROC-AUC (CV): {cv_results['train_roc_auc'].mean():.3f} ± {cv_results['train_roc_auc'].std():.3f}")
print(f"Разница train/test: {cv_results['train_roc_auc'].mean() - cv_results['test_roc_auc'].mean():.3f}")

# Финальная модель
print("\nФИНАЛЬНАЯ МОДЕЛЬ")

pipeline.fit(X_train, y_train)
final_selected_features = pipeline.named_steps['selector'].selected_features_

print(f"Отобрано признаков: {len(final_selected_features)}")
print(f"Финальные признаки: {final_selected_features}")

# Предсказания
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
print(f"Train ROC-AUC: {roc_auc_score(y_train, pipeline.predict_proba(X_train)[:, 1]):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Коэффициенты
lr_model = pipeline.named_steps['lr']
coef_df = pd.DataFrame({
    'Feature': final_selected_features,
    'Coefficient': lr_model.coef_[0]
})
print("\nКоэффициенты финальной модели:")
print(coef_df)
print(f"Intercept: {lr_model.intercept_[0]:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ROC-кривая
plt.figure(figsize=(6, 6))
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
plt.xlim([-0.05, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Анализ переобучения
train_proba = pipeline.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, train_proba)
print(f"\nTrain ROC-AUC: {train_auc:.3f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
print(f"Разница: {train_auc - roc_auc_score(y_test, y_proba):.3f}")

if train_auc - roc_auc_score(y_test, y_proba) > 0.15:
    print("ВНИМАНИЕ: Возможное переобучение!")
else:
    print("Переобучение в пределах нормы ✓")

# График effect sizes (на train данных)
effect_sizes = {}
N = len(X_train_full)

for col in filtered_features:
    try:
        g0 = X_train_full[y_train_full == 0][col].dropna()
        g1 = X_train_full[y_train_full == 1][col].dropna()
        _, p = mannwhitneyu(g0, g1)
        Z = stats.norm.ppf(1 - p/2)
        r = Z / np.sqrt(N)
        if np.isinf(r) or r > 1.0:
            r = 1.0
        effect_sizes[col] = r
    except:
        effect_sizes[col] = 0

# Топ-20
df_plot = pd.DataFrame({
    'Feature': filtered_features,
    'EffectSize': [effect_sizes.get(f, 0) for f in filtered_features]
})
df_plot['AbsEffect'] = df_plot['EffectSize'].abs()
df_plot = df_plot.nlargest(20, 'AbsEffect')
df_plot = df_plot.sort_values('EffectSize', ascending=False)

# Цвета
final_features_set = set(final_selected_features)
df_plot['Color'] = ['#2ca02c' if f in final_features_set else '#d62728' for f in df_plot['Feature']]

# Построение
plt.figure(figsize=(12, 8))
bars = plt.barh(df_plot['Feature'], df_plot['EffectSize'], color=df_plot['Color'], 
                edgecolor='black', linewidth=0.5)

plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Effect Size (r)', fontsize=12)
plt.title('Top-20 Radiomic Features by Effect Size', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

print("\ndf_plot первые 5 строк:")
print(df_plot[['Feature', 'EffectSize', 'Color']].head())
