import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, shapiro
from sklearn.model_selection import (train_test_split, StratifiedKFold, 
                                     cross_val_score, cross_validate)
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import random
import textwrap
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm


# Загрузка данных
data = pd.read_excel(r"C:\Users\artur\OneDrive\Рабочий стол\Работа\Радиомика\Подагра\datasets\dataset_final.xlsx")
status = data["Status"]
predictors = data.iloc[:, :-2]

# Сначала разделяем данные, ПОТОМ отбираем признаки
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    predictors, status, test_size=0.2, random_state=20, stratify=status
)

group_normal = X_train_full[y_train_full == 0]
group_patology = X_train_full[y_train_full == 1]

# Быстрый отбор значимых ТОЛЬКО на train
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


# Фильтрация по иерархической корреляции ТОЛЬКО на train
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


# Отбор топ-5 по effect size ТОЛЬКО на train
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



from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm

# ============ РАСЧЁТ ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ ДЛЯ ВСЕХ МЕТРИК ============

# Из confusion matrix: [[12, 3], [0, 15]]
# TN=12, FP=3, FN=0, TP=15

# 1. Чувствительность (Sensitivity) = TP/(TP+FN) = 15/15 = 1.00
se_ci = proportion_confint(count=15, nobs=15, alpha=0.05, method='wilson')
print(f"Sensitivity: {15/15:.2f} (95% CI: {se_ci[0]:.2f}-{se_ci[1]:.2f})")

# 2. Специфичность (Specificity) = TN/(TN+FP) = 12/15 = 0.80
sp_ci = proportion_confint(count=12, nobs=15, alpha=0.05, method='wilson')
print(f"Specificity: {12/15:.2f} (95% CI: {sp_ci[0]:.2f}-{sp_ci[1]:.2f})")

# 3. Точность (Accuracy) = (TP+TN)/(TP+TN+FP+FN) = 27/30 = 0.90
acc_ci = proportion_confint(count=27, nobs=30, alpha=0.05, method='wilson')
print(f"Accuracy: {27/30:.2f} (95% CI: {acc_ci[0]:.2f}-{acc_ci[1]:.2f})")

# 4. ROC-AUC тестовый — метод Хэнли-МакНила
def auc_ci_hanley_mcneil(y_true, y_proba, alpha=0.05):
    """Доверительный интервал для AUC по методу Хэнли-МакНила"""
    auc_value = roc_auc_score(y_true, y_proba)
    n1 = np.sum(y_true == 1)  # positives
    n0 = np.sum(y_true == 0)  # negatives
    
    q1 = auc_value / (2 - auc_value)
    q2 = 2 * auc_value**2 / (1 + auc_value)
    se = np.sqrt((auc_value * (1 - auc_value) + 
                  (n1 - 1) * (q1 - auc_value**2) + 
                  (n0 - 1) * (q2 - auc_value**2)) / (n1 * n0))
    
    z = norm.ppf(1 - alpha/2)
    ci_lower = auc_value - z * se
    ci_upper = auc_value + z * se
    
    return auc_value, ci_lower, ci_upper

test_auc, test_ci_low, test_ci_high = auc_ci_hanley_mcneil(y_test, y_proba)
print(f"Test ROC-AUC: {test_auc:.3f} (95% CI: {test_ci_low:.3f}-{test_ci_high:.3f})")

# 5. ROC-AUC тренировочный
train_proba = pipeline.predict_proba(X_train)[:, 1]
train_auc, train_ci_low, train_ci_high = auc_ci_hanley_mcneil(y_train, train_proba)
print(f"Train ROC-AUC: {train_auc:.3f} (95% CI: {train_ci_low:.3f}-{train_ci_high:.3f})")

# 6. ROC-AUC кросс-валидационный (mean ± std)
cv_mean = cv_results['test_roc_auc'].mean()
cv_std = cv_results['test_roc_auc'].std()
print(f"CV ROC-AUC: {cv_mean:.3f} ± {cv_std:.3f}")


# ============ ИТОГОВАЯ ТАБЛИЦА ДЛЯ СТАТЬИ ============
print("\n" + "="*60)
print("ТАБЛИЦА ДЛЯ СТАТЬИ")
print("="*60)
print(f"{'Метрика':<25}{'Значение':<12}{'95% ДИ'}")
print("-"*60)
print(f"{'Se (Sensitivity)':<25}{15/15:.2f}{'':8}{se_ci[0]:.2f}-{se_ci[1]:.2f}")
print(f"{'Sp (Specificity)':<25}{12/15:.2f}{'':8}{sp_ci[0]:.2f}-{sp_ci[1]:.2f}")
print(f"{'Accuracy':<25}{27/30:.2f}{'':8}{acc_ci[0]:.2f}-{acc_ci[1]:.2f}")
print(f"{'Train ROC-AUC':<25}{train_auc:.3f}{'':8}{train_ci_low:.3f}-{train_ci_high:.3f}")
print(f"{'Test ROC-AUC':<25}{test_auc:.3f}{'':8}{test_ci_low:.3f}-{test_ci_high:.3f}")
print(f"{'CV ROC-AUC':<25}{cv_mean:.3f}{'':8}{cv_mean-cv_std:.3f}-{cv_mean+cv_std:.3f}")
print("="*60)



'''# Описательные статистики
final_features = [
    'original_firstorder_RootMeanSquared',
    'wavelet-LLL_firstorder_Median'
]

results = []

for feature in final_features:
    # Данные по груупам
    norm = data[data['Status'] == 0][feature].dropna()
    path = data[data['Status'] == 1][feature].dropna()
    
    # Медиана (Q1-Q3)
    norm_median = norm.median()
    norm_q1 = norm.quantile(0.25)
    norm_q3 = norm.quantile(0.75)
    
    path_median = path.median()
    path_q1 = path.quantile(0.25)
    path_q3 = path.quantile(0.75)
    
    # Shapiro-Wilk тест на нормальность
    _, norm_shapiro = shapiro(norm) if len(norm) >= 3 else (None, None)
    _, path_shapiro = shapiro(path) if len(path) >= 3 else (None, None)
    
    # U-тест Манна-Уитни
    _, p_val = mannwhitneyu(norm, path, alternative='two-sided')
    
    # Форматирование p-value
    if p_val < 0.001:
        p_text = '<0.001'
    else:
        p_text = f'{p_val:.3f}'.lstrip('0')
    
    results.append({
        'Feature': feature,
        'Normal (median [Q1; Q3])': f"{norm_median:.2f} [{norm_q1:.2f}; {norm_q3:.2f}]",
        'Pathology (median [Q1; Q3])': f"{path_median:.2f} [{path_q1:.2f}; {path_q3:.2f}]",
        'p (Mann-Whitney)': p_text,
        'Shapiro-Wilk (Normal)': f"{norm_shapiro:.3f}" if norm_shapiro else '—',
        'Shapiro-Wilk (Pathology)': f"{path_shapiro:.3f}" if path_shapiro else '—'
    })

df_stats = pd.DataFrame(results)
df_stats.to_csv('descriptive_stats_final.csv', index=False, encoding='utf-8-sig')




final_features = [
    'original_firstorder_RootMeanSquared',
    'wavelet-LLL_firstorder_Median'
]

# Упрощённые названия для публикации
name_map = {
    'original_firstorder_RootMeanSquared': 'Среднеквадратичное значение (RMS)',
    'wavelet-LLL_firstorder_Median': 'Медиана (wavelet-LLL)'
}

# Размеры для публикации
width_mm_1col = 85   # 1 колонка
width_mm_2col = 180  # 2 колонки
height_mm = 100      
mm_to_inch = 1 / 25.4

def create_boxplots(final_features, data, figsize_type='large'):
    """Создаёт boxplots для указанных признаков"""
    
    if figsize_type == 'large':
        params = {
            'ncols': 2,
            'nrows': 1,
            'figsize': (width_mm_2col * mm_to_inch, height_mm * mm_to_inch),
            'title_fontsize': 11,
            'tick_fontsize': 9,
            'label_fontsize': 9,
            'wrap_width': 25,
            'pad': 0.8,
            'h_pad': 1.5,
            'w_pad': 1.8,
            'box_width': 0.6,
            'title_pad': 8
        }
    else:
        params = {
            'ncols': 1,
            'nrows': 2,
            'figsize': (width_mm_1col * mm_to_inch, height_mm * mm_to_inch),
            'title_fontsize': 9,
            'tick_fontsize': 8,
            'label_fontsize': 8,
            'wrap_width': 20,
            'pad': 0.5,
            'h_pad': 1.2,
            'w_pad': 1.0,
            'box_width': 0.6,
            'title_pad': 6
        }
    
    fig, axes = plt.subplots(params['nrows'], params['ncols'], 
                             figsize=params['figsize'])
    
    if params['nrows'] * params['ncols'] == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    
    for i, feature in enumerate(final_features):
        ax = axes[i]
        
        # Boxplot
        sns.boxplot(x='Status', y=feature, data=data, ax=ax,
                   width=params['box_width'],
                   palette=['#78d7e2', '#ff2400'],
                   fliersize=2)

        if feature == 'original_firstorder_RootMeanSquared':
             ax.set_ylim(0, 230)

        if feature == 'wavelet-LLL_firstorder_Median':
             ax.set_ylim(-100, 600)
        
        # Красивое название
        simple_name = name_map.get(feature, feature.split('_')[-1])
        simple_name = '\n'.join(textwrap.wrap(simple_name, params['wrap_width']))
        
        # U-тест Манна-Уитни
        stat, p_val = stats.mannwhitneyu(
            data[data['Status'] == 0][feature].dropna(),
            data[data['Status'] == 1][feature].dropna(),
            alternative='two-sided'
        )
        
        # Форматирование p-value
        if p_val < 0.001:
            p_text = 'p < 0.001'
        elif p_val < 0.01:
            p_text = f'p = {p_val:.3f}'.replace('0.', '.')
        else:
            p_text = f'p = {p_val:.3f}'
        
        ax.set_title(f'{simple_name}\n({p_text})', 
                    fontsize=params['title_fontsize'],
                    pad=params['title_pad'])
        
        # РУССКИЕ ПОДПИСИ
        ax.set_xticklabels(['Норма', 'Патология'], 
                          fontsize=params['tick_fontsize'])
        ax.set_ylabel('Значение', fontsize=params['label_fontsize'])
        ax.set_xlabel('')
        ax.tick_params(axis='y', labelsize=params['tick_fontsize'])
    
    # Скрываем лишние подграфики
    for j in range(len(final_features), params['nrows'] * params['ncols']):
        axes[j].axis('off')
    
    plt.tight_layout(pad=params['pad'], h_pad=params['h_pad'], w_pad=params['w_pad'])
    return fig

# Убедимся, что Status — это 0=Норма, 1=Патология
if 'Status' not in data.columns:
    data['Status'] = status.values

# Генерация графиков
print("Создание boxplots для 2-колоночного формата...")
fig_large = create_boxplots(final_features, data, figsize_type='large')
fig_large.savefig('boxplots_two_column_new.png', dpi=300, bbox_inches='tight', facecolor='white')
#fig_large.savefig('boxplots_two_column.pdf', format='pdf', bbox_inches='tight')
plt.close(fig_large)

print("Создание boxplots для 1-колоночного формата...")
fig_small = create_boxplots(final_features, data, figsize_type='small')
fig_small.savefig('boxplots_one_column_new.png', dpi=300, bbox_inches='tight', facecolor='white')
#fig_small.savefig('boxplots_one_column.pdf', format='pdf', bbox_inches='tight')
plt.close(fig_small)

print("Готово!")'''