`remove unnecessary features`

1. **remove columns with highly missing values and imbalanced categories**
   - CNT_CHILDREN - zeros >>> ones - removable
   - AMT_INCOME_TOTAL - skewed
   - OWN_CAR_AGE - 66% - missing - remove
   - FLAG_MOBIL - zeros - remove
   - FLAG_EMP_PHONE - zeros >>> ones - removable
   - FLAG_WORK_PHONE - zeros >>> ones - removable
   - FLAG_CONT_MOBILE - skewed - ones - remove
   - FLAG_PHONE - zeros >> ones
   - FLAG_EMAIL - zeros >>> ones - removable
   - REG_REGION_NOT_LIVE_REGION - zeros - remove
   - REG_REGION_NOT_WORK_REGION - zeros >>> ones - removable
   - LIVE_REGION_NOT_WORK_REGION - zeros >>> ones - removable
   - REG_CITY_NOT_LIVE_CITY - zeros >>> ones - removable
   - REG_CITY_NOT_WORK_CITY - zeros >> ones
   - LIVE_CITY_NOT_WORK_CITY - zeros >> ones
   - EXT_SOURCE_1 - 56% - missing - remove
   - EXT_SOURCE_3 - 20% - missing - removable
   - APARTMENTS_AVG - 51% - missing - remove
   - BASEMENTAREA_AVG - 59% - missing -remove
   - YEARS_BEGINEXPLUATATION_AVG - 49% - missing - remove
   - YEARS_BUILD_AVG - 67% - missing - remove
   - COMMONAREA_AVG - 70% - missing - remove
   - ELEVATORS_AVG - 53% missing - remove
   - ENTRANCES_AVG - 50% - missing - remove
   - FLOORSMAX_AVG -50% - missing - remove
   - FLOORSMIN_AVG - 68% - missing - remove
   - LANDAREA_AVG - 59% - missing - remove
   - LIVINGAPARTMENTS_AVG - 68% - missing - remove
   - LIVINGAREA_AVG - 50% - missing - remove
   - NONLIVINGAPARTMENTS_AVG - 69% - missing - remove
   - NONLIVINGAREA_AVG - 55% - missing - remove
   - APARTMENTS_MODE - 51% - missing - remove
   - BASEMENTAREA_MODE - 59% - missing - remove
   - YEARS_BEGINEXPLUATATION_MODE - 49% - missing - remove
   - YEARS_BUILD_MODE - 67% - missing - remove
   - COMMONAREA_MODE - 70% - missing - remove
   - ELEVATORS_MODE - 53% - missing - remove
   - ENTRANCES_MODE - 50% - missing - remove
   - FLOORSMAX_MODE - 50% - missing - remove
   - FLOORSMIN_MODE - 68% - missing - remove
   - LANDAREA_MODE - 59% - missing - remove
   - LIVINGAPARTMENTS_MODE - 68% - missing - remove
   - LIVINGAREA_MODE - 50% - missing - remove
   - NONLIVINGAPARTMENTS_MODE - 69% - missing - remove
   - NONLIVINGAREA_MODE - 55% - missing - remove
   - APARTMENTS_MEDI - 50% - missing - remove
   - BASEMENTAREA_MEDI - 59% - missing - remove
   - YEARS_BEGINEXPLUATATION_MEDI - 49% - missing - remove
   - YEARS_BUILD_MEDI - 67% - missing - remove
   - COMMONAREA_MEDI - 70% - missing - remove
   - ELEVATORS_MEDI - 53% - missing - remove
   - ENTRANCES_MEDI - 50% - missing - remove
   - FLOORSMAX_MEDI - 50% - missing - remove
   - FLOORSMIN_MEDI - 68% - missing - remove
   - LANDAREA_MEDI - 59% - missing - remove
   - LIVINGAPARTMENTS_MEDI - 68% - missing - remove
   - LIVINGAREA_MEDI - 50% - missing - remove
   - NONLIVINGAPARTMENTS_MEDI - 69% - missing - remove
   - NONLIVINGAREA_MEDI - 55% - missing - remove
   - FONDKAPREMONT_MODE - categorical - 68% - missing - remove
   - HOUSETYPE_MODE - categorical - 50% - missing - remove
   - TOTALAREA_MODE - 48% - missing - remove
   - WALLSMATERIAL_MODE - categorical - 51% - missing - remove
   - EMERGENCYSTATE_MODE - categorical - 47% - missing - remove
   - OBS_30_CNT_SOCIAL_CIRCLE - zeros >> ones
   - DEF_30_CNT_SOCIAL_CIRCLE - zeros >>> ones - removable
   - OBS_60_CNT_SOCIAL_CIRCLE - zeros >> others
   - DEF_60_CNT_SOCIAL_CIRCLE - zeros >>> others
   - FLAG_DOCUMENT_2 - zeros - remove
   - FLAG_DOCUMENT_3 - ones >> zeros
   - FLAG_DOCUMENT_4 - zeros - remove
   - FLAG_DOCUMENT_5 - zeros >>> ones - removable - remove
   - FLAG_DOCUMENT_6 - zeros >>> ones - removable
   - FLAG_DOCUMENT_7 - zeros - remove
   - FLAG_DOCUMENT_8 - zeros >>> ones - removable
   - FLAG_DOCUMENT_9 - zeros - remove
   - FLAG_DOCUMENT_10 - zeros - remove
   - FLAG_DOCUMENT_11 - zeros - remove
   - FLAG_DOCUMENT_12 - zeros - remove
   - FLAG_DOCUMENT_13 - zeros - remove
   - FLAG_DOCUMENT_14 - zeros - remove
   - FLAG_DOCUMENT_15 - zeros - remove
   - FLAG_DOCUMENT_16 - zeros - remove
   - FLAG_DOCUMENT_17 - zeros - remove
   - FLAG_DOCUMENT_18 - zeros - remove
   - FLAG_DOCUMENT_19 - zeros - remove
   - FLAG_DOCUMENT_20 - zeros - remove
   - FLAG_DOCUMENT_21 - zeros - remove
   - AMT_REQ_CREDIT_BUREAU_HOUR - missing 13% + zeros 86% - remove
   - AMT_REQ_CREDIT_BUREAU_DAY - missing 13% + zeros 86% - remove
   - AMT_REQ_CREDIT_BUREAU_WEEK - missing 13% + zeros 86% - remove
   - AMT_REQ_CREDIT_BUREAU_MON - missing 13% + zeros 72%, ones 10% - removable
   - AMT_REQ_CREDIT_BUREAU_QRT - missing 13% + zeros 70%, ones 11% - removable
1. **reduce the features based on the correlation**
   - print **correlation matrix**
   - if two features are highly correlated, remove the one which is less correlated with the target
   - apply a threshould to feature correlations with target and filter

`train test split`

`apply imputation techniques`

- categorical -> missing -> **median imputing(small no. of missing values)**, constant imputation, KNN or MICE(having relations between features), **place holder(may include bias with new category)**
- skewed columns -> missing -> median replacement
- normal distributed -> missing -> mean replacement
- correlated features -> missing -> KNNImputation (use **correlation matrix above printed**)

`encoding`

- replace columns with cardinality = 2 with **binary 0, 1**
- do ordinal encoding or binary encoding if there is a order of small no. of data. otherwise big numbers may add bias to the model
- **one-hot encoding** is good if no. of features are very small

`over/under sampling`

`scale the data`

`train and test the model`

- XGBoost as the model
- use classification report, confussion matrix for evaluations

### Imputing Techniques

1. **Mode Imputation**

   Replace missing values with the most frequent category (mode).

   **Pros:**

   - Simple and straightforward.
   - Works well if the missing values are not many.

   **Cons:**

   - Can introduce bias if the mode is not representative of the missing data.

   ```python
   from sklearn.impute import SimpleImputer

   mode_imputer = SimpleImputer(strategy='most_frequent')
   X_train['categorical_feature'] = mode_imputer.fit_transform(X_train[['categorical_feature']])
   X_test['categorical_feature'] = mode_imputer.transform(X_test[['categorical_feature']])
   ```

2. **Constant Imputation**

   Replace missing values with a new category such as "missing" or "unknown".

   **Pros:**

   - Simple and easy to implement.
   - Maintains the fact that these values were originally missing.

   **Cons:**

   - May introduce an artificial category that might not be meaningful.

   ```python
   constant_imputer = SimpleImputer(strategy='constant', fill_value='missing')
   X_train['categorical_feature'] = constant_imputer.fit_transform(X_train[['categorical_feature']])
   X_test['categorical_feature'] = constant_imputer.transform(X_test[['categorical_feature']])
   ```

3. **Frequent Category Imputation**

   If there are several missing values and using the mode might introduce significant bias, consider filling missing values with one of the most frequent categories.

   ```python
   from collections import Counter

   # Find the most common value, excluding NaNs
   most_common = X_train['categorical_feature'].mode()[0]

   X_train['categorical_feature'].fillna(most_common, inplace=True)
   X_test['categorical_feature'].fillna(most_common, inplace=True)
   ```

4. **Imputation Using Model-Based Techniques**

   Use more advanced techniques like the K-Nearest Neighbors (KNN) or Multivariate Imputation by Chained Equations (MICE) that consider the relationships between features.

   **KNN Imputer:**

   ```python
   from sklearn.impute import KNNImputer

   knn_imputer = KNNImputer(n_neighbors=5)
   X_train_imputed = knn_imputer.fit_transform(X_train)
   X_test_imputed = knn_imputer.transform(X_test)
   ```

   **MICE (Iterative Imputer):**

   ```python
   from sklearn.experimental import enable_iterative_imputer
   from sklearn.impute import IterativeImputer

   mice_imputer = IterativeImputer()
   X_train_imputed = mice_imputer.fit_transform(X_train)
   X_test_imputed = mice_imputer.transform(X_test)
   ```

5. **Using a Placeholder**

   If the missing values have a specific reason, you can replace them with a placeholder that indicates why the value is missing (e.g., "not applicable").

   ```python
   X_train['categorical_feature'].fillna('not_applicable', inplace=True)
   X_test['categorical_feature'].fillna('not_applicable', inplace=True)
   ```

### Over-Sampling Techniques

1. **Random Over-Sampling**
   This technique involves randomly duplicating examples from the minority class to increase its size.

   **Pros:**

   - Simple to implement.
   - Helps in balancing the class distribution.

   **Cons:**

   - Can lead to overfitting due to repeated examples.

   ```python
   from imblearn.over_sampling import RandomOverSampler

   ros = RandomOverSampler(random_state=42)
   X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
   ```

2. **SMOTE (Synthetic Minority Over-sampling Technique)**
   SMOTE generates synthetic samples for the minority class by interpolating between existing minority class examples.

   **Pros:**

   - Reduces the risk of overfitting compared to random over-sampling.
   - Creates more diverse synthetic examples.

   **Cons:**

   - Can create ambiguous samples near class boundaries.

   ```python
   from imblearn.over_sampling import SMOTE

   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   ```

3. **ADASYN (Adaptive Synthetic Sampling Approach)**
   Similar to SMOTE, ADASYN generates synthetic samples but focuses more on difficult-to-classify examples.

   **Pros:**

   - Targets hard-to-classify examples, potentially improving model performance.

   **Cons:**

   - More complex to implement.
   - Can introduce noise.

   ```python
   from imblearn.over_sampling import ADASYN

   adasyn = ADASYN(random_state=42)
   X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
   ```

### Under-Sampling Techniques

1. **Random Under-Sampling**
   This technique involves randomly removing examples from the majority class to reduce its size.

   **Pros:**

   - Simple to implement.
   - Reduces the dataset size, making it faster to process.

   **Cons:**

   - Can lead to loss of valuable information.
   - Potentially removes important examples from the majority class.

   ```python
   from imblearn.under_sampling import RandomUnderSampler

   rus = RandomUnderSampler(random_state=42)
   X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
   ```

2. **Tomek Links**
   This technique identifies and removes pairs of nearest neighbors from different classes (Tomek links) to make the decision boundary clearer.

   **Pros:**

   - Removes noisy examples near the decision boundary.
   - Cleans the dataset by eliminating ambiguous examples.

   **Cons:**

   - May not significantly reduce class imbalance.

   ```python
   from imblearn.under_sampling import TomekLinks

   tomek = TomekLinks()
   X_resampled, y_resampled = tomek.fit_resample(X_train, y_train)
   ```

3. **Cluster Centroids**
   This technique uses clustering algorithms to reduce the majority class by replacing a cluster of majority class samples with the cluster centroid.

   **Pros:**

   - Reduces the dataset size while preserving the overall distribution.
   - Mitigates the loss of important samples.

   **Cons:**

   - More computationally intensive.
   - May introduce bias if clusters are not representative.

   ```python
   from imblearn.under_sampling import ClusterCentroids

   cc = ClusterCentroids(random_state=42)
   X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
   ```

### Combined Over-Sampling and Under-Sampling Techniques

1. **SMOTE + Tomek Links**
   Combines SMOTE and Tomek Links to both generate synthetic samples for the minority class and remove noisy examples near the decision boundary.

   **Pros:**

   - Balances the dataset while cleaning up noisy samples.
   - Can improve classifier performance.

   **Cons:**

   - More complex to implement.

   ```python
   from imblearn.combine import SMOTETomek

   smote_tomek = SMOTETomek(random_state=42)
   X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
   ```

2. **SMOTE + ENN (Edited Nearest Neighbors)**
   Combines SMOTE and Edited Nearest Neighbors to generate synthetic samples and then remove examples whose class is different from the majority of its nearest neighbors.

   **Pros:**

   - Generates synthetic samples and reduces noise.
   - Can lead to better model performance.

   **Cons:**

   - Computationally intensive.

   ```python
   from imblearn.combine import SMOTEENN

   smote_enn = SMOTEENN(random_state=42)
   X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
   ```
