#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis, Statistical Analysis and Machine Learning using a Real-World Dataset
# 
# **Course:** BCS 404: Introduction to Data Science with Python
# **Institution:** Accra Technical University, Department of Computer Science
# **Lecturer:** Dr. Joseph Dadzie
# **Student:** Illona Addae
# **Academic Year:** 2025/2026 Second Semester
# 
# **Dataset:** Titanic Dataset (Kaggle) - https://www.kaggle.com/competitions/titanic/data
# 
# This notebook works through six tasks: data acquisition, data cleaning, data visualisation, statistical analysis, a logistic regression model that predicts passenger survival, and a closing discussion.

# ## Task 1: Data Acquisition
# 
# The training file (`train.csv`) was downloaded from the Kaggle Titanic competition page and stored in the `data/` folder of this project. It contains records for 891 of the passengers who were aboard the RMS Titanic when it sank on 15 April 1912, including whether each passenger survived.

# In[1]:


# Core libraries used throughout the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

RANDOM_STATE = 42


# In[2]:


# Import the dataset into Python
df = pd.read_csv("data/train.csv")

print("Dataset dimensions (rows, columns):", df.shape)


# In[3]:


# Column names
print("Column names:")
for col in df.columns:
    print(" -", col)


# In[4]:


# First five observations
df.head()


# In[5]:


# Data types of each column
df.dtypes


# **Observations.** The dataset holds 891 rows and 12 columns. Seven columns are numeric (`PassengerId`, `Survived`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`) and five are text (`Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`). `Survived` is the target variable (0 = did not survive, 1 = survived). `Pclass` is the ticket class (1st, 2nd or 3rd), `SibSp` counts siblings or spouses aboard, `Parch` counts parents or children aboard, and `Embarked` records the port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

# ## Task 2: Data Cleaning
# 
# ### 2.1 Detecting missing values

# In[6]:


# Count and percentage of missing values per column
missing = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing %": (df.isnull().mean() * 100).round(2)
})
missing[missing["Missing Count"] > 0]


# Three columns contain missing values: `Age` (177 values, 19.87%), `Cabin` (687 values, 77.10%) and `Embarked` (2 values, 0.22%).
# 
# ### 2.2 Handling missing values
# 
# Each column is treated differently because the amount and nature of the missingness differ:
# 
# * **Age (19.87% missing).** Age is central to this analysis, so the rows cannot be discarded without losing a fifth of the data. Instead of filling every gap with one global figure, each missing age is replaced with the **median age of passengers sharing the same ticket class and sex**. Age varies noticeably across these groups (first class passengers were older on average), so a group median preserves that structure better than a single overall value. The median is preferred to the mean because the age distribution is right skewed and the median resists the influence of the few elderly passengers.
# * **Cabin (77.10% missing).** With more than three quarters of the values absent, any imputation would essentially invent data. The column is therefore **dropped**. Before dropping it, a simple indicator `HasCabin` is retained, since having a recorded cabin is itself informative (cabin numbers were recorded mainly for first class passengers).
# * **Embarked (0.22% missing).** Only two values are missing, so they are filled with the **mode** (S = Southampton), the port from which the overwhelming majority of passengers boarded. Filling two values with the most common category has a negligible effect on the analysis.

# In[7]:


# Work on a copy so the raw data stays untouched
data = df.copy()

# Age: median within each (Pclass, Sex) group
data["Age"] = data.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda s: s.fillna(s.median())
)

# Cabin: keep an indicator, then drop the sparse column
data["HasCabin"] = data["Cabin"].notnull().astype(int)
data = data.drop(columns=["Cabin"])

# Embarked: fill the two gaps with the mode
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

print("Missing values remaining per column:")
print(data.isnull().sum())


# ### 2.3 Detecting and removing duplicates

# In[8]:


# Full-row duplicates and duplicated passenger identifiers
print("Duplicated full rows:", data.duplicated().sum())
print("Duplicated PassengerId values:", data["PassengerId"].duplicated().sum())

# Remove duplicates if any exist (none are found, so the shape is unchanged)
data = data.drop_duplicates()
print("Shape after duplicate removal:", data.shape)


# **Summary of preprocessing decisions.** No duplicated rows exist, so nothing was removed. Missing ages were imputed with group medians, the two missing embarkation ports were filled with the mode, and the `Cabin` column was dropped after its presence was recorded in `HasCabin`. The cleaned dataset keeps all 891 observations and contains no missing values.

# ## Task 3: Data Visualisation
# 
# ### 3.1 Histogram of passenger ages

# In[9]:


fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(data["Age"], bins=30, color="#4c72b0", edgecolor="white")
ax.set_title("Distribution of Passenger Ages on the Titanic")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("figures/fig1_age_histogram.png", bbox_inches="tight")
plt.show()


# **Interpretation.** The age distribution is right skewed. Most passengers were young adults between about 20 and 35 years, and the frequency falls away steadily after 40. A small but visible group of infants and young children appears at the left of the plot, while passengers above 65 were rare. The tall bars in the mid twenties partly reflect the group-median imputation, which placed missing ages at the medians of their class and sex groups.

# ### 3.2 Bar chart of passenger class distribution

# In[10]:


class_counts = data["Pclass"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(class_counts.index.astype(str), class_counts.values,
       color=["#4c72b0", "#dd8452", "#55a868"])
ax.set_title("Passenger Class Distribution")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Number of Passengers")
for i, v in enumerate(class_counts.values):
    ax.text(i, v + 8, str(v), ha="center")
plt.tight_layout()
plt.savefig("figures/fig2_class_distribution.png", bbox_inches="tight")
plt.show()


# **Interpretation.** Third class dominates the sample with 491 passengers (55.1%), followed by first class with 216 (24.2%) and second class with 184 (20.7%). The Titanic carried far more economy travellers, many of them emigrants, than premium passengers, and this imbalance matters later because class turns out to be strongly linked to survival.

# ### 3.3 Boxplot of age by passenger class

# In[11]:


fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=data, x="Pclass", y="Age", hue="Pclass",
            palette="Set2", legend=False, ax=ax)
ax.set_title("Age Distribution by Passenger Class")
ax.set_xlabel("Passenger Class")
ax.set_ylabel("Age (years)")
plt.tight_layout()
plt.savefig("figures/fig3_age_by_class_boxplot.png", bbox_inches="tight")
plt.show()

print(data.groupby("Pclass")["Age"].median().rename("Median age"))


# **Interpretation.** Age falls steadily as class number rises. First class has the highest median age (38 years) and the widest spread, second class sits in the middle (30), and third class is the youngest (25) with several elderly outliers. This pattern is economically sensible: older passengers had accumulated the wealth needed for expensive first class tickets, while younger travellers and emigrating families filled third class.

# ### 3.4 Scatter plot of Age versus Fare

# In[12]:


fig, ax = plt.subplots(figsize=(8, 5))
scatter = sns.scatterplot(data=data, x="Age", y="Fare", hue="Pclass",
                          palette="deep", alpha=0.6, ax=ax)
ax.set_title("Age versus Fare, Coloured by Passenger Class")
ax.set_xlabel("Age (years)")
ax.set_ylabel("Fare (British pounds)")
plt.tight_layout()
plt.savefig("figures/fig4_age_vs_fare_scatter.png", bbox_inches="tight")
plt.show()

print("Pearson correlation between Age and Fare:",
      round(data["Age"].corr(data["Fare"]), 3))


# **Interpretation.** There is no strong linear relationship between age and fare; the correlation is weakly positive (r = 0.12). Fares are layered by class rather than by age: third class fares cluster tightly below roughly 15 pounds regardless of age, while first class fares spread widely upward. Three passengers paid an extreme fare of just over 512 pounds, visible as isolated points at the top of the plot. Class, not age, is what drives ticket price.

# ### 3.5 Correlation heatmap

# In[13]:


numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
corr = data[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap of Numerical Variables")
plt.tight_layout()
plt.savefig("figures/fig5_correlation_heatmap.png", bbox_inches="tight")
plt.show()


# **Interpretation.** `PassengerId` is excluded because it is only a row label. The strongest relationships are the negative correlation between `Pclass` and `Fare` (about -0.55, higher classes pay more; recall class 1 is numerically smallest) and the positive correlation between `SibSp` and `Parch` (about 0.41, family members travel together). For the target, `Survived` correlates negatively with `Pclass` (about -0.34) and positively with `Fare` (about 0.26): wealthier, higher class passengers survived more often.

# ### 3.6 Pairplot of selected numerical variables

# In[14]:


pair_vars = ["Age", "Fare", "SibSp", "Parch"]
g = sns.pairplot(data, vars=pair_vars, hue="Survived",
                 palette={0: "#c44e52", 1: "#55a868"},
                 plot_kws={"alpha": 0.5, "s": 20}, corner=False)
g.figure.suptitle("Pairwise Relationships of Selected Numerical Variables by Survival",
                  y=1.02)
g.savefig("figures/fig6_pairplot.png", bbox_inches="tight")
plt.show()


# **Interpretation.** The clearest separation between survivors (green) and non-survivors (red) appears along `Fare`: survivors stretch into the higher fare range while non-survivors concentrate near the bottom. The `Age` panels show that young children survived relatively often. `SibSp` and `Parch` are discrete counts, and their panels hint that passengers with very large families (SibSp of 4 or more) rarely survived, while those travelling with one or two relatives did comparatively well.

# ## Task 4: Statistical Analysis
# 
# ### 4.1 Descriptive statistics

# In[15]:


# Numerical variables
data[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].describe().round(2)


# In[16]:


# Categorical variables
data[["Sex", "Embarked"]].describe()


# **Reading the table.** The mean of `Survived` (0.38) says that only 38.4% of the 891 passengers survived. The average passenger was about 29 years old and paid roughly 32 pounds, but the fare column is heavily skewed: the median fare is only 14.45 pounds while the maximum is 512.33, so a few luxury tickets pull the mean far above the typical fare. Most passengers travelled without family (medians of 0 for both `SibSp` and `Parch`). Among the categorical columns, 577 of 891 passengers (64.8%) were male and 646 (72.5%) boarded at Southampton.
# 
# ### 4.2 Frequency distributions

# In[17]:


for col in ["Survived", "Pclass", "Sex", "Embarked"]:
    freq = pd.DataFrame({
        "Count": data[col].value_counts(),
        "Percentage": (data[col].value_counts(normalize=True) * 100).round(2)
    })
    print(f"--- {col} ---")
    print(freq, end="\n\n")


# ### 4.3 Correlation analysis, strongest positive and strongest negative

# In[18]:


corr_matrix = data[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].corr()

# Flatten the matrix, remove self-correlations and duplicate pairs
pairs = corr_matrix.unstack()
pairs = pairs[pairs.index.get_level_values(0) < pairs.index.get_level_values(1)]

strongest_pos = pairs.idxmax(), pairs.max()
strongest_neg = pairs.idxmin(), pairs.min()

print("Correlation matrix:")
print(corr_matrix.round(3), end="\n\n")
print(f"Strongest positive correlation: {strongest_pos[0][0]} and "
      f"{strongest_pos[0][1]} (r = {strongest_pos[1]:.3f})")
print(f"Strongest negative correlation: {strongest_neg[0][0]} and "
      f"{strongest_neg[0][1]} (r = {strongest_neg[1]:.3f})")


# The **strongest positive correlation** is between `Parch` and `SibSp` (r = 0.415): passengers travelling with siblings or a spouse also tended to travel with parents or children, which is exactly what whole families aboard produce. The **strongest negative correlation** is between `Fare` and `Pclass` (r = -0.549): since class 1 is the smallest number but the most expensive ticket, fare rises as the class number falls.
# 
# ### 4.4 Three important statistical findings

# In[19]:


# Finding 1: survival depends heavily on sex
sex_surv = data.groupby("Sex")["Survived"].agg(["mean", "count"])
sex_surv["mean"] = (sex_surv["mean"] * 100).round(1)
print("Survival rate by sex (%):")
print(sex_surv, end="\n\n")

contingency = pd.crosstab(data["Sex"], data["Survived"])
chi2, p, dof, _ = stats.chi2_contingency(contingency)
print(f"Chi-square test (Sex vs Survived): chi2 = {chi2:.2f}, p-value = {p:.3e}")


# In[20]:


# Finding 2: survival falls as passenger class drops
class_surv = data.groupby("Pclass")["Survived"].agg(["mean", "count"])
class_surv["mean"] = (class_surv["mean"] * 100).round(1)
print("Survival rate by passenger class (%):")
print(class_surv, end="\n\n")

contingency2 = pd.crosstab(data["Pclass"], data["Survived"])
chi2_2, p2, dof2, _ = stats.chi2_contingency(contingency2)
print(f"Chi-square test (Pclass vs Survived): chi2 = {chi2_2:.2f}, p-value = {p2:.3e}")


# In[21]:


# Finding 3: survivors paid substantially higher fares
fare_by_outcome = data.groupby("Survived")["Fare"].agg(["mean", "median"]).round(2)
print("Fare by survival outcome:")
print(fare_by_outcome, end="\n\n")

t_stat, p3 = stats.ttest_ind(data.loc[data["Survived"] == 1, "Fare"],
                             data.loc[data["Survived"] == 0, "Fare"],
                             equal_var=False)
print(f"Welch t-test on Fare (survivors vs non-survivors): "
      f"t = {t_stat:.2f}, p-value = {p3:.3e}")


# **Finding 1 - Sex was the strongest single determinant of survival.** 74.2% of women survived against only 18.9% of men. The chi-square test rejects independence between sex and survival overwhelmingly (p far below 0.001), consistent with the historical "women and children first" evacuation order.
# 
# **Finding 2 - Survival declined sharply with class.** First class passengers survived at 63.0%, second class at 47.3% and third class at only 24.2%. The chi-square test again shows the association is highly significant. First class cabins were closer to the boat deck, and access to lifeboats favoured premium passengers.
# 
# **Finding 3 - Money bought safety.** Survivors paid an average fare of about 48 pounds against 22 pounds for those who died, and the Welch t-test confirms the gap is statistically significant. Fare is effectively a proxy for class and cabin location, so this finding reinforces the class effect rather than describing an independent mechanism.

# ## Task 5: Machine Learning - Predicting Survival with Logistic Regression
# 
# ### 5.1 Selecting predictor variables
# 
# The predictors are chosen from variables the exploratory analysis showed to be related to survival: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked` and the engineered `HasCabin` indicator. Identifier-like columns (`PassengerId`, `Name`, `Ticket`) carry no generalisable signal and are excluded. `Sex` is encoded as 0/1 and `Embarked` is one-hot encoded because logistic regression requires numeric inputs.

# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

ml = data.copy()
ml["Sex"] = ml["Sex"].map({"male": 0, "female": 1})
ml = pd.get_dummies(ml, columns=["Embarked"], drop_first=True)

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
            "HasCabin", "Embarked_Q", "Embarked_S"]
X = ml[features]
y = ml["Survived"]

print("Feature matrix shape:", X.shape)
X.head()


# ### 5.2 Splitting into training and testing sets
# 
# An 80/20 split is used. Stratification on the target keeps the 38/62 survival ratio the same in both sets, and a fixed random state makes the experiment reproducible.

# In[23]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print("Training set:", X_train.shape[0], "passengers")
print("Testing set: ", X_test.shape[0], "passengers")


# ### 5.3 Training the Logistic Regression classifier
# 
# Continuous features are standardised (zero mean, unit variance) before fitting. Scaling parameters are learned on the training set only, then applied to the test set, so no information leaks from the test data into training.

# In[24]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(X_train_scaled, y_train)

coefs = pd.Series(model.coef_[0], index=features).sort_values()
print("Model coefficients (standardised features):")
print(coefs.round(3))


# ### 5.4 Predicting the testing data and 5.5 Evaluation

# In[25]:


y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")


# In[26]:


cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Did not survive", "Survived"]).plot(
    cmap="Blues", ax=ax, colorbar=False)
ax.set_title("Confusion Matrix - Logistic Regression on Test Set")
plt.tight_layout()
plt.savefig("figures/fig7_confusion_matrix.png", bbox_inches="tight")
plt.show()


# In[27]:


print(classification_report(y_test, y_pred,
      target_names=["Did not survive", "Survived"]))


# ### 5.6 Discussion of model performance
# 
# The classifier is evaluated against a natural baseline: always predicting "did not survive" would score about 61.6% simply because most passengers died. The logistic regression clearly beats that baseline, so the model has learned real structure from the data.
# 
# Reading the confusion matrix, the model is stronger at identifying passengers who died (higher recall for class 0) than at identifying survivors, where some survivors are misclassified as deaths. This asymmetry is expected: the classes are imbalanced and some survivors, such as third class men, look statistically identical to passengers who died, so no simple model can separate them.
# 
# The coefficient table matches the statistical analysis. `Sex` carries the largest positive weight (being female raises the predicted survival odds most), while `Pclass` has a strong negative weight (moving from first towards third class lowers the odds). `Age` contributes negatively, reflecting the priority given to children. These agreements between the model and the earlier chi-square findings suggest the model is capturing genuine patterns rather than noise.
# 
# Overall, for a first model with modest feature engineering, logistic regression gives solid and interpretable performance. Possible improvements include extracting titles from passenger names, building family size features, and trying tree-based ensembles such as random forests.

# ## Task 6: Discussion and Conclusion (summary)
# 
# This project analysed the Titanic passenger manifest end to end. The data required realistic cleaning: a fifth of ages were missing and were imputed with class and sex group medians, the sparse cabin column was reduced to an indicator, and two missing embarkation ports were filled with the mode. Visual and statistical analysis converged on one story: survival on the Titanic was not random. Sex was the dominant factor (74.2% of women survived against 18.9% of men), class came second (63.0% in first class against 24.2% in third), and fare, a proxy for class, separated survivors from victims. A logistic regression using these variables predicted survival on unseen passengers at an accuracy of roughly 80%, comfortably above the 61.6% majority-class baseline, with coefficients that mirror the statistical findings.
# 
# The main limitations are the modest sample of 891 passengers, the information lost through imputation and through dropping the cabin detail, and the fact that logistic regression assumes a linear relationship between the log odds and the features. The full two-page discussion, with limitations and recommendations, is presented in the accompanying project report.

# In[28]:


# Export key results for the written report
import json

results = {
    "shape": list(df.shape),
    "missing": df.isnull().sum().to_dict(),
    "class_counts": data["Pclass"].value_counts().sort_index().to_dict(),
    "median_age_by_class": data.groupby("Pclass")["Age"].median().to_dict(),
    "age_fare_corr": round(float(data["Age"].corr(data["Fare"])), 3),
    "corr_matrix": corr_matrix.round(3).to_dict(),
    "strongest_pos": [list(strongest_pos[0]), round(float(strongest_pos[1]), 3)],
    "strongest_neg": [list(strongest_neg[0]), round(float(strongest_neg[1]), 3)],
    "sex_survival": data.groupby("Sex")["Survived"].mean().round(4).to_dict(),
    "class_survival": data.groupby("Pclass")["Survived"].mean().round(4).to_dict(),
    "fare_by_outcome": fare_by_outcome.to_dict(),
    "chi2_sex": [round(chi2, 2), float(p)],
    "chi2_class": [round(chi2_2, 2), float(p2)],
    "ttest_fare": [round(t_stat, 2), float(p3)],
    "describe": data[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]]
                .describe().round(2).to_dict(),
    "accuracy": round(float(acc), 4),
    "confusion_matrix": cm.tolist(),
    "coefficients": coefs.round(3).to_dict(),
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
    "report_text": classification_report(y_test, y_pred,
                    target_names=["Did not survive", "Survived"]),
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results exported for the report.")

