# payroll-risk-system

# Introduction :
We built a machine learning classifcition system which basically predicts High Risk employees in payroll data.
The model is designed to help HR, finance, and compliance teams proactively detect employees who may pose financial, policy, or operational risks.
Payroll fraud and compliance risks often arise due to:

* Frequent manual payroll edits

* Policy violations

* Abnormal attendance patterns

* Sudden managerial or role changes

* Manually detecting such cases is time-consuming and error-prone.

This project automates the process by building a binary classification model that predicts whether an employee is High Risk or Low Risk.

# Basic Structure

Rows: 2300 employees

Columns: 32 features

Target Variable: High_Risk (Binary: 0 = Low Risk, 1 = High Risk)

# Dataset Description

File Used: risk_dataset.csv

* Dataset Size

   *Total records: 2300 employees

   *Total features: 32

   *Target column: High_Risk
* Feature Types
  *Categorical Features

  *Department

  *Location

  *Employment_Type

  *Shift_Type

   These represent organizational and work-environment context.

    *Numerical Features

    *Salary & compensation:
     Base_Salary, HRA, Allowances, Bonus

    *Attendance behavior:
Leaves_Taken, Sick_Leaves, Casual_Leaves, Late_Logins

Engagement & performance:
Engagement_Score, Training_Hours

Risk indicators:
Manual_Payroll_Edits, Policy_Violations, Previous_Audit_Flags, Manager_Changes

Aggregated signal:
Risk_Score

Target Variable

High_Risk

   1 → High-risk employee

   0 → Low-risk employee

# Data Loading

Imported the dataset using Pandas

Verified data shape, column names, and basic structure

#Libraries Used:

Pandas

NumPy

Scikit-learn

Matplotlib 

# Data Preprocessing

Identified categorical and numerical features

Applied encoding techniques to convert categorical variables into numeric form

Ensured the dataset was model-ready

# Feature & Target Separation

Independent variables (X) created using all predictive features

Dependent variable (y) set as High_Risk

# Train-Test Split

Split the dataset into:

Training set

Testing set

This ensures the model is evaluated on unseen data and avoids data leakage

# Model Selection & Training

Used a tree-based classification model

Tree-based models were chosen because:

They handle non-linear relationships well

They work effectively with mixed feature types

They reflect real-world decision logic used in audits

The model learns patterns such as:

High payroll edits + low engagement + prior audit flags → High Risk

# Model Evaluation

Evaluated performance using:

Confusion Matrix

Precision

Recall

F1-Score

These metrics are preferred over accuracy because:

Payroll risk datasets are often imbalanced

Missing a high-risk employee is more costly than a false alert

Insights Derived from Data Visualization

During exploratory data analysis (EDA), data visualizations were used to understand patterns, distributions, and risk drivers in the payroll dataset before modeling. The following key insights were observed:

1️⃣ Distribution of High-Risk vs Low-Risk Employees

By visualizing the target variable (High_Risk), it was clear that:

High-risk employees form a smaller proportion of the dataset

The data is imbalanced, which justified using precision, recall, and F1-score instead of accuracy

📌 Insight Taken:
Risk prediction is a rare-event problem, so the model must minimize false negatives.

2️⃣ Salary vs Risk Behavior

Visualizing salary components (Base_Salary, HRA, Allowances) against risk labels showed:

Salary alone does not strongly indicate risk

High-risk employees exist across multiple salary bands

📌 Insight Taken:
Risk is behavior-driven, not purely compensation-driven.

3️⃣ Manual Payroll Edits & Risk

Bar plots and box plots of Manual_Payroll_Edits revealed:

High-risk employees tend to have significantly more manual edits

Low-risk employees show minimal or zero edits

📌 Insight Taken:
Manual intervention in payroll is a strong early warning signal.

4️⃣ Policy Violations & Audit Flags

Visual comparisons of:

Policy_Violations

Previous_Audit_Flags

showed:

A clear upward trend in risk as violations increase

Employees with past audit flags are much more likely to be classified as high risk

📌 Insight Taken:
Historical compliance behavior strongly influences future risk.

5️⃣ Attendance Patterns

Plots of:

Late_Logins

Leaves_Taken

Sick_Leaves

indicated:

High-risk employees show irregular attendance patterns

Not just high leave count, but inconsistent patterns matter

📌 Insight Taken:
Behavioral inconsistency is more important than absolute leave count.

6️⃣ Engagement Score vs Risk

Scatter and box plots of Engagement_Score showed:

High-risk employees generally have lower engagement scores

There is a visible separation between low-risk and high-risk groups

📌 Insight Taken:
Low engagement often correlates with higher operational and compliance risk.

7️⃣ Department & Shift Analysis

Categorical visualizations (count plots) showed:

Certain departments and shift types have higher concentration of risk

Night shifts and operational roles show slightly elevated risk levels

📌 Insight Taken:
Risk is also influenced by work context, not just individual behavior.

8️⃣ Risk Score Validation

Visualizing Risk_Score against High_Risk confirmed:

A strong positive correlation

The score aligns well with final risk classification

📌 Insight Taken:
The engineered risk score is a meaningful predictor and supports the model logic.

# Feature Importance Analysis

After training the machine learning model, feature importance analysis was performed to understand which factors most strongly influence whether an employee is classified as high risk. This step was critical to ensure the model’s decisions align with real-world payroll and audit logic.

Most Influential Features
1️⃣ Manual Payroll Edits

This emerged as the strongest predictor

Employees with frequent manual salary modifications were far more likely to be flagged as high risk

Interpretation:
Manual intervention in payroll is often linked to overrides, corrections, or manipulation, making it a key risk signal.

2️⃣ Previous Audit Flags

Employees previously flagged in audits showed a significantly higher probability of being high risk again

Interpretation:
Historical compliance behavior is a strong indicator of future risk.

3️⃣ Policy Violations

The number of policy violations had a direct positive relationship with risk classification

Interpretation:
Repeated policy breaches reflect poor compliance culture and increase audit concern.

4️⃣ Engagement Score

Lower engagement scores were consistently associated with higher risk

Interpretation:
Disengaged employees are more likely to exhibit irregular behavior, making engagement a meaningful behavioral predictor.

5️⃣ Attendance-Related Features

Key contributors included:

Late_Logins

Leaves_Taken

Sick_Leaves

Interpretation:
Not just high leave counts, but irregular attendance patterns played an important role in risk detection.

6️⃣ Manager Changes

Employees with frequent reporting-manager changes showed elevated risk levels

Interpretation:
Frequent managerial shifts may indicate internal issues, role instability, or governance gaps.

7️⃣ Risk_Score

The engineered Risk_Score aligned strongly with the final prediction

Interpretation:
This validated that the aggregated risk logic used to compute this score was meaningful and well-designed.

# Key Takeaways from Feature Importance

 * Payroll risk is behavioral and compliance-driven, not compensation-driven

 * Historical audit signals significantly impact future risk prediction

 * Tree-based models successfully capture non-linear feature interactions
# Mathematical Justification of Model Choice
1️⃣ Nature of the Problem (Mathematical View)

The task is a binary classification problem:

𝑦
∈
{
0
,
1
}
y∈{0,1}

where

1 → High-Risk employee

0 → Low-Risk employee

The feature space contains:

Continuous variables (salary, engagement, leaves)

Discrete counts (policy violations, payroll edits)

Encoded categorical variables (department, shift type)

2️⃣ Why Linear Models Are Mathematically Insufficient

A linear or logistic regression model assumes a linear decision boundary:

𝑃
(
𝑦
=
1
∣
𝑋
)
=
𝜎
(
𝑤
𝑇
𝑋
+
𝑏
)
P(y=1∣X)=σ(w
T
X+b)

This implies:

Each feature contributes independently

Effects are additive

Interaction terms must be manually engineered

However, in payroll risk:

Risk arises from feature combinations, not single variables

Example (non-linear rule):

High edits AND low engagement AND past audit flag → High risk

This cannot be modeled efficiently by a linear hyperplane.

3️⃣ Why Tree-Based Models Are Mathematically Suitable

A decision tree partitions the feature space into non-overlapping regions:

𝑅
𝑛
=
𝑅
1
∪
𝑅
2
∪
⋯
∪
𝑅
𝑘
R
n
=R
1
	​

∪R
2
	​

∪⋯∪R
k
	​


Each region is defined by rules like:

𝑥
𝑗
≤
𝑡
x
j
	​

≤t

and the prediction is:

𝑦
^
=
majority class in 
𝑅
𝑘
y
^
	​

=majority class in R
k
	​


This allows the model to learn piecewise constant decision functions, which are ideal for rule-based risk detection.

4️⃣ Information Gain & Entropy (Core Mathematics)

Tree splits are chosen by maximizing Information Gain:

𝐼
𝐺
=
𝐻
(
𝑌
)
−
∑
𝑖
∣
𝑆
𝑖
∣
∣
𝑆
∣
𝐻
(
𝑆
𝑖
)
IG=H(Y)−
i
∑
	​

∣S∣
∣S
i
	​

∣
	​

H(S
i
	​

)

Where entropy is:

𝐻
(
𝑌
)
=
−
∑
𝑝
(
𝑦
)
log
⁡
𝑝
(
𝑦
)
H(Y)=−∑p(y)logp(y)

This means:

Features that best reduce uncertainty about High_Risk

Naturally prioritize strong risk signals (audit flags, edits)

Hence, the model mathematically prefers behavioral and compliance features, not salary magnitude.

5️⃣ Ensemble Advantage (Random Forest / Boosted Trees)

Ensemble tree models approximate complex functions:

𝑓
(
𝑋
)
=
∑
𝑚
=
1
𝑀
𝛼
𝑚
ℎ
𝑚
(
𝑋
)
f(X)=
m=1
∑
M
	​

α
m
	​

h
m
	​

(X)

Where:

ℎ
𝑚
(
𝑋
)
h
m
	​

(X) are weak decision trees

The ensemble reduces variance and overfitting

This improves:

Generalization

Stability on noisy HR data

Robustness to outliers

6️⃣ Handling Feature Interactions Automatically

Tree-based models implicitly learn interaction terms:

𝑓
(
𝑥
1
,
𝑥
2
)
≠
𝑓
(
𝑥
1
)
+
𝑓
(
𝑥
2
)
f(x
1
	​

,x
2
	​

)

=f(x
1
	​

)+f(x
2
	​

)

For example:

Salary alone → weak signal

Salary + edits + violations → strong signal

This interaction learning is automatic, unlike linear models.

7️⃣ Robustness to Distribution Assumptions

Tree-based models:

Do not assume normality

Do not assume linearity

Are invariant to monotonic feature scaling

This is crucial because payroll data:

Is skewed

Contains outliers

Has mixed distributions
