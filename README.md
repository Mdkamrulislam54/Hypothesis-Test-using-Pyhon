# Hypothesis-Test-using-Pyhon


Hypothesis_Tests_in_Python.ipynb
Hypothesis_Tests_in_Python.ipynb_
Parametric Hypothesis Tests
Assumptions for Choosing a Parametric Hypothesis Test
image.png

When to Perform which Parametrics Test?
image.png

Generating Synthetic Data and Performing the Tests

[ ]
import pandas as pd
import numpy as np

# Sample Dataset
np.random.seed(42)
sales_data = pd.DataFrame({
    'region': np.random.choice(['North', 'South', 'East', 'West'], 200),
    'campaign_type': np.random.choice(['Email', 'Social Media', 'TV'], 200),
    'before_sales': np.random.normal(1000, 200, 200),
    'after_sales': np.random.normal(1100, 250, 200),
    'email_open': np.random.choice(['Yes', 'No'], 200),
    'gender': np.random.choice(['Male', 'Female'], 200),
    'ad_spend': np.random.normal(5000, 1500, 200),
    'revenue': np.random.normal(7000, 1800, 200),
})


[ ]
sales_data.head(10)

Shapiro-Wilk Test for Normality
When to Use: Validate normality of a sample

Hypotheses:

H₀: Data is normal

H₁: Data is not normal


[ ]
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Check if revenue is normally distributed
stat, p = shapiro(sales_data['revenue'])
print(f"Shapiro-Wilk Test: Statistic={stat:.4f}, p-value={p:.4f}")
if p > 0.05:
    print("✔️ Likely normal (p > 0.05)")
else:
    print("❌ Not normal (p < 0.05)")

Shapiro-Wilk Test: Statistic=0.9895, p-value=0.1488
✔️ Likely normal (p > 0.05)

[ ]
# Q-Q plot
stats.probplot(sales_data['revenue'], dist="norm", plot=plt)
plt.title("Q-Q Plot for Revenue")
plt.show()

# Histogram
sns.histplot(sales_data['revenue'], kde=True)
plt.title("Histogram of Revenue")
plt.show()


Levene’s Test for Equal Variance [Homogeneity of Variance]

[ ]
from scipy.stats import levene

# Revenue across two regions
north = sales_data[sales_data['region'] == 'North']['revenue']
south = sales_data[sales_data['region'] == 'South']['revenue']

stat, p = levene(north, south)
print(f"Levene’s Test: Statistic={stat:.4f}, p-value={p:.4f}")
if p > 0.05:
    print("✔️ Equal variance (p > 0.05)")
else:
    print("❌ Unequal variance (p < 0.05)")


Levene’s Test: Statistic=0.0177, p-value=0.8946
✔️ Equal variance (p > 0.05)
1. Two-Sample T-Test (Independent Groups)
When to Use: Compare means of two independent groups (e.g., Region A vs Region B)

Assumptions:

✅ Data in both groups is normally distributed (Shapiro-Wilk)

✅ Equal variances (Levene's Test)

✅ Independent groups

Hypotheses:

H₀: μ₁ = μ₂ (no difference in means)

H₁: μ₁ ≠ μ₂ (means are different)

Interpretation:

If p < 0.05 → reject H₀ → there is a statistically significant difference in revenue between North and South.


[ ]
from scipy.stats import ttest_ind

# Step 1: Check normality
for region in ['North', 'South']:
    stat, p = shapiro(sales_data[sales_data['region'] == region]['revenue'])
    print(f"{region} Shapiro-Wilk p = {p:.4f}")

# Step 2: Check variance equality
north = sales_data[sales_data['region'] == 'North']['revenue']
south = sales_data[sales_data['region'] == 'South']['revenue']
stat, p = levene(north, south)
print(f"Levene’s p = {p:.4f}")

# Step 3: Perform t-test
stat, p = ttest_ind(north, south, equal_var=(p > 0.05))
print(f"t-test: Statistic = {stat:.4f}, p-value = {p:.4f}")

North Shapiro-Wilk p = 0.7515
South Shapiro-Wilk p = 0.1066
Levene’s p = 0.8946
t-test: Statistic = 0.9304, p-value = 0.3546
2. Paired T-Test (Before vs After Campaign)
When to Use: Compare before/after sales in the same region

Assumptions:

Normality of the differences

Paired data (dependent samples)

Hypotheses:

H₀: μ₁ - μ₂ = 0

H₁: μ₁ - μ₂ ≠ 0


[ ]
from scipy.stats import ttest_rel

t_stat, p_val = ttest_rel(sales_data['before_sales'], sales_data['after_sales'])
print(f"Paired T-Test: t-stat={t_stat:.4f}, p-value={p_val:.4f}")

Paired T-Test: t-stat=-3.3087, p-value=0.0011
3. ANOVA
When to Use: Compare more than 2 groups

Assumptions:

Normality in all groups

Equal variances

Independence

Hypotheses:

H₀: All group means are equal

H₁: At least one differs


[ ]
from scipy.stats import f_oneway

email = sales_data[sales_data['campaign_type'] == 'Email']['revenue']
social = sales_data[sales_data['campaign_type'] == 'Social Media']['revenue']
tv = sales_data[sales_data['campaign_type'] == 'TV']['revenue']

f_stat, p_val = f_oneway(email, social, tv)
print(f"ANOVA: F-stat={f_stat:.4f}, p-value={p_val:.4f}")



ANOVA: F-stat=2.2356, p-value=0.1096

[ ]
#Another Way of Running ANOVA:
from scipy.stats import f_oneway

grouped = sales_data.groupby('campaign_type')['revenue'].apply(list)
stat, p = f_oneway(*grouped)
print(f"ANOVA: Statistic = {stat:.4f}, p-value = {p:.4f}")
ANOVA: Statistic = 2.2356, p-value = 0.1096
4. Chi-Square Test of Independence

[ ]
from scipy.stats import chi2_contingency

contingency = pd.crosstab(sales_data['gender'], sales_data['email_open'])
contingency



[ ]
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"Chi-Square: χ²={chi2:.4f}, p-value={p:.4f}")
Chi-Square: χ²=0.4616, p-value=0.4969
5. Chi-Square Goodness of Fit Test
When to Use: Does observed frequency differ from expected?

Hypotheses:

H₀: Observed = Expected

H₁: Observed ≠ Expected


[ ]
from scipy.stats import chisquare

observed = [100, 110, 130]  # Observed customers across 3 campaigns
expected = [116.67, 116.67, 116.67]

# Ensure the sum of expected frequencies equals the sum of observed frequencies
expected[-1] = sum(observed) - sum(expected[:-1])  # Adjust the last expected value

stat, p = chisquare(f_obs=observed, f_exp=expected)
print(f"Chi-square Goodness of Fit: Stat={stat:.4f}, p={p:.4f}")
Chi-square Goodness of Fit: Stat=7.8706, p=0.0195
6. Correlation Coefficient

[ ]
correlation_coefficient = sales_data['ad_spend'].corr(sales_data['revenue'])
correlation_coefficient
np.float64(0.11570298374999959)

[ ]
from scipy.stats import pearsonr
corr, p_val = pearsonr(sales_data['ad_spend'], sales_data['revenue'])
print(f"Pearson Correlation: r={corr:.4f}, p-value={p_val:.4f}")
Pearson Correlation: r=0.1157, p-value=0.1028
Non-Parametric Hypothesis Tests
When to Use Non-Parametrics Tests over Parametric Tests?
image.png

image.png

Non-Parametric Tests Overview
image.png

1. Mann-Whitney U Test (Non-parametric alternative to Two-Sample t-test)
When to Use: Same goal as above, but normality not assumed

Assumptions:

Ordinal or continuous scale

Independent samples

Hypotheses:

H₀: Distributions are equal (no difference)

H₁: Distributions are different


[ ]
from scipy.stats import mannwhitneyu

stat, p = mannwhitneyu(north, south, alternative='two-sided')
print(f"Mann-Whitney U: Statistic = {stat:.4f}, p-value = {p:.4f}")

Mann-Whitney U: Statistic = 1192.0000, p-value = 0.2972
2. Wilcoxon Signed-Rank Test (Non-parametric paired test)
When to Use: Same as paired t-test, but for non-normal differences


[ ]
from scipy.stats import wilcoxon

stat, p = wilcoxon(sales_data['before_sales'], sales_data['after_sales'])
print(f"Wilcoxon Signed-Rank: Statistic = {stat:.4f}, p-value = {p:.4f}")

Wilcoxon Signed-Rank: Statistic = 7632.0000, p-value = 0.0032
3. Kruskal-Wallis Test (Non-parametric ANOVA)

[ ]
from scipy.stats import kruskal

stat, p = kruskal(email, social, tv)
print(f"Kruskal-Wallis Test: Statistic={stat:.4f}, p-value={p:.4f}")

Kruskal-Wallis Test: Statistic=4.6208, p-value=0.0992
Colab paid products - Cancel contracts here
