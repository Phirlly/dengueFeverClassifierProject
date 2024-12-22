import pandas as pd
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import itertools
import numpy as np

# Load the dataset
file_path = './Dengue_Fever_Prognosis_Dataset.csv'
data = pd.read_csv(file_path)

# Identify DHF and DF columns
dhf_columns = [col for col in data.columns if "DHF" in col]
df_columns = [col for col in data.columns if "DF" in col]

# Filter DHF and DF data
dhf_data = data[["Probe_Set_ID"] + dhf_columns].set_index("Probe_Set_ID").T
df_data = data[["Probe_Set_ID"] + df_columns].set_index("Probe_Set_ID").T

# Add categories for combined data
dhf_data['Category'] = 'DHF'
df_data['Category'] = 'DF'

# Combine data
combined_data = pd.concat([dhf_data, df_data])

# Split DHF and DF data into training and testing sets
dhf_train, dhf_test = train_test_split(dhf_data.drop(columns='Category'), test_size=0.2, random_state=42)
df_train, df_test = train_test_split(df_data.drop(columns='Category'), test_size=0.2, random_state=42)

# Perform ANOVA test for each Probe_Set_ID
anova_results = []
for probe in data["Probe_Set_ID"]:
    # Extract values for DHF and DF in the training set
    dhf_values = dhf_train[probe].dropna().values
    df_values = df_train[probe].dropna().values
    
    # Perform ANOVA test
    f_stat, p_value = f_oneway(dhf_values, df_values)
    anova_results.append({'Probe_Set_ID': probe, 'F-statistic': f_stat, 'p-value': p_value})

# Convert results to DataFrame and sort by p-value
anova_results_df = pd.DataFrame(anova_results).sort_values(by='p-value')
top_genes = anova_results_df.head(10)['Probe_Set_ID'].tolist()

# Prepare data for LDA
train_data = pd.concat([dhf_train[top_genes], df_train[top_genes]])
test_data = pd.concat([dhf_test[top_genes], df_test[top_genes]])
train_labels = ['DHF'] * len(dhf_train) + ['DF'] * len(df_train)
test_labels = ['DHF'] * len(dhf_test) + ['DF'] * len(df_test)

# Iterative LDA on gene pairs
lda_results = []
for gene_pair in itertools.combinations(top_genes, 2):
    X_train = train_data[list(gene_pair)].values
    y_train = np.array([1 if label == 'DHF' else 0 for label in train_labels])
    
    # Fit LDA
    clf = LDA()
    clf.fit(X_train, y_train)
    coef = clf.coef_[0]
    intercept = clf.intercept_[0]
    
    lda_results.append({
        'Gene_Pair': gene_pair,
        'Coefficients': coef,
        'Intercept': intercept,
        'Classifier': clf
    })

# Plot LDA decision boundaries for each gene pair
for lda_result in lda_results:
    gene_pair = lda_result['Gene_Pair']
    clf = lda_result['Classifier']
    coef = lda_result['Coefficients']
    intercept = lda_result['Intercept']

    # Prepare data for the plot
    X = train_data[list(gene_pair)].values
    y = np.array([1 if label == 'DHF' else 0 for label in train_labels])

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label='DF', alpha=0.7)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='DHF', alpha=0.7)

    # Decision boundary
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_vals = -(coef[0] * x_vals + intercept) / coef[1]
    plt.plot(x_vals, y_vals, color='black', linestyle='--', label='Decision Boundary')

    # Plot details
    plt.xlabel(gene_pair[0])
    plt.ylabel(gene_pair[1])
    plt.title(f"LDA Decision Boundary for Gene Pair: {gene_pair}")
    plt.legend()
    plt.show()

# Display ANOVA results for top genes
print("ANOVA Results for Top Genes:")
print(anova_results_df.head(10))
