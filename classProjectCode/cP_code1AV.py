import pandas as pd
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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

# Prepare data for SVM
train_data = pd.concat([dhf_train[top_genes], df_train[top_genes]])
test_data = pd.concat([dhf_test[top_genes], df_test[top_genes]])
train_labels = ['DHF'] * len(dhf_train) + ['DF'] * len(df_train)
test_labels = ['DHF'] * len(dhf_test) + ['DF'] * len(df_test)

# Convert labels to numeric
train_labels_numeric = np.array([1 if label == 'DHF' else 0 for label in train_labels])
test_labels_numeric = np.array([1 if label == 'DHF' else 0 for label in test_labels])

# Iterative SVM training on gene pairs
svm_results = []
for gene_pair in itertools.combinations(top_genes, 2):
    X_train = train_data[list(gene_pair)].values
    y_train = train_labels_numeric
    
    # Fit SVM
    clf = SVC(kernel='linear', probability=True, random_state=42)
    clf.fit(X_train, y_train)
    
    svm_results.append({
        'Gene_Pair': gene_pair,
        'Classifier': clf
    })

# Plot SVM decision boundaries for each gene pair
for svm_result in svm_results:
    gene_pair = svm_result['Gene_Pair']
    clf = svm_result['Classifier']

    # Prepare data for the plot
    X = train_data[list(gene_pair)].values
    y = train_labels_numeric

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label='DF', alpha=0.7, color='blue')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='DHF', alpha=0.7, color='orange')

    # Decision boundary
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_vals = -(clf.coef_[0, 0] * x_vals + clf.intercept_[0]) / clf.coef_[0, 1]
    plt.plot(x_vals, y_vals, color='black', linestyle='--', label='Decision Boundary')

    # Plot details
    plt.xlabel(gene_pair[0])
    plt.ylabel(gene_pair[1])
    plt.title(f"SVM Decision Boundary for Gene Pair: {gene_pair}")
    plt.legend()
    plt.show()

# Evaluate classifiers on the test set and identify the best 5 based on accuracy
classifier_performance = []
for svm_result in svm_results:
    gene_pair = svm_result['Gene_Pair']
    clf = svm_result['Classifier']
    X_test = test_data[list(gene_pair)].values
    
    # Predict on the test set
    predictions = clf.predict(X_test)
    accuracy = np.mean(predictions == test_labels_numeric)
    
    classifier_performance.append({
        'Gene_Pair': gene_pair,
        'Classifier': clf,
        'Accuracy': accuracy
    })

# Sort by accuracy
sorted_classifiers = sorted(classifier_performance, key=lambda x: x['Accuracy'], reverse=True)

# Extract the top 5 classifiers
top_5_classifiers = sorted_classifiers[:5]

# Display the top 5 classifiers and their accuracies
top_5_summary = pd.DataFrame([{
    'Gene_Pair': classifier['Gene_Pair'],
    'Accuracy': classifier['Accuracy']
} for classifier in top_5_classifiers])

print("Top 5 Gene Pair Classifiers:")
print(top_5_summary)

# Plot test set results for each of the top 5 classifiers
for classifier in top_5_classifiers:
    gene_pair = classifier['Gene_Pair']
    clf = classifier['Classifier']
    
    # Prepare test data
    X_test = test_data[list(gene_pair)].values
    y_test = test_labels_numeric
    
    # Predict on test set
    predictions = clf.predict(X_test)
    
    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], label='DF (True)', alpha=0.7, color='blue')
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], label='DHF (True)', alpha=0.7, color='orange')
    
    # Highlight misclassified points
    misclassified = y_test != predictions
    plt.scatter(X_test[misclassified, 0], X_test[misclassified, 1], label='Misclassified', color='red', marker='x')
    
    # Decision boundary
    coef = clf.coef_[0]
    intercept = clf.intercept_[0]
    x_vals = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
    y_vals = -(coef[0] * x_vals + intercept) / coef[1]
    plt.plot(x_vals, y_vals, color='black', linestyle='--', label='Decision Boundary')
    
    # Plot details
    plt.xlabel(gene_pair[0])
    plt.ylabel(gene_pair[1])
    plt.title(f"SVM Test Set Results for Gene Pair: {gene_pair}")
    plt.legend()
    plt.show()