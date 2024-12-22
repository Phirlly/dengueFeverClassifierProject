import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = './Dengue_Fever_Prognosis_Dataset.csv'
data = pd.read_csv(file_path)

# Filter out ND columns and separate DHF and DF
dhf_columns = [col for col in data.columns if "DHF" in col]
df_columns = [col for col in data.columns if "DF" in col]

# Transpose for easier indexing
dhf_data = data[dhf_columns].T
df_data = data[df_columns].T

# Add a column to identify the category
dhf_data['Category'] = 'DHF'
df_data['Category'] = 'DF'

# Combine DHF and DF data
combined_data = pd.concat([dhf_data, df_data])

# Split into training and testing for each category
from sklearn.model_selection import train_test_split

dhf_train, dhf_test = train_test_split(dhf_data, test_size=0.2, random_state=42)
df_train, df_test = train_test_split(df_data, test_size=0.2, random_state=42)

# Prepare labels
dhf_train_labels = ['DHF'] * len(dhf_train.index)
df_train_labels = ['DF'] * len(df_train.index)
dhf_test_labels = ['DHF'] * len(dhf_test.index)
df_test_labels = ['DF'] * len(df_test.index)

train_labels = dhf_train_labels + df_train_labels
test_labels = dhf_test_labels + df_test_labels

# Set Probe_Set_ID as index
data.set_index('Probe_Set_ID', inplace=True)

# Gene of interest
gene = "212185_x_at"

# Extract training and testing data for the specific gene
train_dhf_values = data.loc[gene, dhf_train.index]
train_df_values = data.loc[gene, df_train.index]
test_dhf_values = data.loc[gene, dhf_test.index]
test_df_values = data.loc[gene, df_test.index]

gene_train_data = pd.concat([train_dhf_values, train_df_values])
gene_test_data = pd.concat([test_dhf_values, test_df_values])

# Reshape the data to fit the SVM model
gene_train_data = gene_train_data.values.reshape(-1, 1)
gene_test_data = gene_test_data.values.reshape(-1, 1)

# Train the SVM
svm = SVC(kernel='linear', random_state=42)
svm.fit(gene_train_data, train_labels)

# Predict on the test set
test_predictions = svm.predict(gene_test_data)

# Combine predictions with actual labels for comparison
test_results = pd.DataFrame({
    'Actual': test_labels,
    'Predicted': test_predictions
})

# Display classification results
print("Classification Results for Gene 212185_x_at")
print(test_results)

# Visualize the SVM decision boundary
gene_train_values = np.array(gene_train_data).flatten()
train_labels_numeric = [1 if label == 'DHF' else 0 for label in train_labels]

# Create a range of values for plotting decision boundaries
x_min, x_max = gene_train_values.min() - 1, gene_train_values.max() + 1
x_range = np.linspace(x_min, x_max, 500).reshape(-1, 1)

# Predict decision boundaries
decision_boundary = svm.decision_function(x_range)

# Plot the data points and decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(gene_train_values, train_labels_numeric, c=train_labels_numeric, cmap='coolwarm', label='Training Data')
plt.plot(x_range, decision_boundary, color='black', linestyle='-', label='Decision Boundary')
plt.axhline(0, color='gray', linestyle='--', label='Hyperplane')
plt.xlabel('Gene Expression Levels')
plt.ylabel('Class (0 = DF, 1 = DHF)')
plt.title('SVM Decision Boundary for Gene 212185_x_at')
plt.legend()
plt.show()
