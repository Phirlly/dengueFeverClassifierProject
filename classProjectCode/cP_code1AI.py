#Random Forest Classifier for Dengue Fever Prognosis Dataset
import pandas as pd
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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
dhf_train, dhf_test = train_test_split(dhf_data, test_size=0.2, random_state=42)
df_train, df_test = train_test_split(df_data, test_size=0.2, random_state=42)

# Perform ANOVA test for each Probe_Set_ID
data.set_index('Probe_Set_ID', inplace=True)

anova_results = []
for probe in data.index:
    # Extract gene expression levels for DHF and DF in the training set
    dhf_values = data.loc[probe, dhf_train.index].values
    df_values = data.loc[probe, df_train.index].values
    
    # Perform ANOVA test
    f_stat, p_value = f_oneway(dhf_values, df_values)
    
    # Store the results
    anova_results.append({'Probe_Set_ID': probe, 'F-statistic': f_stat, 'p-value': p_value})

# Convert the results to a DataFrame
anova_results_df = pd.DataFrame(anova_results)

# Sort by p-value (ascending order)
anova_results_df.sort_values(by='p-value', inplace=True)

# Extract top 10 genes based on p-value
top_genes = anova_results_df.head(10)['Probe_Set_ID']

# Extract the values for training and testing DHF and DF patients for these genes
train_dhf_values = data.loc[top_genes, dhf_train.index]
train_df_values = data.loc[top_genes, df_train.index]

test_dhf_values = data.loc[top_genes, dhf_test.index]
test_df_values = data.loc[top_genes, df_test.index]

# Combine them into separate DataFrames for easier interpretation
train_values = pd.concat([train_dhf_values, train_df_values], axis=1, keys=['DHF', 'DF'])
test_values = pd.concat([test_dhf_values, test_df_values], axis=1, keys=['DHF', 'DF'])

# Prepare the training dataset
X_train = train_values.T.reset_index(drop=True)
y_train = ['DHF'] * len(dhf_train) + ['DF'] * len(df_train)

# Prepare the testing dataset
X_test = test_values.T.reset_index(drop=True)
y_test = ['DHF'] * len(dhf_test) + ['DF'] * len(df_test)

# Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test dataset
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display the results
print(f"Random Forest Classifier Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(report)

# Visualize feature importance
feature_importance = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Gene': top_genes,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Gene'], importance_df['Importance'])
plt.xlabel('Importance Score')
plt.ylabel('Genes')
plt.title('Feature Importance for Top Genes')
plt.gca().invert_yaxis()
plt.show()
