# Dengue Fever Prognosis Using Machine Learning

## Authors  
**Kshitij Kadam** – Texas A&M University (Department of Computer Science and Engineering)  
Email: kkadam3@tamu.edu  
**Adekola Okunola** – Texas A&M University (Department of Electrical and Computer Engineering)  
Email: phirlly@tamu.edu  

## Overview
This project aims to differentiate between **Dengue Fever (DF)** and **Dengue Hemorrhagic Fever (DHF)** using gene expression data. Advanced statistical and machine learning techniques, including **ANOVA, Support Vector Machine (SVM), Random Forest Classifier, and Linear Discriminant Analysis (LDA)**, are used to classify patients accurately based on gene expression data.

## Dataset
The dataset consists of gene expression profiles obtained from peripheral blood mononuclear cells (PBMCs) of **26 patients**, covering **1981 genes**. The classification is based on the following labels:
- **DF (Dengue Fever)**
- **DHF (Dengue Hemorrhagic Fever)**

The dataset reference is included in the final report.

## Methodology

### 1. Data Preprocessing
- Load the dataset (`Dengue_Fever_Prognosis_Dataset.csv`).
- Identify columns corresponding to **DHF** and **DF** cases.
- Transpose the dataset for easier processing.
- Split the dataset into **training (80%)** and **testing (20%)** sets.

### 2. Feature Selection with ANOVA
- Apply **Analysis of Variance (ANOVA)** to identify genes with significant differential expression.
- Select the **top 10 genes** with the lowest p-values.

### 3. Classification Models
#### **a) Random Forest Classifier** (See `CP_Code1AI.py`)
- Train a **Random Forest model** using the top selected genes.
- Evaluate the model's performance using **accuracy score** and **classification report**.
- Identify and visualize important features.

#### **b) Support Vector Machine (SVM)** (See `CP_Code1AII.py` & `CP_Code1AV.py`)
- Train **SVM classifiers** for different gene pairs.
- Visualize decision boundaries using **scatter plots**.
- Evaluate classifiers based on **accuracy and misclassification analysis**.

#### **c) Linear Discriminant Analysis (LDA)** (See `CP_Code1AIII.py` & `CP_Code1AIV.py`)
- Apply **LDA** to classify DF and DHF cases based on gene expression data.
- Evaluate performance metrics such as **accuracy and misclassification analysis**.
- Visualize LDA decision boundaries for selected gene pairs.

### 4. Model Evaluation
- The top classifiers are selected based on **accuracy**.
- Misclassified samples are analyzed to identify model limitations.
- Both **SVM and LDA classifiers** achieved **high accuracy (1.0) on selected gene pairs**.

## Results
The analysis identified the **most discriminatory genes** and their impact on classifying DF vs. DHF. The models showed high accuracy and strong separation of classes, as demonstrated by the decision boundary visualizations.

### **Key Findings:**
- **ANOVA** was successful in identifying genes with significant expression differences.
- **SVM & LDA classifiers** performed exceptionally well with **1.0 accuracy** in top gene pairs.
- **Random Forest** identified critical gene importance for classification.

## Project Files
| File Name | Description |
|-----------|-------------|
| `CP_Code1A.py` | ANOVA-based feature selection for top discriminatory genes |
| `CP_Code1AI.py` | Random Forest Classifier for classification |
| `CP_Code1AII.py` | SVM model for individual gene classification |
| `CP_Code1AIII.py` | LDA classifier for gene pair analysis |
| `CP_Code1AIV.py` | LDA classifier with decision boundary visualization |
| `CP_Code1AV.py` | Advanced SVM classifier with iterative gene selection |
| `Final Report.pdf` | Detailed report covering research, methodology, results, and references |

 ## PROGRAMING LANGUAGE, TOOLS, AND LIBRARIES
 * Python
 * Scikit-learn
 * Numpy
 * Matplotlib
 * Pandas

## Future Work
- Validate models with **larger and more diverse datasets**.
- Explore **deep learning techniques** for improved classification.
- Integrate findings into a **real-time clinical decision-support system**.

## Acknowledgments
This project was completed as part of research work at **Texas A&M University**. We appreciate the contributions of researchers in the field of bioinformatics and dengue prognosis.

## References
```bibtex
@article{Nascimento2009,
    author = {Nascimento, E. and Abath, F. and Calzavara, C. and Gomes, A. and Acioli, B. and Brito, C. and Cordeiro, M. and Silva, A. and Andrade, C. M. R. and Gil, L. and Junior, U. B.-N. E. M.},
    title = {Gene expression profiling during early acute febrile stage of dengue infection can predict the disease outcome},
    journal = {PLoS ONE},
    volume = {4},
    number = {11},
    pages = {e7892},
    year = {2009},
    doi = {10.1371/journal.pone.0007892}
}

@book{BragaNeto2020,
    author = {Ulisses Braga-Neto},
    title = {Fundamentals of Pattern Recognition and Machine Learning},
    publisher = {Springer Nature Switzerland AG},
    year = {2020},
    isbn = {978-3-030-27655-3},
    doi = {10.1007/978-3-030-27656-0}
}

@misc{WHO2023,
    author = {{World Health Organization}},
    title = {Dengue and Severe Dengue: Fact Sheet},
    year = {2023},
    note = {Retrieved December 2, 2024, from \url{https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue}}
}

@article{Liu2022,
    author = {Liu, Y. E. and Saul, S. and Rao, A. M. and others},
    title = {An 8-gene machine learning model improves clinical prediction of severe dengue progression},
    journal = {Genome Medicine},
    volume = {14},
    number = {33},
    year = {2022},
    doi = {10.1186/s13073-022-01034-w}
}
