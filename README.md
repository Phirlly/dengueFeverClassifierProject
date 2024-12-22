# Dengue Fever Prognosis Study

## PROBLEM STATEMENT
Dengue infection affects millions worldwide and can often escalate to severe forms such as Dengue Hemorrhagic Fever (DHF). This escalation necessitates early differentiation from Dengue Fever (DF), despite their overlapping clinical presentations. Gene expression profiling during the febrile phase uncovers distinct transcriptional signatures; DHF patients exhibit diminished activation of innate immunity genes alongside heightened expression of apoptosis-related genes. Linear discriminant analysis (LDA) and Support Vector Machine (SVM) models built from the most effective gene pairs demonstrate high accuracy in distinguishing between DHF and DF by employing ANOVA to identify key discriminatory genes. These computational approaches facilitate the early and precise identification of severe cases, which supports timely intervention and optimizes resource allocation. Such bioinformatics-driven strategies advance the field of precision medicine, helping to alleviate both the health and economic burdens of dengue in endemic regions.

## OBJECTIVES
The main objective of this study is to use ANOVA to identify key discriminatory genes from the dengue fever prognosis dataset, which contains gene expression profiles for 1981 genes and clinical outcomes categorized into classical dengue fever (DF), dengue hemorrhagic fever (DHF), and febrile non-dengue cases. Linear discriminant analysis (LDA) and Support Vector Machine (SVM) models built from the most effective gene pairs demonstrate high accuracy in distinguishing between DHF and DF.

## DATASET REFERENCE
The dengue fever prognosis dataset contains gene expression data from peripheral blood mononuclear cells (PBMCs) collected from patients in the early stages of fever. The dataset includes gene expression profiles for 1981 genes and clinical outcomes categorized into classical dengue fever (DF), dengue hemorrhagic fever (DHF), and febrile non-dengue cases. \cite{Nascimento2009}

 ## PROGRAMING LANGUAGE, TOOLS, AND LIBRARIES
 * Python
 * Scikit-learn
 * Numpy
 * Matplotlib
 * Pandas
