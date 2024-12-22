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
