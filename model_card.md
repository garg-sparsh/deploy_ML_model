# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Model Developed By: Sparsh
- Model date: 28th April 2024
- Model Version: 1.0.0s
- Model Type: Classifier
- RandomForest Classifier 

## Intended Use
- To predict the salary classification(more than 50k or not) using basic details and qualifications of human beings to determine whether he/she is capable of get loan and by how much 

## Training Data
https://archive.ics.uci.edu/dataset/20/census+income

## Evaluation Data
https://archive.ics.uci.edu/dataset/20/census+income, divide the census data by 80-20

## Metrics
F1_beta: 0.7219796215429404
Precission : 0.6262626262626263
Recall: 0.6707234617985125

## Ethical Considerations
The dataset has only two categories(<50k or >50k), so there are so many professions where people have very different salary ranges, so we should not assume these are the salalries brackets for these occupations.

## Caveats and Recommendations
This dataset is quite old, so we should only use this to train an ML algorithm just for learning purposes
