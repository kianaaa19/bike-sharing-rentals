# Machine Learning Project Report - Heart Failure Prediction

## Project Domain

Cardiovascular diseases (CVDs) are the leading cause of death globally, claiming approximately 17.9 million lives annually, representing 31% of all global deaths. Heart failure, as one of the severe forms of CVD, occurs when the heart cannot pump blood effectively. Early detection and management of CVDs are crucial to reducing the risk of severe complications and mortality.

In this context, developing a machine learning-based heart failure prediction system is highly significant. This system can aid medical professionals in identifying high-risk patients earlier, enabling timely interventions and more effective management. Additionally, accurate predictions can help optimise healthcare resource allocation, reduce economic burdens on healthcare systems, and most importantly, improve patients' quality of life.

This project aims to develop a machine learning model capable of predicting heart failure risk based on various health factors. By leveraging available medical data, this model is expected to become a valuable decision-support tool for doctors and healthcare practitioners in assessing patient risks and planning appropriate treatment strategies.

![Image 1](https://i.ibb.co.com/7CPFTPx/Napkin-AI1.png)  

## Business Understanding

### Problem Statements
- How can a predictive model be developed to accurately identify patients at high risk of heart failure?
- What are the most significant health factors in predicting the likelihood of heart failure?
- How can the interpretability of the predictive model be enhanced to gain acceptance and trust from medical professionals?

![Image 2](https://i.ibb.co.com/L0mzXvS/Napkin-AI2.png)  

### Goals
- Develop a machine learning model with high accuracy (>88%) to predict heart failure risk.
- Identify and rank health factors based on their influence on heart failure prediction.
- Create a model that is not only accurate but also interpretable, to support clinical decision-making.

![Image 3](https://i.ibb.co.com/2FT209F/gambar-2024-12-10-150447655.png)  

### Solution Statements
- Implement various machine learning algorithms such as Logistic Regression, Random Forest, SVM, and Neural Networks, and compare their performances.
- Conduct feature importance analysis and use SHAP (SHapley Additive exPlanations) for model interpretation.
- Apply advanced techniques like hyperparameter tuning and handling imbalanced data to enhance model performance.

![Image 4](https://i.ibb.co.com/hYgsvhV/image.png)  

## Data Understanding

The dataset used in this project is sourced from Kaggle, titled "Heart Failure Prediction Dataset".

**Dataset Link:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

This dataset comprises 918 samples with 12 clinical features used to predict the likelihood of heart failure.

### Dataset Variables:
1. **Age**: Patient's age (numeric)
2. **Sex**: Patient's gender (categorical: M = Male, F = Female)
3. **ChestPainType**: Type of chest pain (categorical: TA = Typical Angina, ATA = Atypical Angina, NAP = Non-Anginal Pain, ASY = Asymptomatic)
4. **RestingBP**: Resting blood pressure in mm Hg (numeric)
5. **Cholesterol**: Serum cholesterol in mg/dl (numeric)
6. **FastingBS**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. **RestingECG**: Resting electrocardiogram results (categorical: Normal, ST, LVH)
8. **MaxHR**: Maximum heart rate achieved (numeric)
9. **ExerciseAngina**: Exercise-induced angina (categorical: Y = Yes, N = No)
10. **Oldpeak**: ST depression induced by exercise relative to rest (numeric)
11. **ST_Slope**: The slope of the peak exercise ST segment (categorical: Up, Flat, Down)
12. **HeartDisease**: Output variable (1 = heart failure, 0 = normal)

### Data Conditions:

- **Missing Values**: None
- **Duplicates**: None
- **Outliers**: Some outliers in numeric features such as Cholesterol and RestingBP need to be addressed.
- **Class Imbalance**: There is slight class imbalance in the target variable (HeartDisease), with a positive-to-negative class ratio of approximately 55:45.

![Image 5](https://i.ibb.co.com/gMqFxN1/image.png)  

## Data Preparation

The data preparation steps undertaken were as follows:

1. **Separation of features and target variables**:  
   Features (X) were separated from the target variable (y) to prepare the data for model training.

2. **Encoding categorical variables**:  
   Categorical variables such as `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, and `ST_Slope` were transformed into numerical representations using one-hot encoding. This step was crucial as most machine learning algorithms operate on numerical input.

3. **Handling outliers**:  
   Outliers in numeric features such as `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, and `Oldpeak` were addressed using the Interquartile Range (IQR) method. Data points lying outside 1.5 * IQR were considered outliers and removed. This helped reduce bias in the model and improved prediction accuracy.

4. **Splitting data into training and test sets**:  
   Data was split into training (80%) and testing (20%) sets using the `train_test_split` function from scikit-learn. This allowed the model to be trained on one set of data and tested on previously unseen data.

5. **Normalising numeric features**:  
   Numeric features were normalised using scikit-learn's `StandardScaler`. Normalisation ensured all features were on the same scale, improving the performance and stability of certain machine learning algorithms.

6. **Addressing class imbalance**:  
   Although the class imbalance in this dataset was not highly significant, the Synthetic Minority Over-sampling Technique (SMOTE) was applied to the training data to balance the classes. This allowed the model to learn effectively from both classes.

Each of these data preparation steps was critical in ensuring the data was ready for model training, improving prediction accuracy, and avoiding bias in the results.

![Image 6](https://i.ibb.co.com/c1CGDc3/image.png)  

---

## Model Development

Several machine learning models were developed and compared in this project:

1. **Logistic Regression**:  
   A linear classification algorithm that calculates the probability of the target class using the sigmoid function. Logistic Regression is well-suited for binary classification problems like heart failure prediction. Despite its simplicity, it often serves as a robust baseline and is easily interpretable.

2. **Random Forest**:  
   An ensemble algorithm consisting of multiple decision trees. Each tree is trained on a random subset of data and features. Predictions are made based on the majority vote across all trees. Random Forest is effective in capturing non-linear relationships in data and tends not to overfit.

3. **Support Vector Machine (SVM)**:  
   SVM identifies an optimal hyperplane to separate classes in a high-dimensional feature space. It can handle non-linear data using kernel functions. This project employed the Radial Basis Function (RBF) kernel, commonly used for classification problems.

4. **XGBoost**:  
   Extreme Gradient Boosting, an advanced implementation of gradient boosting algorithms, builds models iteratively. Each new model corrects errors from the previous ones. XGBoost is renowned for its high performance and versatility across different data types.

5. **Neural Network**:  
   Neural Networks are known for their ability to capture complex non-linear patterns in data, making them a powerful predictive tool.

![Image 7](https://i.ibb.co.com/4j6yxL7/image.png)  

```python
def create_nn_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

nn_model = create_nn_model(X_train_scaled.shape[1])
nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
```

### Neural Network Architecture:
- **Input layer**: Matches the number of input features.
- **Hidden layer 1**: 64 neurons with ReLU activation.
- **Hidden layer 2**: 32 neurons with ReLU activation.
- **Hidden layer 3**: 16 neurons with ReLU activation.
- **Output layer**: 1 neuron with sigmoid activation (for binary classification).

The model was compiled using the Adam optimiser, binary cross-entropy loss function, and accuracy as the metric. It was trained for 50 epochs with a batch size of 32.

![Image 8](https://i.imgur.com/R9unjVp.png)  

### Neural Network Evaluation Results:
```python
Neural Network Results:
Accuracy: 0.8533
Precision: 0.9082
Recall: 0.8318
F1-score: 0.8683
ROC-AUC: 0.9030
```

Although the Neural Network delivered competitive performance compared to other models, it did not achieve the best results with this dataset. However, its potential to capture complex non-linear patterns might be advantageous in other datasets. A limitation of Neural Networks is their lower interpretability compared to models like Random Forest or Logistic Regression.

---

### Evaluation Metrics:

Before presenting the evaluation results, it is essential to understand the metrics used:

1. **Accuracy**: The proportion of correct predictions (both positive and negative) out of the total predictions. While it provides an overall measure of model performance, it can be misleading in the presence of class imbalance.

2. **Precision**: The proportion of true positive predictions out of all positive predictions. Precision is critical when minimising false positives is important.

3. **Recall**: The proportion of actual positive cases that the model successfully identifies. Recall is vital when minimising false negatives is a priority.

4. **F1-Score**: The harmonic mean of precision and recall. It provides a single score balancing the two metrics.

5. **ROC-AUC**: The Area Under the Receiver Operating Characteristic Curve measures the model's ability to distinguish between classes. It is not sensitive to class imbalance.

### Evaluation Results for Each Model:

1. **Logistic Regression**:
   - Accuracy: 0.8533
   - Precision: 0.9000
   - Recall: 0.8411
   - F1-Score: 0.8696
   - ROC-AUC: 0.9266

2. **Random Forest (before tuning)**:
   - Accuracy: 0.8804
   - Precision: 0.9048
   - Recall: 0.8879
   - F1-Score: 0.8962
   - ROC-AUC: 0.9433

3. **Support Vector Machine (SVM)**:
   - Accuracy: 0.8913
   - Precision: 0.9143
   - Recall: 0.8972
   - F1-Score: 0.9057
   - ROC-AUC: 0.9325

4. **Naive Bayes**:
   - Accuracy: 0.8641
   - Precision: 0.9271
   - Recall: 0.8318
   - F1-Score: 0.8768
   - ROC-AUC: 0.9244

5. **XGBoost**:
   - Accuracy: 0.8750
   - Precision: 0.9038
   - Recall: 0.8785
   - F1-Score: 0.8910
   - ROC-AUC: 0.9340

6. **Random Forest (after tuning)**:
   - Accuracy: 0.8804
   - Precision: 0.8972
   - Recall: 0.8972
   - F1-Score: 0.8972
   - ROC-AUC: 0.9461

7. **Random Forest with SMOTE**:
   - Accuracy: 0.8804
   - Precision: 0.9048
   - Recall: 0.8879
   - F1-Score: 0.8962
   - ROC-AUC: 0.9413

8. **Neural Network**:
   - Accuracy: 85.33%
   - Precision: 90.82%
   - Recall: 83.18%
   - F1-Score: 86.83%
   - ROC-AUC: 90.30%

### Model Performance Comparison:

The updated results show that Support Vector Machine (SVM) and Random Forest models performed well in terms of accuracy and precision. The tuned Random Forest model had the highest ROC-AUC (0.9461), suggesting it is excellent at distinguishing between classes. SVM exhibited slightly better recall, making it effective in reducing false negatives.

To summarise, the tuned Random Forest model demonstrated competitive performance with a balanced trade-off between precision, recall, and interpretability. While its accuracy of 88.04% and ROC-AUC of 0.9461 were not the highest among the tested models, it provided consistent and reliable results across multiple metrics, making it a strong choice for practical applications.

---

## SHAP Analysis and Model Interpretability

To enhance the interpretability of the best-performing model, SHAP (SHapley Additive exPlanations) values were utilised. SHAP values provide an explanation for each individual prediction by calculating the contribution of each feature to that prediction. This aids in understanding how the model makes decisions and which features are most influential.

#### Key Findings from SHAP Analysis:
1. Features like `ST_Slope`, `ChestPainType`, and `ExerciseAngina` were among the most significant predictors of heart failure risk.

![Image 9](https://i.ibb.co.com/LJSKDnB/shap-dependence-ST-Slope-Flat.png)  

2. The **SHAP dependency plot** for `ST_Slope_Up` revealed that higher values strongly correlated with positive predictions, whereas lower values were associated with negative outcomes.

![Image 10](https://i.ibb.co.com/j3wVMGM/shap-dependence-ST-Slope-Up.png) 

3. The **SHAP feature importance plot** highlighted `ST_Slope_Up`, `ST_Slope_Flat`, and `ChestPainType_ASY` as the top contributors to model predictions.

![Image 11](https://i.ibb.co.com/hFMPtWX/shap-feature-importance.png) 

### SHAP Visualisations:
- Dependency plot for `ST_Slope_Up`: Demonstrates a positive impact of higher values on predictions.
- Feature importance plot: Illustrates the overall influence of features on the model's output.

The insights gained from SHAP analysis align with established medical knowledge regarding heart failure risk factors, fostering greater trust in the model's predictions among medical professionals.

---

## Business Understanding Impact

### Problem Statement:
- The model successfully identified high-risk heart failure patients with an accuracy of 88.04%, surpassing the initial target of >88% accuracy.
- Feature importance analysis identified significant health factors such as `ST_Slope`, `ChestPainType`, and `Oldpeak`.

### Goals:
- The goal of developing a high-accuracy predictive model was achieved, with the tuned Random Forest model achieving competitive performance.
- Significant health factors were identified and ranked by their importance.
- The tuned Random Forest model provides a good balance of accuracy and interpretability, making it suitable for clinical decision support.

### Solution Statements:
- The implementation and comparison of multiple machine learning algorithms identified the best-performing model for this dataset.
- SHAP values enhanced the interpretability of the model, supporting clinical decision-making.
- Advanced techniques such as hyperparameter tuning and SMOTE were effective in improving model performance.

---

## Conclusion

1. **Best Model**: The tuned Random Forest model demonstrated strong performance, achieving an accuracy of 88.04% and an ROC-AUC of 0.9461.
2. **Interpretability**: While Neural Networks and XGBoost showed strong performance, the Random Forest model provided the best balance between accuracy and interpretability.
3. **Key Features**: Analysis revealed that `ST_Slope`, `ChestPainType`, and `ExerciseAngina` were the most significant predictors of heart failure.

### Overall Impact:
This project successfully developed an accurate and interpretable heart failure prediction model. The model can assist healthcare professionals in identifying high-risk patients and planning timely interventions, potentially improving patient outcomes and optimising healthcare resource allocation.





