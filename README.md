# 📈 Customer Churn Prediction

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Data Science](https://img.shields.io/badge/Data_Science-Python%20%7C%20ML%20%7C%20AI-brightgreen)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue.svg)](https://www.kaggle.com/code/fatiherencetin/telco-customer-churn)

## 🌟 Overview

Customer churn prediction is a critical aspect for businesses aiming to retain their customer base and enhance profitability. This project leverages machine learning techniques to predict whether a customer is likely to churn, enabling proactive retention strategies. By analyzing customer behavior and transactional data, the models identify patterns that indicate the likelihood of churn, providing valuable insights for decision-making.

## 🗂 Table of Contents
- [🌟 Overview](#-overview)
- [📊 Dataset Description](#-dataset-description)
- [🛠 Data Preprocessing](#-data-preprocessing)
- [🤖 Modeling](#-modeling)
  - [CatBoost](#catboost)
  - [LightGBM](#lightgbm)
  - [Random Forest](#random-forest)
  - [Logistic Regression](#logistic-regression)
- [📈 Performance Metrics](#-performance-metrics)
- [🏆 Results](#-results)
- [🔍 Conclusion](#-conclusion)
- [🔮 Future Work](#-future-work)
- [📄 License](#-license)
- [📫 Contact](#-contact)

## 📊 Dataset Description

The dataset used in this project contains customer information, including demographics, account details, and behavioral data. Each record represents a customer, with the target variable indicating whether the customer has churned (`1`) or not (`0`).

The dataset consists of **7043 observations** and **21 variables** with a total size of **977.5 KB**. Below is an overview of the features:

| **Variable**        | **Description**                                                                                         |
|---------------------|-----------------------------------------------------------------------------------------------------|
| `CustomerId`        | Unique ID for each customer.                                                                          |
| `Gender`            | Gender of the customer.                                                                               |
| `SeniorCitizen`     | Indicates whether the customer is a senior citizen (1: Yes, 0: No).                                   |
| `Partner`           | Indicates whether the customer has a partner (Yes, No).                                              |
| `Dependents`        | Indicates whether the customer has dependents (Yes, No).                                             |
| `tenure`            | Number of months the customer has stayed with the company.                                           |
| `PhoneService`      | Indicates whether the customer has a phone service (Yes, No).                                        |
| `MultipleLines`     | Indicates whether the customer has multiple phone lines (Yes, No, No phone service).                 |
| `InternetService`   | Customer's internet service provider (DSL, Fiber optic, No).                                         |
| `OnlineSecurity`    | Indicates whether the customer has online security (Yes, No, No internet service).                   |
| `OnlineBackup`      | Indicates whether the customer has online backup (Yes, No, No internet service).                     |
| `DeviceProtection`  | Indicates whether the customer has device protection (Yes, No, No internet service).                 |
| `TechSupport`       | Indicates whether the customer has technical support (Yes, No, No internet service).                 |
| `StreamingTV`       | Indicates whether the customer has streaming TV service (Yes, No, No internet service).              |
| `StreamingMovies`   | Indicates whether the customer has streaming movies service (Yes, No, No internet service).          |
| `Contract`          | Type of contract the customer has (Month-to-month, One year, Two years).                             |
| `PaperlessBilling`  | Indicates whether the customer uses paperless billing (Yes, No).                                     |
| `PaymentMethod`     | Customer's payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)). |
| `MonthlyCharges`    | Amount charged to the customer on a monthly basis.                                                   |
| `TotalCharges`      | Total amount charged to the customer.                                                                |
| `Churn`             | Indicates whether the customer churned (Yes, No).                                                   |

---

### **Dataset Story**

This dataset contains demographic, account, and service information about customers and is used to predict whether a customer will churn. Each observation represents a customer, and the `Churn` variable serves as the target variable. By analyzing the features, this dataset aims to provide insights into the patterns and factors influencing customer retention.



You can also find this data set in kaggle!

## 🛠 Data Preprocessing

Data preprocessing is a crucial step to ensure the quality and suitability of the data for modeling. The preprocessing steps undertaken include:

1. **Handling Missing Values**: Identifying and imputing or removing missing data to prevent skewed results.
2. **Encoding Categorical Variables**: Converting categorical features into numerical representations using techniques like One-Hot Encoding and Label Encoding.
3. **Feature Scaling**: Standardizing numerical features to ensure that all features contribute equally to the model's performance.
4. **Feature Selection**: Selecting relevant features that have a significant impact on the target variable to improve model efficiency and performance.
5. **Splitting the Dataset**: Dividing the data into training and testing sets to evaluate model performance effectively.

---

## 🤖 Modeling

Four classification algorithms were implemented to predict customer churn: **CatBoost**, **LightGBM**, **Random Forest**, and **Logistic Regression**. Each model underwent hyperparameter tuning using three optimization techniques: **GridSearchCV**, **RandomizedSearchCV**, and **Optuna**.


<table>
  <tr>
  <td valign="top" width="50%" style="padding: 10px;">
  <a name="catboost"></a>  
  <p align="center">
    <img src="https://img.shields.io/badge/CatBoost-00CED1?style=for-the-badge&logo=catboost&logoColor=white" alt="CatBoost" height="30" />
 
  CatBoost is a gradient boosting algorithm that handles categorical features efficiently without extensive preprocessing. It is known for its high performance and ease of use.
      
  - **Hyperparameters Tuned**:
    
    - Iterations
    - Depth
    - Learning Rate
    - L2 Leaf Regularization
    </p> 
    </td>


    <td valign="top" width="50%" style="padding: 10px;">
    <a name="lightgbm"></a>
      
    <p align="center">
      <img src="https://img.shields.io/badge/LightGBM-FF7F50?style=for-the-badge&logo=lightgbm&logoColor=white" alt="LightGBM" height="30" />
    
      
  LightGBM is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithms, used for ranking, classification, and many other machine learning tasks.
      
  - **Hyperparameters Tuned**:
    
    - Number of Estimators
    - Number of Leaves
    - Learning Rate
    - Minimum Child Samples
    </p>
    </td>
  </tr>
  
  <tr>


  <td valign="top" width="50%" style="padding: 10px;">
  <a name="random-forest"></a>
      
  <p align="center">
    <img src="https://img.shields.io/badge/Random_Forest-228B22?style=for-the-badge&logo=tree&logoColor=white" alt="Random Forest" height="30" />
  
      
  Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification tasks.

  - **Hyperparameters Tuned**:
    
    - Number of Estimators
    - Maximum Depth
    - Minimum Samples Split
    - Minimum Samples Leaf
    - Bootstrap
    </p>
  </td>

    
  <td valign="top" width="50%" style="padding: 10px;">
  <a name="logistic-regression"></a>
      
  <p align="center">
    <img src="https://img.shields.io/badge/Logistic_Regression-8A2BE2?style=for-the-badge&logo=logistic-regression&logoColor=white" alt="Logistic Regression" height="30" />
  
      
  Logistic Regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable. It is widely used for its interpretability and efficiency.
      
  - **Hyperparameters Tuned**:
    
    - Penalty
    - Inverse of Regularization Strength (C)
    - L1 Ratio
    </p>
    </td>
  </tr>
</table>


<!--
### CatBoost
  
CatBoost is a gradient boosting algorithm that handles categorical features efficiently without extensive preprocessing. It is known for its high performance and ease of use.

- **Hyperparameters Tuned**:
  - Iterations
  - Depth
  - Learning Rate
  - L2 Leaf Regularization

---

### LightGBM

LightGBM is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithms, used for ranking, classification, and many other machine learning tasks.

- **Hyperparameters Tuned**:
  - Number of Estimators
  - Number of Leaves
  - Learning Rate
  - Minimum Child Samples

---

### Random Forest

Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes for classification tasks.

- **Hyperparameters Tuned**:
  - Number of Estimators
  - Maximum Depth
  - Minimum Samples Split
  - Minimum Samples Leaf
  - Bootstrap

---

### Logistic Regression

Logistic Regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable. It is widely used for its interpretability and efficiency.

- **Hyperparameters Tuned**:
  - Penalty
  - Inverse of Regularization Strength (C)
  - L1 Ratio
-->

---

## 📈 Performance Metrics

The models were evaluated using the following metrics:
<!--
1. **Accuracy**:
   - **Definition**: The ratio of correctly predicted instances to the total instances.
   - **Importance**: Provides a general measure of how often the classifier is correct.

2. **Area Under the ROC Curve (AUC)**:
   - **Definition**: Measures the ability of the classifier to distinguish between classes.
   - **Importance**: A higher AUC indicates better model performance in ranking positive instances higher than negative ones.

3. **Recall (Sensitivity)**:
   - **Definition**: The ratio of correctly predicted positive instances to all actual positive instances.
   - **Importance**: Reflects the model's ability to identify all relevant cases (churned customers).

4. **Precision**:
   - **Definition**: The ratio of correctly predicted positive instances to all instances predicted as positive.
   - **Importance**: Indicates the accuracy of positive predictions, reducing false positives.

5. **F1-Score**:
   - **Definition**: The harmonic mean of Precision and Recall.
   - **Importance**: Provides a balance between Precision and Recall, especially useful when seeking a balance between the two metrics.

-->

| **Metric**      | **Description**                                                                                 | **Importance**                                                                                     |
|------------------|-----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| **Accuracy**     | Ratio of correctly predicted instances to total instances.                                     | General measure of prediction correctness.                                                        |
| **AUC (ROC)**    | Measures the model’s ability to rank positive instances higher than negative ones.            | High AUC indicates excellent class separability.                                                  |
| **Precision**    | Proportion of true positive predictions to all positive predictions.                          | Indicates model reliability in positive class predictions.                                         |
| **Recall (Sensitivity)**       | Ratio of true positive predictions to all actual positive instances.                          | Reflects model’s capability to identify all churned customers.                                    |
| **F1-Score**     | Harmonic mean of Precision and Recall.                                                        | Balances the trade-off between Precision and Recall for imbalanced datasets.                      |

---

## 🏆 Results


### **CatBoost with GridSearch**

| **Class** | **Precision** | **Recall** | **F1-Score** |
|-----------|---------------|------------|--------------|
| Churn=0   | 0.84          | 0.91       | 0.88         |
| Churn=1   | 0.67          | 0.52       | 0.59         |
| **Accuracy** |             |            | **0.81**     |

**ROC AUC Score**: 0.8538
<!--
### **CatBoost**
- **Accuracy**: 79.8% to 80.3%
- **AUC**: 0.8396 to 0.8538
- **Recall**: 51.36% to 52.28%
- **Precision**: 65.32% to 66.53%
- **F1-Score**: 0.5747 to 0.5851
-->
*CatBoost consistently demonstrates strong performance across all metrics. With the highest AUC among the evaluated models, it showcases superior ability to distinguish between churned and non-churned customers. The balanced precision and recall indicate effective identification of churned customers while maintaining reasonable prediction accuracy.*


---

### **LightGBM with GridSearch**

| **Class** | **Precision** | **Recall** | **F1-Score** |
|-----------|---------------|------------|--------------|
| Churn=0   | 0.85          | 0.89       | 0.87         |
| Churn=1   | 0.63          | 0.54       | 0.58         |
| **Accuracy** |             |            | **0.80**     |

**ROC AUC Score**: 0.8405
<!--
### **LightGBM**
- **Accuracy**: 79.16% to 80.33%
- **AUC**: 0.8273 to 0.8405
- **Recall**: 51.04% to 52.54%
- **Precision**: 63.57% to 65.27%
- **F1-Score**: 0.5658 to 0.5941
-->
*LightGBM exhibits robust Accuracy and AUC, indicating excellent class separation and predictive performance. Its balanced recall and precision highlight its capability to accurately identify churned customers while maintaining good prediction precision. Hyperparameter tuning further enhances its performance, making it a reliable model for churn prediction.*

---

### **Random Forest with GridSearch**

| **Class** | **Precision** | **Recall** | **F1-Score** |
|-----------|---------------|------------|--------------|
| Churn=0   | 0.84          | 0.90       | 0.87         |
| Churn=1   | 0.64          | 0.50       | 0.56         |
| **Accuracy** |             |            | **0.80**     |

**ROC AUC Score**: 0.8467
<!--
### **Random Forest**
- **Accuracy**: 79.43% to 80.33%
- **AUC**: 0.8280 to 0.8492
- **Recall**: 49.81% to 52.28%
- **Precision**: 64.81% to 68.00%
- **F1-Score**: 0.5630 to 0.5851
-->
*Random Forest demonstrates strong Accuracy and AUC, reflecting good overall performance and class separation. With higher precision, it effectively minimizes false positives, though recall remains moderate, indicating that some churned customers may not be identified. Hyperparameter tuning through GridSearchCV, RandomizedSearchCV, and Optuna further refines its performance, enhancing its reliability in churn prediction.*

---

### **Logistic Regression with GridSearch**

| **Class** | **Precision** | **Recall** | **F1-Score** |
|-----------|---------------|------------|--------------|
| Churn=0   | 0.83          | 0.92       | 0.87         |
| Churn=1   | 0.68          | 0.50       | 0.58         |
| **Accuracy** |             |            | **0.81**     |

**ROC AUC Score**: 0.8403
<!--
### **Logistic Regression**
- **Accuracy**: 75.0% to 81.0%
- **AUC**: 0.8403 to 0.8518
- **Recall**: 50.0% to 80.0%
- **Precision**: 51.0% to 68.0%
- **F1-Score**: 0.58 to 0.81
-->
*Logistic Regression achieves high AUC scores, indicating excellent ability to distinguish between churned and non-churned customers. The model maintains a balanced precision and recall, especially after hyperparameter tuning, ensuring reliable identification of churned customers while minimizing false positives. Its interpretability and robustness make it a valuable model for churn prediction tasks.*

---

### **Summary of Performance Metrics**

| **Model**             | **Accuracy** | **AUC**  | **Recall (Churn=1)** | **Precision (Churn=1)** | **F1-Score (Churn=1)** |
|-----------------------|--------------|----------|----------------------|-------------------------|------------------------|
| **CatBoost**          | 0.81         | 0.8538   | 0.52                 | 0.67                    | 0.59                   |
| **LightGBM**          | 0.80         | 0.8405   | 0.54                 | 0.63                    | 0.58                   |
| **Random Forest**     | 0.80         | 0.8467   | 0.50                 | 0.64                    | 0.56                   |
| **Logistic Regression** | 0.81       | 0.8403   | 0.50                 | 0.68                    | 0.58                   |


## 🔍 Conclusion

Among the evaluated models, **CatBoost** and **Logistic Regression** emerge as the top performers, boasting the highest **AUC** values and balanced **Precision** and **Recall**. **LightGBM** and **Random Forest** also demonstrate robust performance, making them reliable alternatives. While **Logistic Regression** offers interpretability and robustness, **CatBoost** provides superior class separation and predictive accuracy. **LightGBM** and **Random Forest** are strong contenders with their ensemble learning capabilities.

For deployment in churn prediction tasks, **CatBoost** stands out due to its high AUC and balanced performance metrics, ensuring both accurate identification of churned customers and reliable prediction reliability. However, the choice between these models may also consider factors such as interpretability, computational efficiency, and specific business requirements.

**Emphasizing the importance of AUC alongside Precision and Recall** offers a comprehensive evaluation of model performance, ensuring both the identification of churned customers and the reliability of predictions. This holistic approach ensures that the selected model not only performs well statistically but also aligns with the practical needs of minimizing customer loss and optimizing retention strategies.

## 🔮 Future Work

- **Feature Engineering:** Explore additional feature engineering techniques to enhance model performance.
- **Model Ensemble:** Combine multiple models to create an ensemble for improved prediction accuracy.
- **Deployment:** Develop a web application or API for real-time churn prediction.
- **Explainability:** Implement model interpretability tools like SHAP to understand feature contributions.

## 📄 License

This project is licensed under the MIT License.

Feel free to contribute to this project by submitting issues or pull requests. Your contributions are welcome!

## 📫 Contact

<p align="left">
  <a href="www.linkedin.com/in/fatih-eren-cetin" target="_blank"  rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/LinkedIn-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" height="30" />
  </a>
  
  <a href="https://medium.com/@fecetinn" target="_blank"  rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white" alt="Medium" height="30" />
  </a>
  
  <a href="https://www.kaggle.com/fatiherencetin" target="_blank"  rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle" height="30" />
  </a>
</p>

## 📚 Additional Resources

- [CatBoost Documentation](https://catboost.ai/docs/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Random Forest in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Logistic Regression in scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

---

Feel free to explore the code, experiment with different models, and contribute to enhancing this project!
