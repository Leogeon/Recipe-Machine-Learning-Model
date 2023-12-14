# RecipeAnalysis
**Authors**: [Wenbin Jiang](https://github.com/Leogeon), [Jevan Chahal](https://github.com/JevanC)

## Exploratory Data:
- Here is our exploratory data analysis on this dataset: [Recipe Analysis](https://leogeon.github.io/RecipeAnalysis/)
  
## Table of Contents:
- [Framing the Problem](#framingtheproblem)
- [Baseline Model](#baselinemodel)
- [Final Model](#finalmodel)
- [Fairness Analysis](#fairnessanalysis)

## Framing the Problem <a name="framingtheproblem"></a>
### Prediction Problem:
- **Type:** Regression
- **Objective:** Our objective with this project is to create a model that can accurately predict the sugar content of a recipe given other nutritional facts.

### Response Variable:
- **Variable:** Sugar content (measured quantitatively)
- **Reason for Choice:** For our response variable we decided to choose sugar. The reason for this is that sugar is the most important nutritional component and being able to accurately predict the level of sugar in a recipe is very beneficial. From a health and safety standpoint, many people are faced with diabetes, and for such people, knowing the exact sugar content of food can be the difference between life and death. 
  
### Evaluation Metrics:
- **Root Mean Square Error (RMSE):**
  
  - **Chosen Because:** RMSE is a standard metric for regression problems. It measures the average magnitude of the errors between predicted and actual values, giving a sense of how far off predictions are. It's particularly useful in this context because it directly relates to the quantity being predicted (sugar content).
  - **Interpretation:** Lower RMSE values are better as they indicate smaller differences between the predicted and actual values.
- **R² Score:**
  
  - **Chosen Because:** R² (Coefficient of Determination) is a statistical measure that represents the proportion of the variance for the dependent variable that's explained by the independent variables. It gives an insight into the goodness of fit of the model.
  - **Interpretation:** An R² score close to 1 indicates that the model explains a large portion of the variance in the response variable.
- **Cross-Validation Scores:**
  - **Chosen Because:** Cross-validation is a robust method for assessing the generalizability of a model. It helps in understanding how the model performs on different subsets of the dataset.
  - **Interpretation:** Consistent and high scores across folds indicate a well-performing and stable model.

## Baseline Model <a name="baselinemodel"></a>
The baseline model we used is a linear regression model designed to predict the 'sugar' content in food items based on other nutritional information. The model is built using Python's scikit-learn library.
### Features in the Model:
- **Calories:** Quantitative
- **Total Fat:** Quantitative
- **Sodium:** Quantitative
- **Protein:** Quantitative
- **Saturated Fat:** Quantitative
- **Carbohydrates:** Quantitative

All features in this model are quantitative, representing measurable quantities expressed as numerical values. There are no ordinal or nominal features in this model. The data preprocessing involved converting string representations of lists in the 'nutrition' column into actual lists using a custom safe_eval function. Missing values in features and the target variable were filled with their respective means.
  
### Model Pipeline:
The model pipeline consists of two stages:
- **StandardScaler:** This standardizes the features by removing the mean and scaling to unit variance. This step is crucial for linear regression, which is sensitive to the scale of input features.
- **LinearRegression:** A linear regression model that fits a linear equation to the observed data.
  
### Model Evaluation:
- **Mean Squared Error (MSE) and R² Score:** These metrics are used to evaluate the model's performance. The MSE is the average squared difference between the actual and predicted values, while the R² score represents the proportion of variance in the dependent variable that is predictable from the independent variables.
- **Cross-Validation Scores:** The model is also evaluated using 5-fold cross-validation to assess its generalization capability. This involves splitting the dataset into 5 parts, training the model on 4 and testing on the 1 remaining part, and repeating this process 5 times.
  
### Performance:
The performance of the model was assessed using the Mean Squared Error (MSE) and R² Score for both the training and testing datasets, along with cross-validation scores for overall assessment.

| Metric        | Training Results            | Testing Results             |
| ------------- | --------------------------- | --------------------------- |
| MSE           | 102.63865766316138          | 110.07593549277638          |
| R² Score      | 0.7571468808446675          | 0.75714688084466755          |

| CV Fold       | Score                       |
| ------------- | --------------------------- |
| 1             | 0.67128828                  |
| 2             | 0.7337971                   |
| 3             | 0.77311845                  |
| 4             | 0.70509366                  |
| 5             | 0.74905523                  |

- **Root Mean Square Error (RMSE):** The RMSE is approximately 102.63 and 110.07 for testing. This value indicates the standard deviation of the prediction errors or residuals. A lower RMSE is generally better, but since this is only the baseline model, we decided to proceed with it.
- **R² Score:** The R² score for training is about 0.757, and for testing, it is approximately 0.757. This score suggests that a significant proportion of the variance in the sugar content is explained by the model, indicating a good fit to the data.
- **Cross-Validation Scores:** The scores range from approximately 0.671 to 0.773. This variation indicates some fluctuation in the model's performance across different data subsets, but the scores are relatively consistent. The model does not show signs of overfitting and should generalize well to new, unseen data.

## Final Model <a name="finalmodel"></a>
In this section, we refined our model by adding new features and employing Lasso Regression for better prediction accuracy and feature selection.
### Added Features and Their Rationale:
- **Number of Ingredients (n_ingredients):**
  - **Rationale:** More ingredients could indicate a higher likelihood of containing sweeteners, thus affecting the sugar content.
- **Number of Steps (n_steps):**
  - **Rationale:** A greater number of steps may correlate with recipe complexity, potentially impacting sugar content through various cooking processes.
- **Number of Minutes (minutes):**
  - **Rationale:** Longer preparation times, typical in baking, might correlate with sugar-rich foods like desserts.
- **Calories Squared (calories_squared):**
  - **Rationale:** To capture non-linear relationships between calories and sugar content.
    
### Modeling Algorithm and Hyperparameters:
- **Algorithm:** Lasso Regression (Least Absolute Shrinkage and Selection Operator).
- **Best Hyperparameters:** Determined through GridSearchCV, focusing on finding the optimal alpha value for regularization.
- **Hyperparameter Selection Method:** GridSearchCV, which explores multiple parameter combinations to optimize model performance.
  
### Improvement Over Baseline Model:
The final model introduces Lasso regression and additional features, which are expected to enhance prediction accuracy and model interpretability:
- **Final Model:** The final model, with Lasso regression and the added features, likely performs better due to several reasons:
  - **Regularization:** Reduces overfitting and selectively shrinks less critical features to zero.
  - **Feature Selection:**  New features enhance the model's ability to capture complex patterns in the data.
  - **Customized Data Transformation:** Employing a ColumnTransformer for tailored preprocessing.
    
### Performance Metrics:
The model's performance is showcased by a slight reduction in the reduction in RMSE and slight increase of the R² score, suggesting improved prediction accuracy and model.

### Best Parameters:
- **Alpha for Lasso Regression:** The best alpha parameter found is 0.1. In Lasso regression, alpha is the parameter that controls the strength of the regularization. A smaller alpha value means less regularization and a value closer to linear regression. The optimal alpha value of 0.1 suggests that some regularization is beneficial for the model, but not too much, which balances between model complexity and the risk of overfitting.
  
### Performance of the Final Model:
The final model's performance, assessed through training and testing data, along with cross-validation, is as follows:

| Metric        | Training Results            | Testing Results             |
| ------------- | --------------------------- | --------------------------- |
| RMSE           | 99.81431897766402           | 107.0671157234895           |
| R² Score      | 0.7677415984965867          | 0.7702417458530844          |

| CV Fold       | Score                       |
| ------------- | --------------------------- |
| 1             | 0.69369467                  |
| 2             | 0.76144211                  |
| 3             | 0.77577528                  |
| 4             | 0.7589276                   |
| 5             | 0.72730318                  |

### Significance of the Improvements:

- **Root Mean Square Error (RMSE):** The final model achieved an RMSE of approximately 99.98 on the training set and 107.06 on the testing set. While it is a small difference, it is still an improvement compared to the RMSE. A lower RMSE indicates that the model's predictions are, on average, closer to the actual sugar values.
- **R² Score:** Simiarly, the R² saw a small improvement but it is still a relatively good score, indicating a good fit to the data.
  
### Significance of the Improvements:
- **Model Accuracy:** The decrease in RMSE suggests that the final model is a bit more accurate in its predictions.
- **Model Fit:** The R² score improved a bit as well, meaning the model a better fit for the data.
- **Impact of Hyperparameter Tuning and Feature Engineering:** The improvements in these metrics also underscore the effectiveness of our hyperparameter tuning with GridSearchCV and the introduction of new features. The combination of these techniques likely helped in capturing more complex relationships in the data, which were not possible with the baseline model.
  
## Fairness Analysis <a name="fairness-analysis"></a>
### Choice of Groups X and Y:
- **Group X:** High Calorie Group (recipes with calories above median)
- **Group Y:** Low Calorie Group (recipes with calories at or below median)

### Evaluation Metric:
- **Metric Used:** Mean Squared Error (MSE)

### Null and Alternative Hypotheses:
- **Null Hypothesis (H0):** Model is fair, MSE for both groups are similar.
- **Alternative Hypothesis (H1):** Model is unfair, significant difference in MSE between groups.

### Test Statistic and Significance Level:
- **Test Statistic:** Difference in MSE between groups.
- **Significance Level:** Typically 0.05.

### Permutation Test Result:
- **Observed Mean Difference:** 20071.233871398035
- **p-value:** 0.0

### Conclusion:
Reject the null hypothesis at a significance level of 0.05. Indicating a statistically significant difference in model performance between High Calorie and Low-Calorie groups, suggesting potential fairness concerns. This kind of makes sense since we can assume the high-calorie group and the low-calorie group have different nutritional content. Higher calorie groups usually contain higher amounts of sugar content (milkshake vs salad).
