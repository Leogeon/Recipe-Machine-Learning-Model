# RecipeAnalysis
**Authors**: [Wenbin Jiang](https://github.com/Leogeon), [Jevan Chahal](https://github.com/JevanC)

## Table of Contents
- [Framing the Problem](#framingtheproblem)
- [Baseline Model](#baselinemodel)
- [Final Model](#finalmodel)
- [Fairness Analysis](#fairnessanalysis)

## Framing the Problem <a name="framingtheproblem"></a>
### Prediction Problem:
- **Type:** Regression
- **Objective:** Predicting the 'sugar' content in food items based on their nutritional information.

### Response Variable:
- **Variable:** Sugar content (measured quantitatively)
- **Reason for Choice:** You chose 'sugar' as the response variable to predict its value based on other nutritional information like calories, total fat, sodium, protein, saturated fat, and carbohydrates. This choice makes sense if the goal is to understand or manage sugar levels in dietary planning, which is a significant consideration in nutrition and health-related fields.
  
### Evaluation Metrics:
- **Root Mean Square Error (RMSE):**
  
  - **Chosen Because:** RMSE is a standard metric for regression problems. It measures the average magnitude of the errors between predicted and actual values, giving a sense of how far off predictions are. It's particularly useful in your context because it directly relates to the quantity being predicted (sugar content).
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

All features in this model are quantitative, as they represent measurable quantities and are expressed as numerical values. There's no mention of ordinal or nominal features in your description.
### Data Preprocessing:
- Handling of String Representations: The 'nutrition' column, initially containing string representations of lists, is converted into actual lists using a custom safe_eval function.
- Handling of Missing Values: Missing values in features and target variables are filled with their respective means.
  
### Model Pipeline:
The model pipeline consists of two stages:
- **StandardScaler:** This is used for feature scaling, which standardizes the features by removing the mean and scaling to unit variance. This is important for linear regression as it is sensitive to the scale of input features.
- **LinearRegression:** A simple linear regression model that fits a linear equation to the observed data.
  
### Model Evaluation:
- **Mean Squared Error (MSE) and R² Score:** These metrics are used to evaluate the model's performance. The MSE is the average squared difference between the actual and predicted values, while the R² score represents the proportion of variance in the dependent variable that is predictable from the independent variables.
- **Cross-Validation Scores:** The model is also evaluated using 5-fold cross-validation to assess its generalization capability. This involves splitting the dataset into 5 parts, training the model on 4 and testing on the 1 remaining part, and repeating this process 5 times.
  
### Performance:
- **Root Mean Square Error (RMSE):** The RMSE is approximately 103.37. This value indicates the standard deviation of the prediction errors or residuals, which are the differences between observed and predicted values. In general, a lower RMSE is better, but the acceptability of this value greatly depends on the context of the data and the scale of the 'sugar' variable. If the range of sugar content is large, an RMSE of 103 might be acceptable, but if the range is smaller, this could be considered high.
- **R² Score:** The R² score of about 0.774 suggests that approximately 77.4% of the variance in the sugar content is explained by your model. This is generally considered a good score, indicating that your model has a strong fit to the data.
- **Cross-Validation Scores:** The cross-validation scores range from approximately 0.671 to 0.773. This range indicates some variation in the model's performance across different subsets of the data, but the scores are relatively consistent. None of the scores are exceptionally low, which is a good sign that your model is not overfitting to a specific part of the data and should generalize well to new, unseen data.

## Final Model <a name="finalmodel"></a>
### Added Features and Their Rationale:
- **Number of Ingredients (n_ingredients):**
  - **Rationale:** The number of ingredients in a recipe can be a proxy for its complexity or diversity. Recipes with more ingredients might have a higher likelihood of containing sweeteners, affecting the sugar content. This feature could capture variations in sugar content that aren't directly related to the other nutritional values.
- **Number of Steps (n_steps):**
  - **Rationale:** Similar to n_ingredients, the number of steps in a recipe might correlate with its complexity. More complex recipes might undergo processes that either add or reduce sugar content, like caramelization or fermentation.
- **Fat-Sugar Interaction (fat_sugar_interaction):**
  - **Rationale:** The interaction term between total fat and sugar content could capture the combined effect of these nutrients on the overall nutritional profile of a recipe. In culinary practice, fat and sugar often work together in recipes to achieve certain textures and flavors, potentially indicating richer or more indulgent dishes.
- **Sodium Squared (sodium_squared):**
  - **Rationale:** Squaring the sodium content could capture non-linear effects of sodium on sugar content. For instance, recipes with very high or very low sodium might have specific sugar profiles not linearly related to the sodium content.
    
### Modeling Algorithm and Hyperparameters:
- **Algorithm:** Lasso Regression (Least Absolute Shrinkage and Selection Operator)
- **Best Hyperparameters:** The best hyperparameters, determined through GridSearchCV, were the optimal alpha value for regularization in Lasso Regression. The exact value would be specified in best_params.
- **Hyperparameter Selection Method:** GridSearchCV systematically works through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance. It's a thorough way to optimize the model.
  
### Improvement Over Baseline Model:
- **Baseline Model:** The initial model used a simple linear regression without additional features or interaction terms.
- **Final Model:** The final model, with Lasso regression and the added features, likely performs better due to several reasons:
  - **Regularization:** Lasso regression includes a regularization term that helps prevent overfitting, especially important when adding more features.
  - **Feature Selection:** Lasso can also perform feature selection by shrinking coefficients of less important features to zero.
  - **Handling Non-linearity:** The squared and interaction terms help capture non-linear relationships that a simple linear regression could miss.
  - **Customized Data Transformation:** The use of a ColumnTransformer allows different preprocessing for different features, tailored to their specific distributions and relationships.
    
### Performance Metrics:
The improvement in the model's performance is evidenced by the decrease in RMSE (Root Mean Square Error) and the increase in the R² score from the baseline to the final model. A lower RMSE indicates better prediction accuracy, while a higher R² score indicates that a greater proportion of variance in the dependent variable is explained by the independent variables in the model.

### Best Parameters:
- **Alpha for Lasso Regression:** The best alpha parameter found is 0.1. In Lasso regression, alpha is the parameter that controls the strength of the regularization. A smaller alpha value means less regularization and a value closer to linear regression. The optimal alpha value of 0.1 suggests that some regularization is beneficial for your model, but not too much, which balances between model complexity and the risk of overfitting.
  
### Performance of the Final Model:
- **Root Mean Square Error (RMSE):** The final model achieved an RMSE of approximately 98.53. This is an improvement compared to the RMSE of 103.37 from the baseline model. A lower RMSE indicates that the model's predictions are, on average, closer to the actual sugar values.
- **R² Score:** The R² score increased to about 0.794. This improvement from the baseline model's 0.774 indicates that the final model explains a higher proportion of the variance in the sugar content. An R² score closer to 1 is generally desirable, showing that the model has a good fit to the data.
  
### Significance of the Improvements:
- **Model Accuracy:** The decrease in RMSE suggests that the final model is more accurate in its predictions.
- **Model Fit:** The increase in R² indicates a better fit to the data, meaning the model is more effective at capturing the underlying relationships between the variables.
- **Impact of Hyperparameter Tuning and Feature Engineering:** The improvements in these metrics also underscore the effectiveness of your hyperparameter tuning with GridSearchCV and the introduction of new features. The combination of these techniques likely helped in capturing more complex relationships in the data, which were not possible with the baseline model.
  
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
Reject the null hypothesis at a significance level of 0.05. Indicating a statistically significant difference in model performance between High Calorie and Low Calorie groups, suggesting potential fairness concerns.
