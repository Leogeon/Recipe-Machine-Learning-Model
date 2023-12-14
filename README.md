# RecipeAnalysis
**Authors**: [Wenbin Jiang](https://github.com/Leogeon), [Jevan Chahal](https://github.com/JevanC)

# Table of Contents
- [Framing the Problem](#framingtheproblem)
- [Baseline Model](#baselinemodel)
- [Final Model](#finalmodel)
- [Fairness Analysis](#fairnessanalysis)

# Framing the Problem <a name="framingtheproblem"></a>

# Baseline Model <a name="baselinemodel"></a>
The baseline model we used is a linear regression model designed to predict the 'sugar' content in food items based on other nutritional information. The model is built using Python's scikit-learn library.

Features in the Model:
Calories: Quantitative
Total Fat: Quantitative
Sodium: Quantitative
Protein: Quantitative
Saturated Fat: Quantitative
Carbohydrates: Quantitative
All features in this model are quantitative, as they represent measurable quantities and are expressed as numerical values. There's no mention of ordinal or nominal features in your description.

## Data Preprocessing:
Handling of String Representations: The 'nutrition' column, initially containing string representations of lists, is converted into actual lists using a custom safe_eval function.
Handling of Missing Values: Missing values in features and target variables are filled with their respective means.
Model Pipeline:
The model pipeline consists of two stages:

## StandardScaler: 
This is used for feature scaling, which standardizes the features by removing the mean and scaling to unit variance. This is important for linear regression as it is sensitive to the scale of input features.

## LinearRegression: 
A simple linear regression model that fits a linear equation to the observed data.
## Model Evaluation:
Mean Squared Error (MSE) and R² Score: These metrics are used to evaluate the model's performance. The MSE is the average squared difference between the actual and predicted values, while the R² score represents the proportion of variance in the dependent variable that is predictable from the independent variables.
Cross-Validation Scores: The model is also evaluated using 5-fold cross-validation to assess its generalization capability. This involves splitting the dataset into 5 parts, training the model on 4 and testing on the 1 remaining part, and repeating this process 5 times.
## Performance and Conclusion:
Root Mean Square Error (RMSE): The RMSE is approximately 103.37. This value indicates the standard deviation of the prediction errors or residuals, which are the differences between observed and predicted values. In general, a lower RMSE is better, but the acceptability of this value greatly depends on the context of the data and the scale of the 'sugar' variable. If the range of sugar content is large, an RMSE of 103 might be acceptable, but if the range is smaller, this could be considered high.
R² Score: The R² score of about 0.774 suggests that approximately 77.4% of the variance in the sugar content is explained by your model. This is generally considered a good score, indicating that your model has a strong fit to the data.
Cross-Validation Scores: The cross-validation scores range from approximately 0.671 to 0.773. This range indicates some variation in the model's performance across different subsets of the data, but the scores are relatively consistent. None of the scores are exceptionally low, which is a good sign that your model is not overfitting to a specific part of the data and should generalize well to new, unseen data.
## Conclusion:
The model appears to be performing well, especially considering the R² score, which is a strong indicator of model fit.
The RMSE value should be interpreted in the context of the data. If the scale of sugar content is large, then the RMSE might be acceptable.
The consistency in cross-validation scores suggests good generalization capability, although some variation is observed.


# Final Model <a name="finalmodel"></a>

# Fairness Analysis <a name="fairnessanalysis"></a>

# Interesting Aggregates <a name="interestingaggregates"></a>
