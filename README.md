# Resampling influence on the SVM model performance

In this project was explored the performance of Support Vector Machine (SVM) on the Breast Cancer Wisconsin (Diagnostic) data set, available on Kaggle (https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) with the goal of identifying when the model retains the best average sensitivity before or after downsampling the majority class.
Before implementing the model the Data Preprocesion step was needed.

### Data pre-processing

- Data inspection

The dataset contains no missing values for neither the target variable nor any features. The two classes denote malignant and benign cases of tumor, with features numerically representing 30 different aspects of the breast mass. Considering the medical origin of the dataset, its size is conventionally small, reaching a total of 569 records. The records are distributed unequally, with mild but notable class imbalance: 357 cases of benign tumor against 212 cases of malignant.

![image](https://user-images.githubusercontent.com/132377563/236198504-756bb223-df5b-4828-b7ff-2eef1242fba0.png)

- Data preparation

In order to examine model behavior in the settings with and without class imbalance, a copy of the original dataset with a downsampled majority class was created. In both versions of the dataset, the features were normalized using min-max scaling and the target values converted from categorical to numerical to be processable by the models. For each version of the dataset, the test size was set to 20% of the original dataset. To ensure reproducibility, a random seed of 0 was employed throughout all randomized sections of code.

- Feature selection

To improve the performance of the models and reduce the dimensionality of the dataset, feature selection was performed using sequential forward selection (SFS). SFS is a greedy search algorithm appropriate for high-dimensional feature spaces that starts with an empty set of features and iteratively adds one feature at a time based on its correlation with the target variable. To avoid multicollinearity and strike a balance between computational efficiency and performance, the feature set was reduced to 3 elements: “texture_worst”, “concave_points_worst” and “radius_worst”. In determining the feature space, the performance of the feature set was compared between both the downsampled and original training sets, yielding identical accuracy of 71% on the generalized linear regression model.


Along with the SVM model, the kernel function was used. The kernel trick makes the model more powerful, expanding the features into a high dimensional space. For this mode, the linear kernel was chosen due to better performance scores, as showcased in Figure 4. The RBF kernel was overfitting the model (Figure 3) and showed lower performance scores, despite changing hyperparameter γ to different values.

![image](https://user-images.githubusercontent.com/132377563/236200401-4c01d5a3-1b5b-4473-9ec9-29e4110e5f8a.png)
