# SGBoost
SGBoost is a simple implement of gradient boosting tree which help to understand the mechanism inside XGBoost. Its supported features include:

- Scikit-learn compatible api
- Built-in loss function. Logistic loss for classification and square error for regression
- Built-in evaluation metric, like accuracy, r2, neg_mean_squared_error
- Cutomize loss function, and evaluation metric just like XGBoost
- Early stopping
- Feature importance
- Handle missing value
- Randomness to avoid overfitting. subsample, colsample_bytree, colsample_bylevel
- Regularization. min_child_weight, reg_lambda, gamma
- Multi-process to find the best tree node split
- Weighted loss function

# Dependencies
- python 2.7
- scikit-learn(>=0.19.0)
- pandas(>=0.19.2)
- numpy(>=1.8.2)

# References

- Tianqi Chen and Carlos Guestrin. [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754). In 22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016
- [Introduction to Boosted Trees](https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf)













