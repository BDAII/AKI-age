# AKI-age
Acute kidney injury (AKI) risk increases with age and the underlying clinical predictors may be heterogeneous across age strata. 
This study aims to uncover the AKI risk factor heterogeneity among general inpatients across age groups using electronic medical records (EMR). 

We designed a novel knowledge mining approach combining artificial intelligence (AI) and expert knowledge to systematically mine AKI risk factor differences among age groups using EMR data. There are two main machine learning methods, namely eXtreme Gradient Boosting (XGBoost) and Tree SHAP method (TreeExplainer).

In this study, we used SHAP value to evaluate the marginal effects (i.e., how the log odds values changed) of the XGBoost model. Moreover, in order to reduce the influence of sample differentiation and enhance the stability and effectiveness of knowledge mining, we used a cross-validation strategy to introduce this data drift, then obtained a weighted average SHAP explanation values (wSHAP), where the weight was the area under the receiver operating characteristic curve (AUC) of the XGBoost model in each fold and the risk score for each variable was derived from the SHAP interpretation using the entire dataset. We verified the effectiveness of the knowledge mining method from the perspectives of accuracy, stability and credibility, and used this approach to clarify the heterogeneity of AKI risk factors between age groups.

