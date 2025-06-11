# British Airways Sentiment Analysis and Feature Ranking

## Project Overview

This project implements sentiment analysis and feature ranking for British Airways customer reviews, as outlined in the provided proposal document. The implementation uses advanced machine learning techniques to identify factors that most influence customer satisfaction.

## Project Structure

1. **Data Collection**: Web scraping of British Airways reviews from airlinequality.com
2. **Data Preprocessing**: Cleaning text data and handling missing values
3. **Tokenization and Stopword Removal**: Processing text for NLP tasks
4. **Sentiment Analysis**: Using Flair to assign sentiment scores
5. **Feature Engineering**: Transforming categorical features and creating TF-IDF vectors
6. **Dimensionality Reduction**: Applying PCA to reduce feature dimensionality
7. **Model Implementation and Comparison**: Training and comparing multiple ML models
8. **Model Evaluation**: Assessing model performance with various metrics
9. **Feature Importance Analysis**: Identifying features that most influence sentiment
10. **Learning to Rank Features**: Prioritizing features that impact customer satisfaction
11. **Advanced Analysis**: Examining sentiment patterns by seat type, traveler type, etc.

## Implementation Details

### Models Implemented
- Support Vector Machines (SVM)
- Decision Trees
- Neural Networks (MLP)
- Random Forest

### Features Used
- TF-IDF features from review text (reduced with PCA)
- Categorical features (Seat Type, Traveler Type, Aircraft, etc.)
- Rating features (Seat Comfort, Cabin Staff, Food, etc.)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score
- ROC AUC
- Confusion Matrices
- Feature Importance Analysis
- SHAP Values for Model Interpretability

## Requirements

The implementation requires the following Python packages:
```
pandas
numpy
matplotlib
seaborn
beautifulsoup4
requests
nltk
scikit-learn
flair
imblearn
xgboost
shap
plotly
```

## How to Run

1. **Data Collection (Optional)**: 
   The code includes a web scraping function that can be enabled by uncommenting the relevant section. Since web scraping is time-consuming, the default behavior assumes that the data has already been scraped.

2. **Run the Complete Pipeline**:
   ```
   python ba_sentiment_analysis.py
   ```

3. **Output**:
   - The script will generate several visualizations showing the analysis results
   - The best model will be saved as 'best_model_[MODEL_NAME].pkl'
   - Performance metrics will be saved to 'model_comparison_results.csv'
   - Feature importance data will be saved to 'feature_importance.csv'

## Extending the Project

The implementation follows the structure outlined in the proposal but can be extended in several ways:

1. **More Advanced NLP Techniques**:
   - Word embeddings (Word2Vec, GloVe)
   - BERT or RoBERTa for contextual embeddings
   - Topic modeling with LDA

2. **Additional ML Models**:
   - Gradient Boosting (XGBoost, LightGBM)
   - Ensemble methods
   - Deep learning approaches (LSTM, CNN)

3. **Interactive Visualizations**:
   - Create an interactive dashboard using Plotly Dash or Streamlit
   - Build a web application for real-time sentiment analysis

## Project Notes

This implementation addresses all the requirements in the project proposal:
- Sentiment classification using multiple ML models
- Dimensionality reduction with PCA
- Learning-to-rank methods for feature prioritization
- Comprehensive evaluation with robust metrics

The code is designed to handle class imbalance using SMOTE and includes thorough data preprocessing steps to ensure high-quality input for the ML models.

## Results Summary

The implementation provides:
1. Comparative performance of different ML models
2. Visualization of feature importance and PCA components
3. Analysis of sentiment by seat type and traveler type
4. Correlation between rating features and overall sentiment
5. Common words in positive and negative reviews
6. Actionable insights for service improvement

This comprehensive analysis helps identify which features of the British Airways service most influence customer satisfaction, enabling targeted improvements to enhance the overall customer experience.
