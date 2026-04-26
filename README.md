# CMPSC445_BestRestaurants

Google Colab Link: [Notebook](https://colab.research.google.com/drive/18_MEik5CJeBPTcUSqhI1DiTd0E5WBWdg?usp=sharing)

### Project Overview
This project builds a machine learning based restaurant recommendation system using the Yelp Open Dataset. Its purpose is to estimate and rank the quality of restaurants by analyzing user reviews and business metadata. The system combines natural language processing (NLP) and supervised machine learning techniques to estimate restaurant ratings. A Random Forest Regression model was trained on engineered features such as sentiment scores obtained from revews and review count statistics. The final model is deployed in a web application that allows users to filter restaurants and receive ranked recommendations based on the predicted ratings. Instead of just predicting ratings, the system operates as a ranking engine that outputs the best restaurants based on predicted scores. 

### Significance of Project
Online restuarant reviews is a major factor in decision making, but large review platforms like Yelp contain noisy, inconsistent, and subjective data. This project addresses the challenge of transforming unstructured review data into meaningful predictions that can provide better decision making. 

The significance of the project include:
- Automating restaurant evaluation
  - using machine learning instead of raw ratings
- extracting insights from text reviews
  - using sentiment analysis
- improving user decision making
  - using ranked recommendations
- demonstrating end to end machine learning deployment
  - data processing -> web application
    
### Data Collection
[Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)

The dataset used in this project is the Yelp Open Dataset, which provides large scale information on businesses and user reviews. 

#### Data Sources
- [yelp_academic_dataset_business.json](https://drive.google.com/file/d/1_oCTyYl1NnNf4k2yJFJJctxTtnaoVG5s/view?usp=sharing):
  - contains metadata about businesses such as name, location, categories, ratings, etc.
- [yelp_academic_dataset_review.json](https://drive.google.com/file/d/1HMtyEy2YGoVDrkWp5HHcb2LlKcE1yfe3/view?usp=sharing):
  - contains user reveiws, including review text and ratings.
- [Cleaned dataset](https://github.com/mrjpmclaughlin/CMPSC445_BestRestaurants/blob/main/datasets/restaurants_clean.csv):
  - After preprocessing and feature engineering, the dataset was transformed into a structured format with one row per restaurant. Each row contains aggregated features derived from multiple reviews, along with metadata used for filtering and ranking.

#### Data Selection
To focus the analysis to find the "Best in PA":
- Only businesses located in **Pennsylvania (PA)** were selected
- Only businesses that had **Restaurant** within the *categories* column were included
- Up to 200,000 reviews were processed 
- Approximately 2,000+ restaurants remained after filtering and aggregation
  
### Data Processing and Feature Engineering
#### 1. Filtering and Merging
The raw Yelp Open Dataset comtains businesses from multiple categories and geographic regions, making it too large and noisy for targeted analysis. To focus on the problem scope, filtering was applied.
- Only businesses located in **Pennsylvania (PA)** were selected
- Only businesses that had **Restaurant** within the *categories* column were included
- **yelp_academic_dataset_business.json** and **yelp_academic_dataset_review.json** files were merged using the common key **business_id**. This allowed each review to be linked with the corresponding restaurant.

These steps were necessary because machine learning model require structured dta, and merging provides a combination of text review information with business attributes to create an unified dataset for analysis.

#### 2. Sentiment Analysis
To extract meaningful information from unstructured review text, sentiment analysis was performed using [VADER (Valence Aware Dictionary and Sentiment Reasoner)](https://github.com/cjhutto/vaderSentiment).
- Each review text was passed through VADER                              
  - Values close to +1 indicate positive sentiment
  - Values close to -1 indicate negative sentiment
  - Values near 0 indicate neutral sentiment
 
Raw text data cannot be directly used in machine learning models so sentiment analysis was applied to transform qualitative text into quantitative numerical values, which allow the model to learn from customer reviews. For this project, we used **VADER (Valence Aware Dictionary and Sentiment Reasoner)**, a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. However, there are limitations that apply. Since VADER is a rule-based approach, it may not fully capture complex language patterns such as sarcasm or context-dependent meaning. Despite the limitations, it still provides a fsat and effective baseline for sentiment extraction. 

#### 3. Aggregation
#### 4. Feature Engineering
#### 5. Final Dataset

### Model Development

### Demo

### Discussion 

### Conclusion

### AI Usage
