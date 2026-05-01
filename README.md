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
 
Raw text data cannot be directly used in machine learning models so sentiment analysis was applied to transform qualitative text into quantitative numerical values, which allow the model to learn from customer reviews. For this project, we used **VADER (Valence Aware Dictionary and Sentiment Reasoner)**, a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. However, there are limitations that apply. Since VADER is a rule-based approach, it may not fully capture complex language patterns such as sarcasm or context-dependent meaning. Despite the limitations, it still provides a fast and effective method for sentiment extraction. 

#### 3. Aggregation
Since each restaurant has multiple reviews, the dataset was transformed from a review-level structure to restaurant-level structure. For each restauarnt:
- average sentiment score across all reviews was computed
- total number of reviews was calculated
- average review rating was derived

Machine learning models require consistent input granularity so if each review were treated as a separate data point, then restaurants with more reviews would be overrepresented. Aggregation ensures that each restaurant contributes equally to the model, which prevents bias toward businesses with high number of reviews.

#### 4. Feature Engineering
Feature engineering was performed to improve model performance and address data distribution issues. The review count for the dataset is highly skewed with most restaurants with few reviews while small number of businesses having large review counts. To reduce this imbalance, a logarithmic transformation was applied
```bash 
np.log1p(agg_df["review_count"])
```
This help prevent large restaurants from dominating the model and reduce skewness in the distribution.

**Feature Set**

The final features used for modeling include:
- sentiment score
- log review count (popularity in normalized form)

These features were chosen because they capture two main factors in restaurant quality:
- What people think (sentiment)
- How many people think it (popularity)
#### 5. Final Dataset
After preprocessing and feature engineering, the dataset was transformed into a structured format which is important for supervised learning. Each row represents a restaurant with both input features and target output. It is also essential for integration with web application for real-time filtering and ranking.

### Model Development
The two main input for the model are sentiment score and review count. The output for the model is the predicted business rating which would represent the Yelp star system.
The algorithm we used was a random forest regressor which is an ensemble method. It uses decision trees to make predictions. Our model used 100 trees. Then the average is taken from all the trees to create the final prediction. We used this algorithm because it reduces overfitting when compared to a single decision tree and it can get non-linear relationships between features and target.
The performance of the model was mainly measured by the RMSE and MAE. The model performed well given the methodolgy. It was able to successfully show the relationship between the sentiment and ratings. The feature importance analysis shows that both sentiment and popularity contribute to predictions. The model does have  limitations. Among them is the fact that only two features were used to predict restaurant rating so it isn't necessarily taking a lot of complexity into account. 

**Random Forest**
For this project, a Random Forest regressor was used due to its ability to handle non-linear relationships and provide stable predictions with minimal tuning required. Restaurant quality is influenced by many factors like sentiment and popularity, which might not have a simple linear relationship with ratings. Random Forest models are suited for this because they combine multiple trees to capture the patterns in the data. 

Reasoning:
- Handles Non-Linearity
- Reduce Overfitting
- Works well with small feature set
- Feature Importance
- robust to noise

### Web Application
For this project, the pretrained model was saved using pickle and deployed into an interactive web application using Flask. This allow users to interact with the model in real time and receive ranked restaurant recommendations. To make the web application accessible outside the local environment, Ngrok was used. Ngrok creates a secure public URL that tunnels to the local Flask server running in Google Colab. This allows the application to be accessed through a browser without needing a full cloud deployment.

How it works:
- The Flask app runs locally on a port (e.g., 5000)
- Ngrok creates a public URL
- Requests from the URL are forwarded to the local server

Key Features:
- City filtering
  - Users can select a specific city in Pennsylvania (PA) or view all restaurants
- Custom ranking size
  - Users can choose how many restaurant to display (5, 10, 20, 50)
- Ranking
  - Restaurants are ranked based on predicted scores from the model
- Data Visualization
  - Multiple chares are included for better interpretability
    - Bar chart of top restaurants
    - Feature importance visualization
    - Score distribution chars
    - Sentiment vs Predicted score plot

### Demo Video

[Video Link](https://github.com/mrjpmclaughlin/CMPSC445_BestRestaurants/tree/main/demo)

### Discussion 
The project shows that there is a relationship between review sentiment, review volume, and restaurant rating. Using sentiment scores from VADER and review counts the model was able to accurately make predictions. The random forest regressor worked well because it captured the non-linear patterns and combines multiple decision trees to help with stability. There were limitations though. the model was more on the simple side so it didn't account for other real-world factors like price or location. VADER also does not always catch linguistic nuance.
### Conclusion
The project demonstrates ho wmachine learning can be applied to real-world text data to generate useful predictions. We were able to use concepts learned throughout the semester like preprocessing, feature engineering, and model evaluation in a practical way. The model performs within reason, but to improve accuracy more complexity would have to be introduced like more advanced text representations. 
### AI Usage
AI was used to help with the website design to give it a clean look and to help with the process of running it in Colab. It was also used to help research for having the URL and using Ngrok.
