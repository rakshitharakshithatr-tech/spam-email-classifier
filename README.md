# Spam Email Classifier Using Machine Learning

## Project Description

This project classifies SMS messages as SPAM or NOT SPAM using Machine Learning.

The project uses TF-IDF for text feature extraction and the Naive Bayes
algorithm for classification.

## Technologies Used

- Python
- Google Colab
- Pandas
- Scikit-learn
- TF-IDF
- Naive Bayes

## Dataset

The project uses a dataset containing 5572 SMS messages.

The messages are classified into:
- Ham – Normal message
- Spam – Unwanted message

## Features

- Loads the SMS dataset
- Prepares the data
- Converts text into numerical features using TF-IDF
- Trains a Naive Bayes classifier
- Calculates model accuracy
- Classifies new messages

## Model Performance

The model achieved an accuracy of **96.68%** on the test dataset.

## Testing

The classifier was tested with different messages.

Example:

SPAM:
"Congratulations! You have won a free lottery prize. Click now!"

NOT SPAM:
"Hi, please send me the notes for tomorrow's class."

## Conclusion

The Spam Email Classifier successfully identifies unwanted messages
using Machine Learning. The project demonstrates the use of Natural
Language Processing and classification techniques for spam detection.

## Future Enhancements

- Use a larger dataset
- Try different Machine Learning algorithms
- Add a web interface
- Add real-time email classification
- Improve detection of new spam patterns
