import nltk
nltk.download(['punkt', 'wordnet', 'omw-1.4'])


# import statements
import sys
import pickle
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier




def load_data(database_filepath):
    engine = create_engine("sqlite:///" + database_filepath)
    df = pd.read_sql("SELECT * FROM messages_with_categories", engine)[1:100]
    X = df.message
    category_names = df.columns[4:].values
    Y = df[category_names]
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False)
        #'clf__n_neighbors': (5, 20, 50),
        #'clf__weights': ('uniform', 'distance')
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    acc_all = []
    f1_all = []
    precision_all = []
    recall_all = []
    for i in range(len(category_names)):
        acc = (Y_pred[:,i] == Y_test.iloc[:,i]).mean()
        f1 = f1_score(Y_test.iloc[:,i], Y_pred[:,i], labels=np.unique(Y_pred[:,i]), zero_division=0)
        precision = precision_score(Y_test.iloc[:,i], Y_pred[:,i], labels=np.unique(Y_pred[:,i]), zero_division=0)
        recall = recall_score(Y_test.iloc[:,i], Y_pred[:,i], labels=np.unique(Y_pred[:,i]), zero_division=0)
        acc_all.append(acc)
        f1_all.append(f1)
        precision_all.append(precision)
        recall_all.append(recall)
        print("Accuracy for", category_names[i], ":", acc)
        print("F1 score for", category_names[i], ":", f1)
        print("Precision for", category_names[i], ":", precision)
        print("Recall for", category_names[i], ":", recall)
    
    acc_all_df = pd.DataFrame({'category_name':category_names, 'accuracy':acc_all, 'f1_score':f1_all, 'precision':precision_all, 'recall':recall_all})
    return acc_all_df


def save_model(model, model_filepath):
    file_to_store = open(model_filepath, "wb")
    pickle.dump(model, file_to_store)
    file_to_store.close()
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
