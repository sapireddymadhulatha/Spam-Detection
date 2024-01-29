from flask import *
import numpy as np 
import pandas as pd
import nltk
import matplotlib.pyplot as plt
#import seaborn as sns
import pandas
import string
import pprint
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)


@app.route("/", methods=["POST","GET"])
def index():
    return render_template("home.html")

@app.route("/home", methods=["POST","GET"])
def home():
    msg = []
    if request.method=="POST":
        msg1 = request.form['message']

        df_sms = pd.read_csv('static/spam.csv',encoding='latin-1')
        df_sms.head()
        
        df_sms = df_sms.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
        df_sms = df_sms.rename(columns={"v1":"label", "v2":"sms"})
        df_sms.head()
             
##       print(len(df_sms))
        df_sms.label.value_counts()
        df_sms.describe()
        
        df_sms['length'] = df_sms['sms'].apply(len)
        df_sms.head()

        df_sms['length'].plot(bins=50, kind='hist')

        df_sms.loc[:,'label'] = df_sms.label.map({'ham':0, 'spam':1})
##        print(df_sms.shape)
        df_sms.head()

        documents = ['Hello, how are you!','Win money, win from home.','Call me now.','Hello, Call hello you tomorrow?']

        lower_case_documents = []
        lower_case_documents = [d.lower() for d in documents]
        #print(lower_case_documents)

        sans_punctuation_documents = []

        for i in lower_case_documents:
            sans_punctuation_documents.append(i.translate(str.maketrans("","", string.punctuation)))
            
##        print(sans_punctuation_documents)

        preprocessed_documents = [[w for w in d.split()] for d in sans_punctuation_documents]
##        print(preprocessed_documents)

        frequency_list = []

        frequency_list = [Counter(d) for d in preprocessed_documents]
        pprint.pprint(frequency_list)

        count_vector = CountVectorizer()

        count_vector.fit(documents)
        count_vector.get_feature_names()


        doc_array = count_vector.transform(documents).toarray()
        


        frequency_matrix = pd.DataFrame(doc_array, columns = count_vector.get_feature_names())
        


        X_train, X_test, y_train, y_test = train_test_split(df_sms['sms'], 
                                                            df_sms['label'],test_size=0.20, 
                                                            random_state=1)     


        count_vector = CountVectorizer()

        training_data = count_vector.fit_transform(X_train)

        testing_data = count_vector.transform(X_test)

        naive_bayes = MultinomialNB()
        naive_bayes.fit(training_data,y_train)
        MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
        predictions = naive_bayes.predict(testing_data)

        a=msg1
        training = count_vector.transform([a])
        result=naive_bayes.predict(training)
        acc=accuracy_score(y_test, predictions)
        pre=precision_score(y_test, predictions)
        rc=recall_score(y_test, predictions)
        fs=f1_score(y_test, predictions)
        result=np.array(result)
        result=np.append(result,acc)
        result=np.append(result,pre)
        result=np.append(result,rc)
        result=np.append(result,fs)
        
    
##        msg.append()
##        msg.append()
##        msg.append()
##        msg.append()
##        print('gdfhgkfhfh' +msg)
        
        return render_template("home.html", data=result)



@app.route("/form", methods=["POST","GET"])
def page():
    return render_template("form.html")
if __name__ == "__main__":
    app.run(debug=True)
