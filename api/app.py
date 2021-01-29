from flask import Flask
from flask import request
from flask import json
import boto3
import pickle
import pandas as pd

# Nome do bucket e dos arquivos
BUCKET_NAME = 'py-pickles'
CLF_FILE_NAME = 'text_clf.pkl'
VEC_FILE_NAME = 'text_vec.pkl'
LABELS_FILE_NAME = 'ids.pkl'

app = Flask(__name__)

S3 = boto3.client('s3', region_name='us-east-2')


def memoize(f):
    memo = {}

    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]

    return helper

@app.route('/')
@app.route('/help/')
@app.route('/course/')
@app.route('/course/classifier/')
@app.route('/course/classifier/help/')
def home():
    return "Para utilizar a API use:\n http POST IP/course/classifier/'nome do curso'"

@app.route('/course/classifier/<string:name>', methods=['POST'])
def classifier(name):
    prediction = predict(name)
    
    responseObject = {}
    responseObject['statusCode'] = 200
    responseObject['headers'] = {}
    responseObject['headers']['Content-Type'] = 'application/json'
    responseObject['body'] = json.dumps(prediction)

    return responseObject


@memoize
def load_vec(key):
    response_vec = S3.get_object(Bucket=BUCKET_NAME, Key=key)
    vec_str = response_vec['Body'].read()

    vec = pickle.loads(vec_str)

    return vec

@memoize
def load_clf(key):
    response_clf = S3.get_object(Bucket=BUCKET_NAME, Key=key)
    clf_str = response_clf['Body'].read()

    clf = pickle.loads(clf_str)

    return clf


@memoize
def load_labels(key):
    response_labels = S3.get_object(Bucket=BUCKET_NAME, Key=key)
    labels_str = response_labels['Body'].read()

    labels = pickle.loads(labels_str)

    return labels


def predict(data):
    vec = load_vec(VEC_FILE_NAME)
    clf = load_clf(CLF_FILE_NAME)
    labels = load_labels(LABELS_FILE_NAME)

    result_df = pd.DataFrame(clf.predict_proba(vec.transform([data]))*100, columns=labels.values).transpose().reset_index().rename(columns={'index':'name', 0:'prob'}).sort_values('prob', ascending = False).head(5).reset_index()

    if result_df.prob[0] > 70:
      trust = True
    else:
      trust = False

    return {
      'Trusty': trust,
      'First': {
        'Name': result_df.name[0],
        'Probability': result_df.prob[0]
        },
      'Second': {
        'Name': result_df.name[1],
        'Probability': result_df.prob[1]
        },
      'Third': {
        'Name': result_df.name[2],
        'Probability': result_df.prob[2]
        },
      'Fourth': {
        'Name': result_df.name[3],
        'Probability': result_df.prob[3]
        },
      'Fifth': {
        'Name': result_df.name[4],
        'Probability': result_df.prob[4]
        }
      }


if __name__ == '__main__':
    # listen on all IPs
    app.run(host='0.0.0.0')

