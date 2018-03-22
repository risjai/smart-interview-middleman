from flask import Flask, request
import pickle
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

pickle_in = open("extra_trees.pickle", "rb")
classifier = pickle.load(pickle_in)

#weights = {'Problem Solving': 80, 'Design': 80, 'CS Skills': 50, 'Test Enumeration': 40, 'Communication': 30}
weights = [80,80,50,40,30]

def calculateScore(data):
    numerator = denominator = 0
    for i in range(0, len(data) - 2):
        if(data[i] != -1):
            numerator += data[i]*weights[i]
            denominator += weights[i]
    return numerator/denominator

@app.route("/", methods = ['GET', 'POST'])
def func():
    return "Hello World!"

@app.route("/api/scores/", methods = ['POST'])
def hello():
    try:
        content = request.json
        print(content)
        data = np.array([content['PROBLEM_SOLVING'], content['DESIGN'], content['CS_SKILLS'],
        content['TEST_ENUMERATION'], content['COMMUNICATION'], 1000, 1000])
        print(data)
        score = calculateScore(data);
        print(score)
        prob = classifier.predict_proba(data.reshape(1, -1))[0][1]
        return json.dumps({"status" : "SUCCESS" , "result": prob, "score": score})
    except Exception as e:
        print("Exception is ",e)
        return json.dumps({"status" : "FAILURE" , "result": str(e)})

@app.route("/api/final_score/", methods = ['POST'])
def final_score():
    try:
        content = request.json
        print(content)
        scoreArray = [];
        decisionArray = [];
    except Exception as e:
        print("Exception is ",e)
        return json.dumps({"status" : "FAILURE" , "result": str(e)})
    

if __name__ == '__main__':
    app.run()