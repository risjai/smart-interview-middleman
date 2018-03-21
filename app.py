from flask import Flask, request
import pickle
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

pickle_in = open("extra_trees.pickle", "rb")
classifier = pickle.load(pickle_in)

@app.route("/", methods = ['GET', 'POST'])
def func():
    return "Hello World!"

@app.route("/api/scores/", methods = ['POST'])
def hello():
    try:
        content = request.json
        print(content)
        data = np.array([content['PROBLEM_SOLVING'], content['DESIGN'], content['CS_SKILLS'],
        content['TEST_ENUMERATION'], content['COMMUNICATION'], np.random.randint(1, 250), np.random.randint(1, 101)])
        print(data)
        print(classifier.predict_proba(data.reshape(1, -1))[0][1])
        return json.dumps({"status" : "SUCCESS" , "result": classifier.predict_proba(data.reshape(1, -1))[0][1]*100})
    except Exception as e:
        print("Exception is ",e)
        return json.dumps({"status" : "SUCCESS" , "result": str(e)})
    

if __name__ == '__main__':
    app.run(debug=False, port =5555)