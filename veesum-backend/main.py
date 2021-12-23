from flask import Flask, jsonify, redirect, url_for, request
from werkzeug.wrappers import response
from model import final

app = Flask(__name__)

def getSummary(speech):
  # final(speech)
  return final(speech)

def generateSummary(speechArray):
  summaryArray = []
  for i in range(len(speechArray)):
    val = (speechArray[i].values())
    summaryArray.insert(i, getSummary(*val))
  return summaryArray

@app.route('/get-prediction',methods = ['POST'])
def login():
   if request.method == 'POST':
      res = []
      for i in range(len(request.json)):
        participant = (request.json[i]["participant_name"])
        res.insert(i, {
          "participant_name": participant,
          "summary": generateSummary(request.json[i]["speech"])
        })

      print(res)
      return jsonify(res)

if __name__ == "__main__":
  app.run(debug=True)
