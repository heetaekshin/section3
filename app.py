from flask import Flask
import os
from flask import render_template
import csv
import json


app = Flask(__name__)

app.config['USERS_CSV_FILE'] = os.path.join(os.path.dirname(__file__), 'users.csv')

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/users', methods=['GET'])
def users():
    path = os.path.dirname(__file__)
    with open(f'{path}/users.csv',newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        newname=[]
        for idx, row, in enumerate(spamreader):
            if idx==0:
                continue
            newname.append(row[0])
        newname = {'users': newname}
    return newname

@app.route('/users/', defaults={'user_order':1})
@app.route('/users/<user_order>',methods=['GET'])
def display_user(user_order):
    path = os.path.dirname(__file__)
    with open(f'{path}/users.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        newname = []
        for idx, row, in enumerate(spamreader):
            if idx == 0:
                continue
            newname.append(row[0])
    return newname[int(user_order)-1]

if __name__ == '__main__':
    app.run(debug=True)