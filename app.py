import pickle

from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

app=Flask(__name__)

#Deserialize /depickle - Loading the model
clf=pickle.load(open('model.pkl','rb'))

#jinja2 template- Template engine- will select the templates from the templates folder
@app.route("/")#decrators->route the url - when user access the route url hello() method get invoked
def hello():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict_class():
    print([x for x in request.form.values()])#POST method is working fine- communication created successfully
    features=[int(x) for x in request.form.values()]#maintain the input same as the data that u trained
    with open('sst.pkl','rb') as file:
        sst=pickle.load(file)
    #label encode,normalize - 2 conflicts

    # sst=StandardScaler()
    # sst=sst.fit(X_train)#similar scale
    output=clf.predict(sst.transform([features]))
    print(output)
    if output[0]==0:
        return render_template("index.html",pred="the person will not purchase the SUV")
    else:
        return render_template("index.html", pred="the person will  purchase the SUV")


if __name__ =="__main__":#checks if current script is the entry point of program
    app.run(debug=True)#will create a flask local server