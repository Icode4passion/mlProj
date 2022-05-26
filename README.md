As a python and machine learning enthusiast , i love reading blogs and code.
In this article we will use Python and Flask in Deploying Machine Learning algorithms. 

The main focus of the article will be how Flask/Python is used in deploying ML algorithms , 
we will not deep dive in to the Algorithms and their performance . 

The take away from the article will be how dataset is fed to flask app and 
how are we sending the prediction back via jinja templete and what modules are encountered during the process. 



-----------------------------------------

there are many options in for Machine learning algorithms to be programmed , but Sciket



-------------------------------------------------------

## Creating Flask App

Lets get started with the code . I will not be concentrating on the Flask App Structure in this article please be noted . 
In the process of building the app we will be installing the libraries as needed.



```sh

mkdir mlproj
cd mlproj
python -m venv env
cd env/Scripts 
activate

pip install -r requirements.txt

```

the above commands are used to create a folder , virtual environment and active the environmet with the libraries to be installed .

requirements.txt has the following packages to be installed 

```
Flask
joblib
numpy
pandas
requests
scikit-learn
scipy
six
sklearn
```

### Let's code 

In order for us to create a Machine Learning flask app we need a program or python file to train the data and results of the algorithm that ran with appropriate data has to dumped into a pkl file and other program reads the data and provides routes via flaks app to give us the predections of the data.

#### Machine learning Modeling code 

Consider the first program to be a pure Machine Learning Algorithm python code . Will try to explain the process what exactly happens in the below code . 
In the below code for traing the model we will be using DecissionTreeClassifier from Sklearn library.

Create a model.py in the folder and add below code in it

```python 

import joblib
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

```

We are importing the pandas , numpy and matplotlib modules along with DecisionTreeClassifier.


```python
df = pd.read_csv('CO2 Emissions_Data.csv')

X = df.iloc[:,:4]
y = df['CO2emissions']
X_train , X_test, y_train , y_test = train_test_split(X, y , test_size=0.3,random_state=1)
dtc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train, y_train)
joblib.dump(dtc,"dtc.pkl")
y_predict = dtc.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_predict))
print("Accuracy:",metrics.accuracy_score(y_test, y_predict))

```
Read the data from the local directory which was already dowloaded from [Kaggle](https://www.kaggle.com/) we are using pandas module to read it.
Now once the data is read and is stored in the form of Dataframes .

Since you have come along till here i would like to focus on the real problem that we are trying to solve .

The Article here has data related to Co2 Emission , how the CO2 emission is creating problem to environment and this data set actually deals with Vehicles emitting CO2 and what factors contribute for this emission . Our main goal is to predict the Co2 Emission of the vehicle when provided with the similar kind of Data and to see how accurately the system or model has predicted it.

The above code trains the data and stores the resut in the pickle formate . IN terms of Machine Learning the model has been trained we will be deducting or predecting the result based on the data provided in the pickle.

```
- After training a scikit-learn model, it is desirable to have a way to persist the model for future use without having to retrain. The following sections give you some hints on how to persist a scikit-learn model. 

What is joblib Python?
Image result for joblib python wiki
Joblib is a set of tools to provide lightweight pipelining in Python
```



#### Flask Specific code

Once the ML model is trained and ready we now from flask end , please try to import the specific modules 

```python
from flask import Flask , request , render_template
import joblib
import pandas as pd

```

Add the routes in the app.py. Since the model has already been trained and our goal is to predict the CO2 emission , we need to input data from flask so that the trained model can get the prediction provided by DecissionTreeModel from model.py

```python 


@app.route('/treepredict',methods=['GET','POST'])
def treepredict():
    if request.method=="POST":
        dtc = joblib.load("dtc.pkl")
        
        engineSize = request.form.get('engineSize')
        cylinders = request.form.get('cylinders')
        fuelConsumptionCity = request.form.get('fuelConsumptionCity')
        fuelConsumptionComb = request.form.get('fuelConsumptionComb')
        
        X = pd.DataFrame([[engineSize,cylinders,fuelConsumptionCity,fuelConsumptionComb]],columns=["engineSize","cylinders","fuelConsumptionCity","fuelConsumptionComb"])
        
        prediction = dtc.predict(X)[0]
        
    else:
        prediction = "No sufficient data"
    
    return render_template("out.html", output=prediction)

```

The data from the forms is stored in respective variables and store in dataframe and is used as input to the model,
since we already trained the model and the result is stored in the pkl form the         
dtc = joblib.load("dtc.pkl") , dtc has predict method which takes in the form values and sends the output to jinja.

in the same directory create a html page and paste the below code .
code for treepredict.html

```html
<!DOCTYPE html>
<html>
    <head>
        <title>Machine Learning App</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    </head>
    
    <body>
        <div id="contact_form" style="text-align: center;">
            <h2>
                CO2 Emission Prediction Data using KNC Algorithm 
            </h2>
        </div>
    <div id="contact_form" style="text-align: center;">       

        <form name="form", method="POST", style="text-align: center;">
            <fieldset>

                <div class="input-box">
                    <label for="name" id="engineSize">engineSize</label>
             <input type="number" step="any" name="engineSize", placeholder="Enter Engine Size" required/>
            </div>
            <div class="input-box">
                <label for="name" id="cylinders">cylinders</label>
            <input type="number" name="cylinders", placeholder="Enter weight in kg" required/>
            </div>
            <div class="input-box">
                <label for="name" id="fuelConsumptionCity">fuelConsumptionCity</label>
             <input type="number" step="any" name="fuelConsumptionCity", placeholder="Enter Fuel Consumption City" required/>
                </div>
                <div class="input-box">
                    <label for="name" id="fuelConsumptionComb">fuelConsumptionComb</label>   
             <input type="number" name="fuelConsumptionComb", placeholder="EnterFuel Consumption Comb" required/>
             </div>
             <div class="input-box">
                <input type="submit" name="submit" class="button" id="submit_btn" value="Run" />

             </div>

            </fieldset>
            
        </form>
        <div >

            <table>
                <tr>
                    
                </tr>
                <tr>
                    <td>
                        <img src="https://c8.alamy.com/comp/P3J2YW/vector-artistic-drawing-illustration-of-car-air-co-pollution-P3J2YW.jpg"
                        class="img-fluid" alt="Sample image" width="400" 
                        height="300">
                    </td>
                    
                </tr>                
                <tr>
                    <td>
                        <h3>CO2 Combission</h3> 
                    </td>
                    <td>
                        <h4 style="text-align: center;">{{ output }}</h4>
                    </td>
                </tr>
              </table>
                     
                
         </div>
        
    </div>
    </body>
</html>



```


Once the html file is created now lets roll the sleaves and run the application by adding the below code in app.py

```python 

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

```

We are specifing flask to run on local host and by default the port is 5000 , debug=True menas , app needs to be reloaded when eve any change is made in app.py

App ran successfully with out any error 

![[https://postimg.cc/XXj8k0NM]]



 


Run the docker flask app

```sh
docker-compose build
docker-compose up

```

# mlProj

These days people are talking about Machine learning , it is being used by many purposely or inadvertently . Before diving deep in to it , we will try to understand the basic definition , how it us it and where can we apply it. Learning this can be fun .

> Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. -- Wikipedia 

Machine learning is a fascinating and powerful field of study filled with algorithms and data. - [Jason Brownlee](https://machinelearningmastery.com/machine-learning-tribe/)


In lay man terms Machine learning is a black box where in we feed the data to it , black box has some magical algorithms which does the magic and provide you the desired output .  

Step 1: What is the problem?
        -- The problem that we have taken here is 

Step 2: Why does the problem need to be solved?
Step 3: How would I solve the problem?






