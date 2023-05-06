import os
from flask import Flask, render_template, request, make_response
import csv
import pandas as pd
import numpy as  np
import io
from io import StringIO
import pickle
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go

app= Flask(__name__)
APP__ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    return render_template("upload.html")

@app.route("/upload", methods=['POST'])


def upload():

    name = request.form['name']
    age = request.form['age']
    gender = request.form['gender']


    target = APP__ROOT
    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        process_file(destination)


    print(os.getcwd())

    file__filename= "patient_parameters.csv"
    destination_ = os.path.abspath(file__filename)
    graphJSON= final_process(destination_)
    table= tabulation(destination_,name,age,gender)

    return render_template('notdash2.html', name=name, age=age, gender=gender, graphJSON=graphJSON,tables=[table.to_html()],titles=[''])

def time_domain_pm(nn_intervals):
  nn_absdiff=np.diff(nn_intervals)
  nn_sqabs_diff= np.power(nn_absdiff,2)
  HR = 60000/(np.mean(nn_intervals))
  MEAN_RR = np.mean(nn_intervals)
  MEDIAN_RR = np.median(nn_intervals)
  SDRR=np.std(nn_intervals)
  SDSD=np.std(nn_absdiff)
  RMSSD= np.sqrt(np.mean(nn_sqabs_diff))
  NN25=nn_absdiff[np.where(nn_absdiff > 25)]
  NN50= nn_absdiff[np.where(nn_absdiff > 50)]
  try:
    pNN25 = float(len(NN25))/float(len(nn_absdiff))
  except:
    pNN25 = np.nan
  try:
    pNN50 = float(len(NN50))/float(len(nn_absdiff))
  except:
    pNN50 = np.nan

  return [MEAN_RR, MEDIAN_RR, SDRR, RMSSD, SDSD, HR, pNN25, pNN50]


def process_file(file_):
    df=pd.read_csv(file_,parse_dates = ['Time'], index_col=0)
    df.rename(columns={df.columns[0]: "IBI"}, inplace=True)
    df_new = df[['IBI']]
    df = df_new[(df_new["IBI"] >= 400) & (df_new["IBI"] <= 1333)]
    raw_nn = np.array(df).squeeze()
    t = 5
    t = t * 60
    sums = 0
    for idx in range(len(raw_nn)):
        sums += raw_nn[idx] * 0.001
        if (sums >= t):
            break

    nn_intervals = raw_nn[0:idx]
    res = time_domain_pm(nn_intervals)
    testDict = {"MEAN_RR": [res[0]], "MEDIAN_RR": [res[1]], "SDRR": [res[2]], "RMSSD": [res[3]], "SDSD": [res[4]],
                "HR": [res[5]], "pNN25": [res[6]], "pNN50": [res[7]]}
    testFile = pd.DataFrame(testDict)

    for j in range(idx + 1, len(raw_nn)):
        nn_intervals = raw_nn[j - idx:j]
        res = time_domain_pm(nn_intervals)
        testDict = {"MEAN_RR": res[0], "MEDIAN_RR": res[1], "SDRR": res[2], "RMSSD": res[3], "SDSD": res[4],
                    "HR": res[5], "pNN25": res[6], "pNN50": res[7]}
        #testFile = testFile.append(testDict, ignore_index=True)
        testFile=pd.concat([testFile,pd.DataFrame([testDict])],ignore_index=True)

    count = 1
    test_sample = testFile
    #test_sample.to_csv("{}_patient_parameters.csv".format(file_.split('.')[0]))
    test_sample.to_csv("patient_parameters.csv")


def final_process(file):

    #df = pd.read_csv(StringIO(result))
    df= pd.read_csv(file)
    new_df= df.drop(['Unnamed: 0'], axis=1)
    new_df=new_df.reset_index(drop=True)


    # load the model from disk
    import xgboost as xgb
    loaded_model = pickle.load(open("D:/FYP/Saved_Model/xgb_clf.pkl", 'rb'))
    new_df['prediction'] = loaded_model.predict(new_df)
    new_df.rename(columns={'prediction': 'Stress_Levels'}, inplace=True)
    Stress_map = {0: 'no stress', 1: "Medium stress)", 2: "High_stress"}
    new_df['Stress_Levels'] = new_df['Stress_Levels'].map(Stress_map)

    new_df.to_csv("patient_stress_pred.csv")

    #plot chart
    new_df = new_df.groupby(['Stress_Levels'])['Stress_Levels'].count().reset_index(name='count')
    fig = px.pie(new_df, values='count', names='Stress_Levels', title='Stress Level distribution throughout the duration of IBI measurement', color='Stress_Levels',
             color_discrete_map={'no stress':'cyan',
                                 'Medium stress':'yellow',
                                 'High_stress':'red',
                                 })

    fig.update_traces(textposition='inside', textinfo='percent+label',insidetextorientation='radial')
    #fig.update_layout(uniformtext_minsize=12, margin=dict(t=0, b=0, l=0, r=0))
    fig.update_layout(uniformtext_minsize=12)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    #return render_template('notdash2.html', graphJSON=graphJSON, header=header, description=description)
    return (graphJSON)

def tabulation(file,name,age,gender):
    name,gender=name,gender
    age=int(age)
    gender_map={"Female":1,"F":1,"Male":0,"M":1}
    table_hrv = {}
    li = []
    df_new = pd.read_csv(file)
    table_hrv["Name"]=name
    table_hrv["Age"]=age
    table_hrv["Gender"]=gender
    table_hrv["SDNN"] = df_new.loc[:, 'SDRR'].mean()
    table_hrv["RMSSD"] = df_new.loc[:, 'RMSSD'].mean()
    table_hrv["pNN50"] = df_new.loc[:, 'pNN50'].mean()
    table_hrv["HR"] = df_new.loc[:, 'HR'].mean()


    df = pd.DataFrame(table_hrv, index=[0])
    df["Gender"]=df["Gender"].map(gender_map)
    #li.append(df)
    #df = pd.concat(li, axis=0, ignore_index=True)

    df['SDNN_feedback'] = 'N'
    df['pNN50_feedback'] = 'N'
    df['RMSSD_feedback'] = 'N'
    df['HR_feedback'] = 'N'

    # SDNN feedback
    df.loc[(((df['Age'] >= 10) & (df['Age'] < 29)) & (df['SDNN'] < 144) & (df['Gender'] == 0)) | (
                ((df['Age'] >= 30) & (df['Age'] < 49)) & (df['SDNN'] < 116) & (
                    df['Gender'] == 0)), 'SDNN_feedback'] = 'BN'
    df.loc[(((df['Age'] >= 50) & (df['Age'] < 69)) & (df['SDNN'] < 87) & (df['Gender'] == 0)) | (
                ((df['Age'] >= 70) & (df['Age'] < 99)) & (df['SDNN'] < 99) & (
                    df['Gender'] == 0)), 'SDNN_feedback'] = 'BN'

    df.loc[(((df['Age'] >= 10) & (df['Age'] < 29)) & (df['SDNN'] < 104) & (df['Gender'] == 1)) | (
                ((df['Age'] >= 30) & (df['Age'] < 49)) & (df['SDNN'] < 99) & (
                    df['Gender'] == 1)), 'SDNN_feedback'] = 'BN'
    df.loc[(((df['Age'] >= 50) & (df['Age'] < 69)) & (df['SDNN'] < 95) & (df['Gender'] == 1)) | (
                ((df['Age'] >= 70) & (df['Age'] < 99)) & (df['SDNN'] < 91) & (
                    df['Gender'] == 1)), 'SDNN_feedback'] = 'BN'

    df.loc[(((df['Age'] >= 10) & (df['Age'] < 29)) & (df['SDNN'] > 217) & (df['Gender'] == 0)) | (
                ((df['Age'] >= 30) & (df['Age'] < 49)) & (df['SDNN'] > 176) & (
                    df['Gender'] == 0)), 'SDNN_feedback'] = 'AN'
    df.loc[(((df['Age'] >= 50) & (df['Age'] < 69)) & (df['SDNN'] > 147) & (df['Gender'] == 0)) | (
                ((df['Age'] >= 70) & (df['Age'] < 99)) & (df['SDNN'] > 147) & (
                    df['Gender'] == 0)), 'SDNN_feedback'] = 'AN'

    df.loc[(((df['Age'] >= 10) & (df['Age'] < 29)) & (df['SDNN'] > 190) & (df['Gender'] == 1)) | (
                ((df['Age'] >= 30) & (df['Age'] < 49)) & (df['SDNN'] > 159) & (
                    df['Gender'] == 1)), 'SDNN_feedback'] = 'AN'
    df.loc[(((df['Age'] >= 50) & (df['Age'] < 69)) & (df['SDNN'] > 155) & (df['Gender'] == 1)) | (
                ((df['Age'] >= 70) & (df['Age'] < 99)) & (df['SDNN'] > 137) & (
                    df['Gender'] == 1)), 'SDNN_feedback'] = 'AN'

    # RMSSD feedback
    df.loc[(((df['Age'] >= 10) & (df['Age'] < 29)) & (df['RMSSD'] < 35) & (df['Gender'] == 0)) | (
                ((df['Age'] >= 30) & (df['Age'] < 49)) & (df['RMSSD'] < 21) & (
                    df['Gender'] == 0)), 'RMSSD_feedback'] = 'BN'
    df.loc[(((df['Age'] >= 50) & (df['Age'] < 69)) & (df['RMSSD'] < 14) & (df['Gender'] == 0)) | (
                ((df['Age'] >= 70) & (df['Age'] < 99)) & (df['RMSSD'] < 17) & (
                    df['Gender'] == 0)), 'RMSSD_feedback'] = 'BN'

    df.loc[(((df['Age'] >= 10) & (df['Age'] < 29)) & (df['RMSSD'] < 25) & (df['Gender'] == 1)) | (
                ((df['Age'] >= 30) & (df['Age'] < 49)) & (df['SDNN'] < 21) & (
                    df['Gender'] == 1)), 'RMSSD_feedback'] = 'BN'
    df.loc[(((df['Age'] >= 50) & (df['Age'] < 69)) & (df['RMSSD'] < 18) & (df['Gender'] == 1)) | (
                ((df['Age'] >= 70) & (df['Age'] < 99)) & (df['SDNN'] < 14) & (
                    df['Gender'] == 1)), 'RMSSD_feedback'] = 'BN'

    df.loc[(((df['Age'] >= 10) & (df['Age'] < 29)) & (df['RMSSD'] > 71) & (df['Gender'] == 0)) | (
                ((df['Age'] >= 30) & (df['Age'] < 49)) & (df['RMSSD'] > 47) & (
                    df['Gender'] == 0)), 'RMSSD_feedback'] = 'AN'
    df.loc[(((df['Age'] >= 50) & (df['Age'] < 69)) & (df['RMSSD'] > 30) & (df['Gender'] == 0)) | (
                ((df['Age'] >= 70) & (df['Age'] < 99)) & (df['RMSSD'] > 27) & (
                    df['Gender'] == 0)), 'RMSSD_feedback'] = 'AN'

    df.loc[(((df['Age'] >= 10) & (df['Age'] < 29)) & (df['RMSSD'] > 61) & (df['Gender'] == 1)) | (
                ((df['Age'] >= 30) & (df['Age'] < 49)) & (df['RMSSD'] > 41) & (
                    df['Gender'] == 1)), 'RMSSD_feedback'] = 'AN'
    df.loc[(((df['Age'] >= 50) & (df['Age'] < 69)) & (df['RMSSD'] > 32) & (df['Gender'] == 1)) | (
                ((df['Age'] >= 70) & (df['Age'] < 99)) & (df['RMSSD'] > 30) & (
                    df['Gender'] == 1)), 'RMSSD_feedback'] = 'AN'

    # pNN50 feedback

    df.loc[(((df['Age'] >= 10) & (df['Age'] < 29)) & (df['pNN50'] < 0.13) & (df['Gender'] == 0)) | (
                ((df['Age'] >= 30) & (df['Age'] < 49)) & (df['pNN50'] < 0.03) & (
                    df['Gender'] == 0)), 'pNN50_feedback'] = 'BN'
    df.loc[(((df['Age'] >= 50) & (df['Age'] < 69)) & (df['pNN50'] < -0.01) & (df['Gender'] == 0)) | (
                ((df['Age'] >= 70) & (df['Age'] < 99)) & (df['pNN50'] < 0.01) & (
                    df['Gender'] == 0)), 'pNN50_feedback'] = 'BN'

    df.loc[(((df['Age'] >= 10) & (df['Age'] < 29)) & (df['pNN50'] < 0.05) & (df['Gender'] == 1)) | (
                ((df['Age'] >= 30) & (df['Age'] < 49)) & (df['pNN50'] < 0.03) & (
                    df['Gender'] == 1)), 'pNN50_feedback'] = 'BN'
    df.loc[(((df['Age'] >= 50) & (df['Age'] < 69)) & (df['pNN50'] < 0.01) & (df['Gender'] == 1)) | (
                ((df['Age'] >= 70) & (df['Age'] < 99)) & (df['pNN50'] < 0) & (
                    df['Gender'] == 1)), 'pNN50_feedback'] = 'BN'

    df.loc[(((df['Age'] >= 10) & (df['Age'] < 29)) & (df['pNN50'] > 0.39) & (df['Gender'] == 0)) | (
                ((df['Age'] >= 30) & (df['Age'] < 49)) & (df['pNN50'] > 0.23) & (
                    df['Gender'] == 0)), 'pNN50_feedback'] = 'AN'
    df.loc[(((df['Age'] >= 50) & (df['Age'] < 69)) & (df['pNN50'] > 0.09) & (df['Gender'] == 0)) | (
                ((df['Age'] >= 70) & (df['Age'] < 99)) & (df['pNN50'] > 0.05) & (
                    df['Gender'] == 0)), 'pNN50_feedback'] = 'AN'

    df.loc[(((df['Age'] >= 10) & (df['Age'] < 29)) & (df['pNN50'] > 0.29) & (df['Gender'] == 1)) | (
                ((df['Age'] >= 30) & (df['Age'] < 49)) & (df['pNN50'] > 0.17) & (
                    df['Gender'] == 1)), 'pNN50_feedback'] = 'AN'
    df.loc[(((df['Age'] >= 50) & (df['Age'] < 69)) & (df['pNN50'] > 0.09) & (df['Gender'] == 1)) | (
                ((df['Age'] >= 70) & (df['Age'] < 99)) & (df['pNN50'] > 0.08) & (
                    df['Gender'] == 1)), 'pNN50_feedback'] = 'AN'

    # HR feedback

    df.loc[(((df['Age'] >= 10) & (df['Age'] < 29)) & (df['HR'] < 66) & (df['Gender'] == 0)) | (
                ((df['Age'] >= 30) & (df['Age'] < 49)) & (df['HR'] < 69) & (df['Gender'] == 0)), 'HR_feedback'] = 'BN'
    df.loc[(((df['Age'] >= 50) & (df['Age'] < 69)) & (df['HR'] < 67) & (df['Gender'] == 0)) | (
                ((df['Age'] >= 70) & (df['Age'] < 99)) & (df['HR'] < 61) & (df['Gender'] == 0)), 'HR_feedback'] = 'BN'

    df.loc[(((df['Age'] >= 10) & (df['Age'] < 29)) & (df['HR'] < 75) & (df['Gender'] == 1)) | (
                ((df['Age'] >= 30) & (df['Age'] < 49)) & (df['HR'] < 72) & (df['Gender'] == 1)), 'HR_feedback'] = 'BN'
    df.loc[(((df['Age'] >= 50) & (df['Age'] < 69)) & (df['HR'] < 64) & (df['Gender'] == 1)) | (
                ((df['Age'] >= 70) & (df['Age'] < 99)) & (df['HR'] < 65) & (df['Gender'] == 1)), 'HR_feedback'] = 'BN'

    df.loc[(((df['Age'] >= 10) & (df['Age'] < 29)) & (df['HR'] > 86) & (df['Gender'] == 0)) | (
                ((df['Age'] >= 30) & (df['Age'] < 49)) & (df['HR'] > 83) & (df['Gender'] == 0)), 'HR_feedback'] = 'AN'
    df.loc[(((df['Age'] >= 50) & (df['Age'] < 69)) & (df['HR'] > 89) & (df['Gender'] == 0)) | (
                ((df['Age'] >= 70) & (df['Age'] < 99)) & (df['HR'] > 83) & (df['Gender'] == 0)), 'HR_feedback'] = 'AN'

    df.loc[(((df['Age'] >= 10) & (df['Age'] < 29)) & (df['HR'] > 91) & (df['Gender'] == 1)) | (
                ((df['Age'] >= 30) & (df['Age'] < 49)) & (df['HR'] > 86) & (df['Gender'] == 1)), 'HR_feedback'] = 'AN'
    df.loc[(((df['Age'] >= 50) & (df['Age'] < 69)) & (df['HR'] > 84) & (df['Gender'] == 1)) | (
                ((df['Age'] >= 70) & (df['Age'] < 99)) & (df['HR'] > 81) & (df['Gender'] == 1)), 'HR_feedback'] = 'AN'
    newgen_map={0:"Male",1:"Female"}
    df["Gender"] = df["Gender"].map(newgen_map)

    #df.iloc[:, [0,1,2,3,7,4,9,5,8,6,10]]
    df2=df[["Name","Age","Gender","SDNN","SDNN_feedback","RMSSD","RMSSD_feedback","pNN50","pNN50_feedback","HR","HR_feedback"]]
    df2.to_csv("hrvParams_tabulation.csv")
    return df2






if __name__ == "__main__":
    app.run(port=4555, debug=True)