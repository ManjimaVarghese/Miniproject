from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def la():
    return render_template("admin/home.html")

@app.route('/loaddataset')
def loaddataset():
    import pandas as pd
    df = pd.read_csv("C:\\project\\manjima\\manjima\\static\\winequality-red.csv")
    x=df.values[:]
    return render_template("admin/loaddataset.html",x=x)

@app.route('/machinelearningalgoaccuracy')
def machinelearningalgoaccuracy():
    import pandas as pd
    df = pd.read_csv("C:\\project\\manjima\\manjima\\static\\winequality-red.csv")
    y = df.quality
    x = df.drop('quality', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    from pandas.core.common import random_state
    from sklearn.svm import SVC, LinearSVC
    svc_lin = SVC(kernel='linear', random_state=0)
    svc_lin.fit(x_train, y_train)

    pred1= svc_lin.predict(x_test)
    acc1= accuracy_score(pred1,y_test)

    c1= confusion_matrix(pred1,y_test)





    # support vector machiene algorithm(RBF clasiifier)
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel='rbf', random_state=0)
    svc_rbf.fit(x_train, y_train)
    pred2 = svc_rbf.predict(x_test)
    acc2 = accuracy_score(pred2, y_test)
    c2 = confusion_matrix(pred2, y_test)

    # naive bayes algorithm
    from sklearn.naive_bayes import GaussianNB
    guass = GaussianNB()
    guass.fit(x_train, y_train)
    pred3 = guass.predict(x_test)
    acc3 = accuracy_score(pred3, y_test)
    c3 = confusion_matrix(pred3, y_test)

    # Random forest algorithm
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(x_train, y_train)
    pred4 = forest.predict(x_test)
    acc4 = accuracy_score(pred4, y_test)
    c4 = confusion_matrix(pred4, y_test)
    return render_template("admin/algorithmaccuracies.html",svcscore=acc1,svcrpf=acc2,gauss=acc3,rf=acc4,c1=c1,c2=c2,c3=c3,c4=c4)


@app.route('/user_datamining')
def user_datamining():
    import pandas
    spath="C:\\project\\manjima\\manjima\\static\\"
    str = "C:\\project\\manjima\\manjima\\static\\winequality-red.csv"
    pd = pandas.read_csv(str)
    import numpy
    import numpy
    import seaborn
    col = ["volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol",	"quality"]
    correlation = numpy.corrcoef(pd[col].values.T)
    seaborn.set(font_scale=0.5)
    heatmap = seaborn.heatmap(correlation, cbar=True, annot=True, square=True, yticklabels=col, xticklabels=col)
    heatmap.get_figure().savefig(spath+"correlation.jpg", dpi=200)


    c = pd.plot(x='volatile acidity', y='quality', style='o')
    c.get_figure().savefig(spath+"1.jpg", dpi=150)

    c = pd.plot(x='citric acid', y='quality', style='o')
    c.get_figure().savefig(spath + "2.jpg", dpi=150)

    c = pd.plot(x='residual sugar', y='quality', style='o')
    c.get_figure().savefig(spath + "3.jpg", dpi=150)

    c = pd.plot(x='chlorides', y='quality', style='o')
    c.get_figure().savefig(spath + "4.jpg", dpi=150)

    c = pd.plot(x='free sulfur dioxide', y='quality', style='o')
    c.get_figure().savefig(spath + "5.jpg", dpi=150)

    c = pd.plot(x='total sulfur dioxide', y='quality', style='o')
    c.get_figure().savefig(spath + "6.jpg", dpi=150)

    c = pd.plot(x='density', y='quality', style='o')
    c.get_figure().savefig(spath + "7.jpg", dpi=150)

    c = pd.plot(x='pH', y='quality', style='o')
    c.get_figure().savefig(spath + "8.jpg", dpi=150)

    c = pd.plot(x='sulphates', y='quality', style='o')
    c.get_figure().savefig(spath + "9.jpg", dpi=150)

    c = pd.plot(x='alcohol', y='quality', style='o')
    c.get_figure().savefig(spath + "10.jpg", dpi=150)



    return render_template("admin/datamining.html",col=col)


@app.route('/predictionload')
def predictionload():
    return render_template('admin/Form.html', pred4=-1)

@app.route('/predictionloadpost',methods=['post'])
def predictionloadpost():

    f0=float(request.form["0"])
    f1=float(request.form["1"])
    f2=float(request.form["2"])
    f3=float(request.form["3"])
    f4=float(request.form["4"])
    f5=float(request.form["5"])
    f6=float(request.form["6"])
    f7=float(request.form["7"])
    f8=float(request.form["8"])
    f9=float(request.form["9"])
    f10=float(request.form["10"])

    test=[[f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]]

    import pandas as pd
    df = pd.read_csv("C:\\project\\manjima\\manjima\\static\\winequality-red.csv")
    y = df.quality
    x = df.drop('quality', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)

    from sklearn.svm import SVC, LinearSVC
    svc_lin = SVC(kernel='linear', random_state=0)
    svc_lin.fit(x_train, y_train)
    pred1 = svc_lin.predict(test)

    # support vector machiene algorithm(RBF clasiifier)
    from sklearn.svm import SVC
    svc_rbf = SVC(kernel='rbf', random_state=0)
    svc_rbf.fit(x_train, y_train)
    pred2 = svc_rbf.predict(test)


    # naive bayes algorithm
    from sklearn.naive_bayes import GaussianNB
    guass = GaussianNB()
    guass.fit(x_train, y_train)
    pred3 = guass.predict(test)

    # Random forest algorithm
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    forest.fit(x_train, y_train)
    pred4 = forest.predict(test)

    return render_template('admin/Form.html',pred1=pred1[0],pred2=pred2[0],pred3=pred3[0],pred4=pred4[0])


if __name__ == '__main__':
    app.run(threaded=False)
