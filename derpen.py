from sklearn import datasets
from sklearn import svm
import joblib

class DerpenData():
    def __init__(self):
        self.data = datasets.load_iris()
        self.clf = svm.SVC()

    def set_clf(self, clf):
        self.clf = clf

    def train_data(self):
        self.clf.fit(self.data.data[:-1], 
                     self.data.target_names[self.data.target][:-1])

    def predict(self, data):
        try:
            input = int(data)
            input_data = self.data.data[input].reshape(1, -1)
        except Exception as error:
            return error

        try:
            predict_result = self.clf.predict(input_data)
        except Exception as error:
            predict_result = error
        return predict_result
    
    def save_model(self, name="Model"):
        joblib.dump(self.clf, f"{name}.pkl") 
    
    def load_model(self, name):
        self.clf = joblib.load(name)