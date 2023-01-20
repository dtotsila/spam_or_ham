from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils import read_data, plot_cf
import pickle

# Read Train Test data
X_train, y_train = read_data('train')
X_test, y_test = read_data('test')

# init classifier
dtc = DecisionTreeClassifier(min_samples_split=7, random_state=252)

# train
dtc.fit(X_train, y_train)

# test
y_pred = dtc.predict(X_test)

# Classification Evaluation
plot_cf(confusion_matrix(y_test, y_pred), "Decision Tree Classifier")
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Store trained model
pickle.dump(dtc, open('models/dtc.pkl', 'wb'))
