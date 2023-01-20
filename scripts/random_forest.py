from sklearn.ensemble import RandomForestClassifier
from utils import read_data, plot_cf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# Read Data
X_train, y_train = read_data('train')
X_test, y_test = read_data('test')

# init classifier
rf = RandomForestClassifier(n_estimators=250, random_state=0)

# Train
rf.fit(X_train, y_train)

# Test
y_pred = rf.predict(X_test)

# Classification Evaluation
plot_cf(confusion_matrix(y_test, y_pred), "Random Forest")
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Store trained model
pickle.dump(rf, open('models/rf.pkl', 'wb'))
