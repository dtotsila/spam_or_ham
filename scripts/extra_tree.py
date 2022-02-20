from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
import pickle
from utils import read_data, plot_cf

# Read Train Test Data
X_train, y_train = read_data('train')
X_test, y_test = read_data('test')

# init classifier
etc = ExtraTreesClassifier(n_estimators=37, random_state=252)

# Train
etc.fit(X_train,y_train)

# Test
y_pred = etc.predict(X_test)


# Classification Evaluation
plot_cf(confusion_matrix(y_test,y_pred),"Extra Trees Classifier")
print(classification_report(y_test,y_pred)) 
print(accuracy_score(y_test,y_pred))

# Store trained model
pickle.dump(etc, open('models/etc.pkl', 'wb'))
