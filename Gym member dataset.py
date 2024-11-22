import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("/content/gym_members_exercise_tracking.csv")
print(df)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
print(df)

y = df['Max_BPM']
print(y)

x = df[["Age","Height (m)","Gender"]]
print(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df[['Gender','Age','Height (m)']],df['Max_BPM'],test_size=0.2)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.linear_model import LogisticRegression as logisticregression
# Create an instance of the LogisticRegression class
logres = logisticregression()
# Now you can call fit on the instance
logres.fit(x_train,y_train)
y_pred = logres.predict(x_test)
print(y_pred)

print(y_test)

print(y_pred)

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix (y_test,y_pred)
print(cnf_matrix)

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
print
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print("Precision:",metrics.precision_score(y_test, y_pred))
# print("Recall:",metrics.recall_score(y_test, y_pred))
