import pandas as pd
df=pd.read_csv("D:\\python project\\backend\\ml\\student-performance-ml\\data\\student_data.csv")
x=df[["Hours","Sleep","Attendance"]]
y=df["Marks"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
import pickle
pickle.dump(model,open("model.pkl","wb"))
print("Model trained and saved as model.pkl")
from sklearn.metrics import r2_score
y_pred=model.predict(x_test)
score=r2_score(y_test, y_pred)
percentage=score*100
print(f"Model Accuracy: {percentage:.2f}%")