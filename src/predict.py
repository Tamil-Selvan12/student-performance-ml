import pickle
model=pickle.load(open("model.pkl","rb"))
hours=float(input("Enter hours studied: "))
sleep=float(input("Enter hours of sleep: "))
attendance=float(input("Enter attendance percentage: "))
prediction=model.predict([[hours,sleep,attendance]])
print("Predicted Marks:", prediction[0])
