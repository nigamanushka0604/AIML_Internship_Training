import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import joblib

# Load data
data = pd.read_csv('House_Rent_Dataset.csv')

# Split features and target
y = data['Rent']
x = data.drop(['Rent'], axis=1)
x = x.drop(['Posted On', 'Floor', 'Area Locality'], axis=1, errors='ignore')

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Build pipeline
pipeline = Pipeline([
    ('encoder', ce.LeaveOneOutEncoder()),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Train
pipeline.fit(x_train, y_train)

# Evaluate
score = pipeline.score(x_test, y_test)
print("R² score on test data:", score)

# Save the trained pipeline
joblib.dump(pipeline, "house_rent_pipeline.pkl")
print("Pipeline trained and saved as house_rent_pipeline.pkl")