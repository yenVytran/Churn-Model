import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#các thư viện yêu cầu
df=pd.read_csv("Churn_Modelling.csv")
#đọc dữ liệu
df.head()
df.describe()
#Mô tả thống kê dữ liệu
df.info()
df.shape
#in ra kích thước dữ liệu
df.isnull().sum()
#kiểm tra dữ liệu khuyết thiếu
df.duplicated().sum()
#Kiểm tra dữ liệu bị trùng lặp
df=pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
# Mã hóa biến phân loại Geography và Gender bằng One-Hot Encoding
df
df.nunique()

ax=sns.countplot(x=df['Exited'], data=df);
plt.title('Exited')
plt.show()
plt.title('Target column Distribution')
plt.pie(df['Exited'].value_counts(),labels = ['Not Exited', 'Exited'],autopct='%.1f%%',explode=(0,0.1),startangle=90,shadow=True)
plt.show()
#Vẽ biểu đồ thống kê số lượng khách hàng rời bỏ ngân hàng

cat_features = ['Geography_Spain','Geography_Germany','Gender_Male','NumOfProducts','HasCrCard','IsActiveMember']
num_features = ['Tenure','Balance','EstimatedSalary','Age','CreditScore']
for i in cat_features:
    ax=sns.countplot(x=df[i], data=df);
    plt.title(i)
    plt.show()
# Xem xét mối tương quan giữa biến mục tiêu và các biến phân loại

for i in num_features:
    sns.boxplot(x='Exited', y=df[i], data=df)
    plt.title(i)
    plt.show()

df
""" Xây dựng mô hình """
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X = df.drop("Exited", axis=1)
y = df.Exited

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Khởi tạo StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Lưu trữ kết quả
accuracies = []

# Vòng lặp cross-validation
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Xử lý dữ liệu mất cân bằng bằng ADASYN
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Tinh chỉnh siêu tham số bằng RandomizedSearchCV
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Khởi tạo RandomForestClassifier
    rf_model = RandomForestClassifier(random_state=42)

    random_search = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=50, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
    random_search.fit(X_train_resampled, y_train_resampled)

    best_model = random_search.best_estimator_

    # Dự đoán và đánh giá
    y_pred = best_model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))
# In kết quả
print(f'Accuracy Scores for each fold: {accuracies}')
print(f'Mean Accuracy: {sum(accuracies) / len(accuracies)}')