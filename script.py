# Load the dataset
crops = pd.read_csv("./soil_measures.csv")

# Extract the features & target
extracted_features = crops.drop(columns=["crop"])
features = extracted_features.columns.to_list()
Y = crops["crop"]

# Initiate the dictionary of performance results
feature_performance = {"N":None, "P":None, "K":None, "ph":None}
performance_values = []

# Begin training & predict for each feature
for feature in features:
    feature_val = crops[feature]
    X_train, X_test, y_train, y_test = train_test_split(np.array(feature_val).reshape(-1, 1), Y, test_size=0.2, random_state=42)
    log_reg = LogisticRegression(multi_class="multinomial")
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    score = metrics.f1_score(y_test, y_pred, average="weighted")
    feature_performance[feature] = score

for feature in features:
    performance_values.append(feature_performance[feature])

for feature, val in feature_performance.items():
    print(f"F1-score for {feature}: {val}")
    
best_feature = max(performance_values)
key = [k for k, v in feature_performance.items() if v == best_feature]
real_key = None

for n in key:
    real_key = n

best_predictive_feature = {real_key: best_feature}
print(best_predictive_feature)
