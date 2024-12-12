# I did the actual training in a jupyter notebook, just copying here to show what I did

# from src.finetune_data_cleaning.bert_embeddings import BertFeatureExtractor
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report, accuracy_score
# extractor = BertFeatureExtractor()
# # train_embeddings = extractor.transform(jdb_train)

# X_train = train_embeddings
# y_train = jdb_train['contains_person'].values
# X_train, X_val, y_train, y_val = train_test_split(train_embeddings, y_train, test_size=0.2, random_state=42)
# C = 1
# svm_model = SVC(kernel='rbf', C=C, gamma=0.001,random_state=42, probability=True)
# svm_model.fit(X_train, y_train)
# y_val_pred =  svm_model.predict_proba(X_val)
# thres = 0.5
# y_val_pred = np.where(y_val_pred[:,1]>thres,1,0)
# val_accuracy = accuracy_score(y_val, y_val_pred)
# print(val_accuracy)
# print(classification_report(y_val, y_val_pred))

# if __name__ == "__main__":
#     print('testing imports')