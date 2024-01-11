import image_processing as imp
import bovw as bw
import numpy as np
import cv2
import joblib
import os
from tqdm import tqdm   

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def save_svm_model(svm_model, file_path):
    print('saving svm_model!!!')
    # k = svm_model.shape[0]
    joblib.dump(svm_model, f'{file_path}/svm_model.pkl', compress=3)

def save_svm_pca_model(svm_model, file_path):
    print('saving svm_pca_model!!!')
    # k = svm_model.shape[0]
    joblib.dump(svm_model, f'{file_path}/svm_model_PCA_230d.pkl', compress=3)
def save_features(data, filepath):
    print('saving feature!!!')  
    joblib.dump(data, filepath)
    print('saved')

def load_features(file_path):
    print('loading file')
    print('loaded')
    return joblib.load(file_path)

# Khai báo số cụm:
n_cluster = 1000

# Khai báo các đường dẫn: 
train_base_path = r'baocao_computervision\dataset _v2\train'

train_grayimages_path = r'baocao_computervision\data\train\images\gray\grayimages.joblib'
train_descriptors_path = r'baocao_computervision\data\train\descriptors\image_descriptors.joblib'
train_labelWindex_path = r'baocao_computervision\data\train\labelWindex\labelWindex.joblib'

valid_grayimages_path = r'baocao_computervision\data\valid\images\gray\grayimages.joblib'
valid_descriptors_path = r'baocao_computervision\data\valid\descriptors\image_descriptors.joblib'
valid_labelWindex_path = r'baocao_computervision\data\valid\labelWindex\labelWindex.joblib'



codebook_model_path = r'baocao_computervision\model\codebook\codebook_1000.pkl'

# Load train_image:
if( not os.path.exists(train_descriptors_path)):
    train_brg_image, train_gray_images, train_labelWindex = imp.load_images(train_base_path)
    _, train_descriptors = imp.extract_visual_features(train_gray_images)
    
    save_features(train_gray_images, train_grayimages_path )
    save_features(train_labelWindex, train_labelWindex_path)
    save_features(train_descriptors, train_descriptors_path)


train_grayimages = load_features(train_grayimages_path)
train_descriptors = load_features(train_descriptors_path)
train_labelWindex = load_features(train_labelWindex_path)

# Load valid_image:
valid_base_path = r'baocao_computervision\dataset _v2\valid'
if( not os.path.exists(valid_descriptors_path)):
    valid_brg_image, valid_gray_images, valid_labelWindex = imp.load_images(valid_base_path)
    _, valid_descriptors = imp.extract_visual_features(valid_gray_images)
    
    save_features(valid_gray_images, valid_grayimages_path )
    save_features(valid_labelWindex, valid_labelWindex_path)
    save_features(valid_descriptors, valid_descriptors_path)


valid_grayimages = load_features(valid_grayimages_path)
valid_descriptors = load_features(valid_descriptors_path)
valid_labelWindex = load_features(valid_labelWindex_path)


# Thực hiện xây dựng codebook:

if( not os.path.exists(codebook_model_path)):
    n_centroids = bw.build_codebook(train_descriptors, n_cluster)
    bw.save_codebook(n_centroids, r'baocao_computervision\model\codebook')

n_centroids = bw.load_codebook(codebook_model_path)

train_vectors = []
print('represent vector!')
for item in tqdm(train_descriptors):
    train_vectors.append(bw.represent_image_features(item, n_centroids))
    
train_labels = [item[1] for item in train_labelWindex]

valid_vectors = []
print('represent vector!')
for item in tqdm(valid_descriptors):
    valid_vectors.append(bw.represent_image_features(item, n_centroids))
    
valid_labels = [item[1] for item in valid_labelWindex]

# Thực hiện xây dựng mô hình SVM
#_______________________________________
model = SVC(kernel='linear', C = 30)

model.fit(train_vectors, train_labels)

path_svm_model = r'baocao_computervision\model\smv'
file_svm_model = r'baocao_computervision\model\smv\svm_model.pkl'

save_svm_model(model, path_svm_model)
accuracy = model.score(valid_vectors, valid_labels)
pred_labels = model.predict(valid_vectors)
for t_label, p_label in zip(valid_labels, pred_labels):
    print(f'true: {t_label}, pred: {p_label}')

print('accuracy: ', accuracy)
input("continue?")
#________________________________________

# Thực hiện xây dựng gridsearch để tối ưu mô hình SVM:


# svm_model = SVC()

# Định nghĩa các tham số cần tối ưu
# param_grid = {'C': [0.01, 0.1,  1, 10, 100], 'kernel': ['linear', 'rbf', 'poly'], 'gamma': [0.1, 0.01, 0.001]}

# # Sử dụng GridSearchCV để tìm kiếm các tham số tốt nhất
# grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(train_vectors, train_labels)

# # Hiển thị kết quả tối ưu
# print("Các tham số tối ưu là: ", grid_search.best_params_)
# print("Điểm chính xác tốt nhất trên tập kiểm tra: {:.2f}".format(grid_search.best_score_))

# # Đánh giá mô hình trên tập kiểm tra
# test_accuracy = grid_search.score(valid_vectors, valid_labels)
# print("Điểm chính xác trên tập kiểm tra: {:.2f}".format(test_accuracy))

# pred_labels = grid_search.predict(valid_vectors)
# for t_label, p_label in zip(valid_labels, pred_labels):
#     print(f'true: {t_label}, pred: {p_label}')
# print(len(valid_labels))

# from sklearn.neighbors import KNeighborsClassifier

# knn_model = KNeighborsClassifier()

# # Định nghĩa các tham số cần tối ưu
# param_grid = {'n_neighbors': [1, 3, 5, 7], 'weights': ['uniform', 'distance'], 'p': [1, 2]}

# # Sử dụng GridSearchCV để tìm kiếm các tham số tốt nhất
# grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(train_vectors, train_labels)

# # Hiển thị kết quả tối ưu
# print("Các tham số tối ưu là: ", grid_search.best_params_)
# print("Điểm chính xác tốt nhất trên tập kiểm tra: {:.2f}".format(grid_search.best_score_))

# # Đánh giá mô hình trên tập kiểm tra
# test_accuracy = grid_search.score(valid_vectors, valid_labels)
# print("Điểm chính xác trên tập kiểm tra: {:.2f}".format(test_accuracy))


#  Thực hiện PCA giảm số chiều của các vector :
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
train_vectors_standardized = scaler.fit_transform(train_vectors)
valid_vectors_standardized = scaler.transform(valid_vectors)
scaler_file_path = r'baocao_computervision\model\smv_pca\scaler.pkl'
save_features(scaler,scaler_file_path )

# x_axis = []
# y_axis = []

# # Thực hiện PCA trên tập huấn luyện
# for n_comp in range(100, 232, 2):
#     pca = PCA(n_components=n_comp)  # Chọn số thành phần chính mong muốn
#     X_train_pca = pca.fit_transform(train_vectors_standardized)

#     # Áp dụng phép biến đổi PCA đã học từ tập huấn luyện vào tập kiểm thử
#     X_test_pca = pca.transform(valid_vectors_standardized)

#     model = SVC(kernel='linear', C = 30)

#     model.fit(X_train_pca, train_labels)

#     path_svm_model = r'baocao_computervision\model\smv'
#     file_svm_model = r'baocao_computervision\model\smv\svm_model.pkl'

#     # save_svm_model(model, path_svm_model)
#     accuracy = model.score(X_test_pca, valid_labels)
#     pred_labels = model.predict(X_test_pca)
#     # for t_label, p_label in zip(valid_labels, pred_labels):
#     #     print(f'true: {t_label}, pred: {p_label}')

#     print(f'accuracy with {n_comp}: ', accuracy)
#     x_axis.append(n_comp)
#     y_axis.append(accuracy)

# import matplotlib.pyplot as plt
# plt.plot(x_axis, y_axis, marker='.', label='Training Accuracy')
# plt.bar(x_axis, y_axis)

# # Đặt tên trục và tiêu đề
# plt.xlabel('Số lượng chiều')
# plt.ylabel('Độ chính xác')
# plt.title('Độ chính xác tương ứng với số chiều')

# # Hiển thị chú thích
# plt.legend()

# # Hiển thị biểu đồ
# plt.show()


# Thực hiện xây dựng ma trận nhầm lẫn:
pca = PCA(n_components=230)  # Chọn số thành phần chính mong muốn
X_train_pca = pca.fit_transform(train_vectors_standardized)

pca_transform_path = r'baocao_computervision\model\smv_pca\pca_model.pkl'

save_features(pca, pca_transform_path)

# Áp dụng phép biến đổi PCA đã học từ tập huấn luyện vào tập kiểm thử
X_test_pca = pca.transform(valid_vectors_standardized)

model = SVC(kernel='linear', C = 30)

model.fit(X_train_pca, train_labels)

path_svm_model = r'baocao_computervision\model\smv_pca'
file_svm_model = r'baocao_computervision\model\smv\svm_model_PCA_230d.pkl'

save_svm_pca_model(model, path_svm_model)
accuracy = model.score(X_test_pca, valid_labels)
pred_labels = model.predict(X_test_pca)
for t_label, p_label in zip(valid_labels, pred_labels):
    print(f'true: {t_label}, pred: {p_label}')

print(f'accuracy with {230}: ', accuracy)




# from sklearn.neighbors import KNeighborsClassifier

# knn_model = KNeighborsClassifier()

# # Định nghĩa các tham số cần tối ưu
# param_grid = {'n_neighbors': [1, 3, 5, 7], 'weights': ['uniform', 'distance'], 'p': [1, 2]}

# # Sử dụng GridSearchCV để tìm kiếm các tham số tốt nhất
# grid_search = GridSearchCV(knn_model, param_grid, cv=5, scoring='accuracy')
# grid_search.fit(X_train_pca, train_labels)

# # Hiển thị kết quả tối ưu
# print("Các tham số tối ưu là: ", grid_search.best_params_)
# print("Điểm chính xác tốt nhất trên tập kiểm tra: {:.2f}".format(grid_search.best_score_))

# # Đánh giá mô hình trên tập kiểm tra
# test_accuracy = grid_search.score(X_test_pca, valid_labels)
# print("Điểm chính xác trên tập kiểm tra: {:.2f}".format(test_accuracy))
# pred_labels = grid_search.best_estimator_.predict(X_test_pca)



import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# # Tính ma trận nhầm lẫn
cm = confusion_matrix(valid_labels, pred_labels)

# In ra ma trận nhầm lẫn
print("Confusion Matrix:")
print(cm)

# Vẽ biểu đồ heatmap cho ma trận nhầm lẫn


classes = [i for i in range(0, 10)]
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.legend()
plt.show()


