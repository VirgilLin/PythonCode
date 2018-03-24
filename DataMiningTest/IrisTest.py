import numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from collections import defaultdict
from operator import itemgetter
###################################################数据预处理#####################################################################################
dataset = load_iris()
X = dataset.data
y = dataset.target
# print(dataset.DESCR)
n_samples,n_features = X.shape  #n_samples为样例个数，n_features为个体的特征
#离散化特征值
# print(X)
attribute_means = X.mean(axis=0)# data的平均值，axis=0表示列， axis=1表示行以上结果为 ：  array([ 5.84333333,  3.054     ,  3.75866667,  1.19866667])
assert attribute_means.shape == (n_features,)  #保准每个样本都有特征
X_d = np.array(X >= attribute_means,dtype = 'int')   #特征值大于平均值则为1，小于则为0
# print(X_d)
# 设置随机值为固定值，方便检查
random_state = 14  #随机划分训练集和测试集的随机数
#X_train为训练集，X_test为测试集，y_train为训练集的类别信息，y_test为测试集的类别信息
X_train,X_test,y_train,y_test = train_test_split(X_d,y,random_state=random_state)#训练集默认为四分之一
print(X_test)
# print(y_test)
print("There are {0} training samples".format(y_train.shape))
print("There are {0} testing samples".format(y_test.shape))
#划分好测试集和训练集，下面可以开始分类，这里实现的是ONER算法
###################################################数据预处理结束####################################################################
# 定义训练函数

# 参数：

#     X：二维数组，行表示一个植物（实例），列表示特征

#     y_true：每个植物所属类别

#     feature：特征编号
###############################################################训练函数##################################################
def train(X,y_true,feature):
    n_samples, n_features = X.shape  # n_samples为样例个数，n_features为个体的特征
    assert 0<=feature<n_features   #检查遍历是否合法
    values = set(X[:,feature])      #获得该特征的所有特征值{0,1}
    predictors = dict()             #预测器
    errors = []                     #错误率
    for current_value in values:   #遍历所有的特征值（只有0和1）
        most_frequent_class,error = train_feature_value(X,y_true,feature,current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)    #正反面的错误率相加
    total_error = sum(errors)
    return predictors,total_error

#计算某特征出现次数最多的类别，在其他类别出现的次数
def train_feature_value(X,y_ture,feature_index,value):
    class_counts = defaultdict(int)     #存储特征出现次数的字典
    for sample,y in zip(X,y_ture):    #计算每一个特征出现的次数
        if sample[feature_index] == value: #该特征是否符合特征值
            class_counts[y] += 1       #出现符合特征的时候，该类数组数目加一
    #找出各个特征出现最多的类别
    sorted_class_counts = sorted(class_counts.items(),key = itemgetter(1),reverse=True)#各个类别排序,降序排列
    most_frequent_class = sorted_class_counts[0][0]        #出现特征最多的种类
    #除了出现次数最多的类别之外的各个取值的错误率相加
    #n_samples = X.shape[1]
    error = sum([class_count for class_value,class_count in class_counts.items()
                 if class_value !=most_frequent_class])   #错误个数，不是错误的百分比，如果class_value不是
                                                          #most_frequent_class,则把class_count加起来
    return most_frequent_class,error

##############################测试算法#############################
#测试集和训练集上面已经区分
#计算所有的预测器
# all_predictors = {variable:train(X_train,y_train,variable in range(X_train.shape[1]))}
all_predictors = {}
errors = {}
for feature_index in range(X_train.shape[1]):
    predictors,total_error = train(X_train,y_train,feature_index)    #算法计算出当2（第三）特征错误率最低，当2（第三）特征为1时，则是第三种花，当2（第三）特征为0时，则是第一种花
    all_predictors[feature_index] = predictors  #装入所有预测值
    errors[feature_index] = total_error         #装入所有错误率
best_variable,best_error = sorted(errors.items(),key=itemgetter(1))[0]
print("The best model is based on variable{0} and has error {1:.2f}".format(best_variable,best_error))
#选择最好的模型
model = {'variable':best_variable,
         'predictor':all_predictors[best_variable]#[0]    #把最好的分类的预测取出
         }
print(model)

def predict(X_test,model):#预测结果
    variable = model['variable']     #用来分类的特征
    predictor = model['predictor']   #分类的预测器
    y_predicted = np.array([predictor[int(sample[variable])]for sample in X_test])#2特征为1时，设置为2；为0时为0
    return y_predicted
#实现
y_predicted = predict(X_test,model)
print(y_predicted)#   预测
print(y_test)     #   测试
accuracy = np.mean(y_predicted==y_test)*100
print("The test accuracy is {0:.1f}%".format(accuracy))