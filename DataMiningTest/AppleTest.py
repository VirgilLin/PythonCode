import numpy as np
from collections import defaultdict
from operator import itemgetter
valid_rules = defaultdict(int)
invalid_rules = defaultdict(int)
num_occurances = defaultdict(int)
datafile_name = "affinity_dataset.txt"
X = np.loadtxt(datafile_name)
n_samples, n_features = X.shape
features = ["bread","milk","cheese","apples","banana"]
print(X[:5])
for sample in X:
    for premise in range(5):
        if sample[premise] == 0:continue
        num_occurances[premise] += 1   #统计买苹果的总数
        for conclusion in range(n_features):
            if premise == conclusion:
                continue  #跳过买了苹果又买苹果
            if sample[conclusion] == 1:
                valid_rules[(premise,conclusion)] += 1 #买了苹果又买了其他的东西
            else:
                invalid_rules[(premise,conclusion)] += 1  #买了苹果但没买其他的东西
support = valid_rules  #存放结果的字典
confidence = defaultdict(float)
for premise,conclusion in valid_rules.keys():
    # print(premise)
    confidence[(premise,conclusion)] = valid_rules[(premise,conclusion)]/num_occurances[premise]
##############################直接输出，confidence[]中数据是无序的############################
# for premise_count,conclusion_count in confidence:
#     # print(premise_count)
#     premise_name = features[premise_count]
#     conclusion_name = features[conclusion_count]
#     print("If a person buys {0} they will also buy {1}".format(premise_name,conclusion_name))
#     print("Confidence:{0:.3f}".format(confidence[(premise_count,conclusion_count)]))
#     print("Support:{0}".format(support[(premise_count,conclusion_count)]))
#     print("")
##############################################################################################
def print_rule(premise,conclusion,support,confidence,features):
    premise_name = features[premise]
    conclusion_name = features[conclusion]
    print("If a person buys {0} they will also buy {1}".format(premise_name, conclusion_name))
    print("Confidence:{0:.3f}".format(confidence[(premise,conclusion)]))
    print("Support:{0}".format(support[(premise,conclusion)]))
    print("")
sorted_support = sorted(support.items(),key = itemgetter(1),reverse = True)#根据支持度排序
sorted_confidence = sorted(confidence.items(),key = itemgetter(1),reverse = True)#根据置信度排序
for index in range(5):
    print("Rule #{0}".format(index + 1))
    (premise,conclusion) = sorted_support[index][0]
    #(premise,conclusion) = sorted_confidence[index][0]
    print_rule(premise,conclusion,support,confidence,features)
