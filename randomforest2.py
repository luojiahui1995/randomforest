import pandas as pd
import numpy as np
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve

# 加载数据
##### 加载训练和测试数据
#####--------------------------------------------------------------------------------------------------
#这里读入数据的时候我们没有做任何的处理（像去除空值这些）
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

train_num,train_var_num = np.shape(train)
test_num,test_var_num = np.shape(test)
print("训练集：有", train_num, "个样本", "每个样本有", train_var_num, "个变量.")
print("测试集：有", test_num, "个样本", "每个样本有", test_var_num, "个变量.")
print(train.info())
print(test.info())

#离群点检测
def detect_outliers(df, n, features):
    '''
    输入：
    df：数据框，为需要检测的样本集
    n：正整数，样本特征超出四分位极差个数的上限，有这么多个特征超出则样本为离群点
    features:列表，用于检测是否离群的特征
    输出：

    '''
    outlier_indices = []
    outlier_list_col_index = pd.DataFrame()

    #对每一个变量进行检测
    for col in features:
        #计算四分位数相关信息
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3-Q1
        #计算离群范围
        outlier_step = 1.5*IQR
        #计算四分位数时如果数据上有空值，这些空值也是参与统计的，所以统计出来的Q1、Q3、IQR这些数据有可能是NAN，但是这并不要紧，在判断是否大于或小于的时候跟NAN比较一定是false，因而样本并不会因为空值而被删除掉
        #空值会在后面特征工程时再做处理

        #找出特征col中显示的离群样本的索引
        outlier_list_col = df[(df[col] < Q1-outlier_step) | (df[col] > Q3 + outlier_step)].index
        #额外存储每一个特征在各样本中的离群判断
        temp = pd.DataFrame((df[col] < Q1-outlier_step) | (df[col] > Q3+outlier_step), columns=[col])
        #将索引添加到一个综合列表中，如果某个样本有多个特征出现离群点，则该样本的索引会多次出现在outlier_indices里
        outlier_indices.extend(outlier_list_col)
        #额外存储每一个特征在各样本中的离群判断，方便查看数据
        outlier_list_col_index = pd.concat(objs=[outlier_list_col_index, temp], axis=1)
    #选出有n个以上特征存在离群现象的样本
    outlier_indices = Counter(outlier_indices)
    multiple_outliers=list(k for k,v in outlier_indices.items() if v>n)
    return multiple_outliers,outlier_list_col_index
#获取离群点
outliers_to_drop, outlier_col_index = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])
#这里选取了"Age","SibSp","ParCh","Fare"四个数值型变量；另一个数值型变量舱位等级没选是因为该变量只有1、2、3级不可能有离群点，其他符号型变量诸如性别、登录港口，也只有有限的类型，一般不可能离群，也没有必要分析是否离群。
print(train.loc[outliers_to_drop])
print(outlier_col_index.loc[outliers_to_drop])#查看哪个特征对样本成为离群点有决定作用.
print(train.describe())

train = train.drop(outliers_to_drop, axis = 0).reset_index(drop=True)#删除离群点

#整合数据集
train_len,train_var_num=np.shape(train)
dataset=pd.concat(objs=[train,test],axis=0).reset_index(drop=True)
dataset=dataset.fillna(np.nan)#数据集缺失值用NAN填充
print(dataset)


#特征间关联性分析
g=sns.heatmap(train[["Survived","Age","SibSp","Parch","Pclass","Fare"]].corr(),annot=True,fmt = ".2f",cmap = "coolwarm")
plt.show()

#年龄Age与生存率的关系
g=sns.kdeplot(train["Age"],color = "Green", shade = True)
g=sns.kdeplot(train["Age"][(train["Age"].notnull())&(train["Survived"]==0)],color="Red",shade=True)
g=sns.kdeplot(train["Age"][(train["Age"].notnull())&(train["Survived"]==1)],color="Blue",shade=True)
g.set_xlabel("Age")
g.set_ylabel("Frequency")
g=g.legend(["All","Not Survived","Survived"])
plt.show()

#性别Sex与生存率的关系
g=sns.barplot(data=train,x="Sex",y="Survived")
g.set_ylabel("Survival Probability")
plt.show()
#船上兄弟姐妹与配偶数与生存率的关系
g=sns.barplot(data=train,x="SibSp",y="Survived")
g.set_ylabel("Survival Probability")
plt.show()
#船上父母或子女数量Parch与生存率的关系
g=sns.barplot(data=train,x="Parch",y="Survived")
g.set_ylabel("Survival Probability")
plt.show()
#船票等级Pclass与生存率的关系
g=sns.barplot(data=train,x="Pclass",y="Survived")
g.set_ylabel("Survival Probability")
plt.show()
#票价Fare与生存率的关系
g=sns.kdeplot(train["Fare"],color="Green",shade=True)
g=sns.kdeplot(train["Fare"][train["Survived"]==0],color="red",shade=True,ax=g)
g=sns.kdeplot(train["Fare"][train["Survived"]==1],color="blue",shade=True,ax=g)
g.set_xlabel("Fare")
g.set_ylabel("Frequency")
g=g.legend(["All","Not Survived","Survived"])
plt.show()

#Fare特征的缺失值进行填充
dataset["Fare"]=dataset["Fare"].fillna(dataset["Fare"].median())#这里用到了整个数据集
#test["Fare"]=test["Fare"].fillna(dataset["Fare"].median())
#下面利用log函数进行数据变换
# dataset["Fare"]=dataset["Fare"].map(lambda i:np.log(i) if i>0 else 0)#map()函数具体将元素进行映射的功能
# #查看变换后的数据分布
# g=sns.distplot(dataset["Fare"],color="M",label="Skewness:%.2f"%(dataset["Fare"].skew()))
# g.legend(loc="best")
# plt.show()

#查看Embarked与生存率的关系
g=sns.barplot(data=train,x="Embarked",y="Survived")
g.set_ylabel("Survival Probability")
plt.show()

print(train["Cabin"].describe())#船舱的情况

#性别Sex的数值化
dataset["Sex"]=dataset["Sex"].map({"male":0,"female":1})
train["Sex"]=train["Sex"].map({"male":0,"female":1})

#填充Age缺失值
#获取Age缺失值索引
index_NaN_age=list(dataset["Age"][dataset["Age"].isnull()].index)
for i in index_NaN_age:
    age_med=dataset["Age"].median()#如果通过关联特征找不到匹配的值，则用整个数据的中值填充
    age_pred=dataset["Age"][((dataset["SibSp"]==dataset.iloc[i]["SibSp"])&(dataset["Parch"]==dataset.iloc[i]["Parch"])&(dataset["Pclass"]==dataset.iloc[i]["Pclass"]))].median()
    if not np.isnan(age_pred):
        dataset["Age"].iloc[i]=age_pred
    else:
        dataset["Age"].iloc[i]=age_med

#填充值后再看一次Age在不同Survived下的分布情况
g=sns.factorplot(data=dataset,x="Survived",y="Age",kind="violin")
plt.show()


#对Name进行处理
#查看Name
print(dataset["Name"].head())
#下面直接提取名字中间部分
dataset_title=[i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"]=pd.Series(dataset_title)
# 查看详情
print(dataset["Title"].describe())
print(dataset["Title"].unique())
dataset["Title"]=dataset["Title"].replace(['Mr','Don'],'Mr')
dataset["Title"]=dataset["Title"].replace(['Mrs','Miss','Mme','Ms','Lady','Dona','Mlle'],'Ms')
dataset["Title"]=dataset["Title"].replace(['Sir','Major','Col','Capt'],'Major')
dataset["Title"]=dataset["Title"].replace(['Rev'],'Rev')
dataset["Title"]=dataset["Title"].replace(['Dr'],'Dr')
dataset["Title"]=dataset["Title"].replace(['Master','Jonkheer','the Countess'],'Jonkheer')

#我们查看各组的幸存率情况：
g=sns.barplot(data=dataset[:train_len],x="Title",y="Survived")
g.set_ylabel("Survival Probability")
plt.show()
#下面将姓名数值化
dataset["Title"]=dataset["Title"].map({'Mr':0,'Ms':1,'Major':2,'Jonkheer':3,'Rev':4,'Dr':5})
dataset["Title"]=dataset["Title"].astype(int)
#将Title哑变量化
dataset=pd.get_dummies(dataset,columns=["Title"],prefix="TL")
# 去掉name这一特征
dataset.drop(labels = ["Name"], axis = 1, inplace = True)
print(dataset.info())

#对Ticket进行处理
Ticket=[]
for i in list(dataset["Ticket"]):
    if not i.isdigit():
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0])
    else:
        Ticket.append("X")
print(Ticket)
dataset["Ticket"]=Ticket
#查看替换后的情况
print(dataset["Ticket"].describe())
#查看不同船票的生存率
g=sns.barplot(data=dataset,x="Ticket",y="Survived")
g.set_ylabel("Survival Probability")
plt.show()

print(dataset["Cabin"].isnull().sum())
#将船舱信息进行替换
dataset["Cabin"]=pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin']])
#再来查看一下船舱信息
print(dataset["Cabin"].describe())
print(dataset["Cabin"].isnull().sum())
#查看不同船舱的幸存率
g=sns.barplot(data=dataset[:train_len],x="Cabin",y="Survived")
g.set_ylabel("Survival Probability")
plt.show()
#利用哑变量将Cabin信息数值化
dataset=pd.get_dummies(dataset,columns=["Cabin"],prefix="Cabin")
#再来查看一下船舱信息
print(dataset.info())
#利用哑变量将Ticket数值化
dataset=pd.get_dummies(dataset,columns=["Ticket"],prefix="T")

#将Embarked哑变量化
dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
#将Pclass哑变量化
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")
#去除passengerID
dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)

#查看最终数据
print(dataset.head())
print(dataset.info())

#重新获取训练数据和测试数据
train=dataset[:train_len]
train["Survived"]=train["Survived"].astype(int)
Y_train=train["Survived"]
X_train=train.drop(labels=["Survived"],axis=1)
test=dataset[train_len:]
test.drop(labels=["Survived"],axis=1,inplace=True)
# 搜索随机森林的最佳参数
RFC = RandomForestClassifier()
## 设置参数网络
rf_param_grid = {"max_depth": [None],
                 "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=None, scoring="accuracy", n_jobs= 1, verbose = 1)
gsRFC.fit(X_train,Y_train)
RFC_best = gsRFC.best_estimator_
print(RFC_best)
# 打印最佳得分
print(gsRFC.best_score_)


# 效果评估
#####--------------------------------------------------------------------------------------------------
### 效果评估之学习曲线
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",X_train,Y_train,cv=None)
plt.show()








