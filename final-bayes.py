# Natural Language Processing

# Importing the libraries


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import os
import re
import json
path = r'/Users/nick/Desktop/nip_All'                    #  获取文件目录，下面是所有的表格
 
#reference from https://blog.csdn.net/weixin_41768008/article/details/111220577

# 新建列表，存放文件名
file_list = []
 
# 新建列表存放每个文件数据(依次读取多个相同结构的Excel文件并创建DataFrame)
dfs = []

for root,dirs,files in os.walk(path): # 第一个为起始路径，第二个为起始路径下的文件夹，第三个是起始路径下的文件。
    for file in files:
        file_path = os.path.join(root, file)
        file_list.append(file_path) # 使用os.path.join(dirpath, name)得到全路径
        
        with open(file_path) as json_data:
            data = json.load(json_data)
            df =pd.DataFrame.from_dict(data, orient='index').T.set_index('conference')
            i = pd.DataFrame.from_dict(df['reviews'][0][0],orient='index').T
            m=df.iloc[:,[1,5]].reset_index()
            j=pd.concat([m, i],axis=1) # axis=0 as default
        # 将excel转换成DataFrame
        dfs.append(j)   # 多个df的list
# 将多个DataFrame合并为一个
df1 = pd.concat(dfs)
df1['accepted']=df1['accepted'].astype('int')
df1=df1.iloc[::,1:10]
# Importing the dataset
path = r'/Users/nick/Downloads/PeerRead-master/data/iclr_all'                    #  获取文件目录，下面是所有的表格
 
# 新建列表，存放文件名
file_list = []
 
# 新建列表存放每个文件数据(依次读取多个相同结构的Excel文件并创建DataFrame)
dfs = []

for root,dirs,files in os.walk(path): # 第一个为起始路径，第二个为起始路径下的文件夹，第三个是起始路径下的文件。
    for file in files:
        file_path = os.path.join(root, file)
        file_list.append(file_path) # 使用os.path.join(dirpath, name)得到全路径
        
        with open(file_path) as json_data:
            data = json.load(json_data)
            df =pd.DataFrame.from_dict(data, orient='index').T.set_index('conference')
            i = pd.DataFrame.from_dict(df['reviews'][0][0],orient='index').T
            m=df.iloc[:,[1,5]].reset_index()
            j=pd.concat([m, i],axis=1) # axis=0 as default
        # 将excel转换成DataFrame
        dfs.append(j)   # 多个df的list
# 将多个DataFrame合并为一个
df2 = pd.concat(dfs)
df2['accepted']=df2['accepted'].astype('int')
df3=pd.concat([df1,df2])
# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")
eng_stopwords  = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
from nltk.stem.porter import *
def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    ps = PorterStemmer()
    words = [ps.stem(w)  for w in words if w not in eng_stopwords]
    return ' '.join(words)

df3['clean_comments'] = df3.comments.apply(clean_text)
corpus =list(df3['clean_comments'])
#corpus.append(df4['clean_comments'])

df3.to_csv('out.csv')

df4=pd.read_csv(r'out.csv')


'''

df3[['y','con'] ]= df3['conference'] .str.extract(r'[0-9]+|[a-z]+' , expand=False)
eng_stopwords = set(stopwords)
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ',df4['comments'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(w)  for w in words if w not in eng_stopwords]
    review = ' '.join(review)
    corpus.append(review)
'''

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(ngram_range=(2,2),tokenizer=token.tokenize,max_features = 6000)
X7 = cv.fit_transform(corpus).toarray()
y7 = df4.iloc[:, 4].values



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X7, y7, test_size = 0.20, random_state = 0)

#under_samplimg
from imblearn.under_sampling import RandomUnderSampler
sampler = RandomUnderSampler(random_state=2,replacement=True)
X_rs, y_rs = sampler.fit_resample(X2 , y2 )


#SMOTE
from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

m=y_train[y_train==0]
mm=y_train[y_train==1]

#MODEL
from sklearn.naive_bayes import MultinomialNB  #朴素贝叶斯
from sklearn.model_selection import GridSearchCV    #网格搜索和交叉验证
estimator = MultinomialNB()
#准备参数
param_dict = {
    "alpha":[1,2,3]} #ealpha：拉普拉斯平滑系数值设定的可能取值
estimator = GridSearchCV(estimator,param_grid=param_dict,cv=10) 
#执行预估器，训练模型
estimator.fit(X_train,y_train)

#estimation
#方法1：比对真实值和预测值
y_pred = estimator.predict(X_test)  #计算预测值
print(y_pred)
y_test == y_pred  #比对真实值和预测值，相同的返回True
#方法2：直接计算准确率
accuracy=estimator.score(X_test,y_test)
print(accuracy)

# 3、check result of CV
# 最佳参数：best_params_
print("best_params:",estimator.best_params_)
# 验证集的最佳结果：best_score_
print("best_score：",estimator.best_score_)
# 最佳估计器：best_estimator_
print("best_estimator",estimator.best_estimator_)
# 交叉验证结果：cv_results_
print(estimator.cv_results_)  

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import accuracy_score
p=accuracy_score(y_test, y_pred)
from sklearn.metrics import f1_score
f1=f1_score(y_test, y_pred)
from sklearn.metrics import recall_score
recall=recall_score(y_test, y_pred)
print(p)
print(f1)
print(recall)

#导入要用的库
import sklearn.metrics as metrics 
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score as AUC
import matplotlib.pyplot as plt

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc=metrics.auc(fpr,tpr)

from sklearn.metrics import roc_curve
from sklearn.metrics import auc



# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('ROC curve of naive-bayes(bi-gram)')
plt.legend(loc="lower right")



#LENGTH
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import os
import re
import json
k1=pd.read_csv(r'out.csv')
c1= k1.groupby("conference").count()

c1 = k1.value_counts("conference")
print(c1)
k1.value_counts("conference").unique()

plt.bar(k1.conference.unique(),
        k1.value_counts("conference"), 
        width=0.5, 
        bottom=None, 
        align='center', 
        color=['lightsteelblue', 
               'cornflowerblue', 
               'royalblue', 
               'midnightblue', 
               'navy', 
               'darkblue', 
               'mediumblue'])
plt.xticks(rotation='vertical')
plt.show()
c1.to_csv('c1.csv')
c1=pd.read_csv(r'c1.csv')


s=c1['a'].value_counts()
s=s.to_frame()
s.columns = ['count']
s.to_csv('s.csv')
s=pd.read_csv(r's.csv')
index = s['conference'].unique()
a=s[s['accepted']==0]
b=s[s['accepted']==1]

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = k1.iloc[:, 15]
y = k1.iloc[:, 4].values

'''
# Splitting the dataset into the Training set and Test set
#分成測試跟訓練
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

print(y_train[:10])
'''

#cal length
import matplotlib.pyplot as plt
import numpy as np
text_len_li = list(map(len, X))
print("the shortest length", min(text_len_li))
print("the longest length=", max(text_len_li))
print("average length=", np.mean(text_len_li))
plt.hist(text_len_li, bins=range(min(text_len_li), max(text_len_li)+50, 50))
plt.title("Text length distribution")
plt.show()


def realdata(papertexts):
    s = ""
    for i in range(len(papertexts)):
        w_list = papertexts[i].split()
        indexvalue= w_list.index("abstract")+1 if "abstract" in w_list else 0
        s = s+ " ".join( w_list[indexvalue: ] )
    return s
