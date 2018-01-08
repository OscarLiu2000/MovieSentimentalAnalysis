import  pandas as pd
from gensim import similarities,models
import random
import  re
import pickle
import  os

if os.path.isfile('data.bin'):
    trainData, trainLabel, testData, testLabel=pickle.load(open('./data.bin', 'rb'))
else:

    data=pd.read_csv('./labeledTrainData.tsv',sep='\t')
    stop_words=[" "];
    sentences=[]
    for index in data.index:
        id=data.loc[index]['id']
        sentiment=data.loc[index]['sentiment']
        review=data.loc[index]['review']
        #去掉HTML标签
        review=re.sub(r'<.*?>','',review)
        review_list=[w.strip('.,?!\\"\'') for w in review.split(' ') if w not in stop_words]
        sentences.append(models.doc2vec.TaggedDocument(review_list,[id]))
    #转换为向量
    doc2vec=models.Doc2Vec(sentences)


    #取30% 作为测试数据
    total = len(data.index)
    testIndexs=random.sample(range(total),int(total*0.3))

    trainData=[]
    trainLabel=[]
    testData=[]
    testLabel=[]
    for index in range(total):
        id = data.loc[index]['id'];
        if index in testIndexs:
            testData.append(doc2vec.docvecs[id])
            testLabel.append(data.loc[index]['sentiment'])
        else:
            trainData.append(doc2vec.docvecs[id])
            trainLabel.append(data.loc[index]['sentiment'])
    doc2vec.save('./d2v.bin')
    pickle.dump((trainData,trainLabel,testData,testLabel),open('./data.bin','wb'))


from sklearn.linear_model import LogisticRegression;
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  GradientBoostingClassifier
from sklearn.svm import LinearSVC
LR=LogisticRegression()
LR.fit(trainData,trainLabel)

RF=RandomForestClassifier()
RF.fit(trainData,trainLabel)

GBDT=GradientBoostingClassifier()
GBDT.fit(trainData,trainLabel)

svm=LinearSVC()
svm.fit(trainData,trainLabel)

#模型分数
print(LR.score(testData,testLabel))
print(RF.score(testData,testLabel))
print(GBDT.score(testData,testLabel))
print(svm.score(testData,testLabel))

pickle.dump(LR,open('./LR.model','wb'))



# MovieSentimentalAnalysis
