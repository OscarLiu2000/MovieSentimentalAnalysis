from flask import Flask,request
import  pandas as pd
import random
import re
import json
import pickle
from gensim import similarities,models,corpora
import  numpy as np
app = Flask(__name__)

data=pd.read_csv('./labeledTrainData.tsv',sep='\t')
LR=pickle.load(open('./LR.model','rb'))
doc2vec=models.Doc2Vec.load('d2v.bin')
trainData, trainLabel, testData, testLabel=pickle.load(open('./data.bin', 'rb'))


#获得一条评论


@app.route('/')
def index():
    with open('./index.html',encoding='utf-8') as f:
        return f.read()


@app.route('/review')
def review():
    index=random.sample(range(len(data.index)),1)[0]
    review=re.sub(r'<.*?>','',data.loc[index]["review"])
    sentiment=data.loc[index]["sentiment"];
    return json.dumps({"review":review,"sentiment":str(sentiment)})
@app.route('/predict', methods=['POST'])
def predict():
    review=request.form['review']
    stop_words = [" "];
    # 去掉HTML标签
    review = re.sub(r'<.*?>', '', review)
    review_list = [w.strip('.,?!\\"\'') for w in review.split(' ') if w not in stop_words]
    vec=doc2vec.infer_vector(review_list)
    label = LR.predict(np.array(vec).reshape(1, -1))
    return str(label[0])





if __name__ == '__main__':
    app.run()