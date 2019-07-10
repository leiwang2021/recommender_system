def Recommend(train,user_id,W,K):  
    rank = dict()  
    ru = train[user_id]  
    for i,pi in ru.items(): 
        tmp=W[i]
        for j,wj in sorted(tmp.items(),key=lambda d: d[1],reverse=True)[0:K]: 
            if j not in rank.keys():
                rank[j]=0
            ##r如果用户已经购买过，则不再推荐
            if j in ru:  
                continue  
            ##待推荐的书本j与用户已购买的书本i相似，则累加上相似分数
            rank[j] += pi*wj  
    return rank  

if __name__ == '__main__':  
    Train_Data = {'A':{'i1':1,'i2':1 ,'i4':1},  
     'B':{'i1':1,'i4':1},  
     'C':{'i1':1,'i2':1,'i5':1},
     'D':{'i2':1,'i3':1},
     'E':{'i3':1,'i5':1},
     'F':{'i2':1,'i4':1}
        }  
    W=ItemSimilarity_alpha(Train_Data)
    Recommend(Train_Data,'C',W,3)  