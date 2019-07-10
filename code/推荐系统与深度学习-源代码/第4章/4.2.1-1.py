def ItemSimilarity(train):  
    C = dict()   ##书本对同时被购买的次数
    N = dict()   ##书本被购买用户数
    for u,items in train.items():
        for i in items.keys(): 
            if i not in N.keys():
                N[i]=0
            N[i] += 1  
            for j in items.keys():  
                if i == j:  
                    continue  
                if i not in C.keys():
                    C[i]=dict()
                if j not in C[i].keys():
                    C[i][j]=0
                ##当用户同时购买了i和j，则加1
                C[i][j] += 1  
    W = dict()  ##书本对相似分数
    for i,related_items in C.items():
        if i not in W.keys():
            W[i]=dict()        
        for j,cij in related_items.items(): 
            W[i][j] = cij / math.sqrt( N[i] * N[j])  
    return W  

if __name__ == '__main__':  
    Train_Data = {'A':{'i1':1,'i2':1 ,'i4':1},  
     'B':{'i1':1,'i4':1},  
     'C':{'i1':1,'i2':1,'i5':1},
     'D':{'i2':1,'i3':1},
     'E':{'i3':1,'i5':1},
     'F':{'i2':1,'i4':1}
        }  
    W=ItemSimilarity(Train_Data)
