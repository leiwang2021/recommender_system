def defItemIndex(DictUser):
    DictItem=defaultdict(defaultdict) 
    ##遍历每个用户
    for key in DictUser: 
        ##遍历用户k的购买记录
        for i in DictUser[key]:
            DictItem[i[0]][key]=i[1]
    return DictItem
    
def defUserSimilarity(DictItem):  
    N=dict()   #用户购买的数量
    C=defaultdict(defaultdict)  
    W=defaultdict(defaultdict)  
    ##遍历每个物品
    for key in DictItem: 
        ##遍历用户k购买过的书
        for i in DictItem[key]:  
            #i[0]表示用户的id ，如果未计算过，则初始化为0
            if i[0] not in N.keys():  
                N[i[0]]=0  
            N[i[0]]+=1     
            ## (i,j)是物品k同时被购买的用户两两匹配对          
            for j in DictItem[key]:  
                if i(0)==j(0):  
                    continue    
                if j[0] not in C[i[0]].keys():  
                    C[i[0]][j[0]]=0  
                #C[i[0]][j[0]]表示用户i和j购买同样书的数量  
                C[i[0]][j[0]]+=1      
    for i,related_user in C.items():  
        for j,cij in related_user.items():  
            W[i][j]=cij/math.sqrt(N[i]*N[j])   
    return W   