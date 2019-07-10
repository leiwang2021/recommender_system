def svdEst(userData,xformedItems, user, simMeas, item) :
    n = shape(xformedItems)[0]
    simTotal = 0.0; ratSimTotal = 0.0
    # 对于给定的用户，for循环所有物品，计算与item的相似度
    for j in range(n) :
        userRating = userData[:,j]
        if userRating == 0 or j == item : continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        # print便于了解相似度计算的进展情况
        print ('the %d and %d similarity is : %f' % (item, j, similarity))
        # 对相似度求和
        simTotal += similarity
        # 对相似度及评分值的乘积求和
        ratSimTotal += similarity * userRating
    if simTotal == 0 : return 0
    else : return ratSimTotal/simTotal
    
    
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=svdEst):
    U,Sigma,VT = linalg.svd(dataMat)
    # 使用奇异值构建一个对角矩阵
    Sig4 = mat(eye(4)*Sigma[:4])
    # 利用U矩阵将物品转换到低维空间中
    xformedItems = dataMat.T * U[:, :4] * Sig4.I
    print('xformedItems=',xformedItems)
    print('xformedItems行和列数', shape(xformedItems))
    
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    print('dataMat[user, :].A=',dataMat[user, :].A)
    print('nonzero(dataMat[user, :].A == 0)结果为', nonzero(dataMat[user, :].A == 0))
    # 如果不存在未评分物品，退出函数，否则在所有未评分物品上进行循环
    if len(unratedItems) == 0: return ('you rated everything')
    itemScores = []
    for item in unratedItems:
        print('item=',item)
   # 对于每个未评分物品，通过调用standEst()来产生该物品基于相似度的预测评分。
        estimatedScore = estMethod(dataMat[user, :],xformedItems, user, simMeas, item)
        # 该物品的编号和估计得分值会放在一个元素列表itemScores
        itemScores.append((item, estimatedScore))
    # 寻找前N个未评级物品
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]
                  
                  
myMat=mat(loadExData())
result=recommend(myMat, 1, estMethod=svdEst)
#print(result)