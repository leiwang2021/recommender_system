def CosSimilarCompute(U_k, W_t):
    if( len(U_k) != len(W_t) ):
        assert("user vector size not equals item vector ")
    if(len(U_k)==0):
        assert("user vector size  equal to 0")
    i = 0
    Score1 = 0
    Score2 = 0
    Score3 = 0
    while( i < len(U_k) ):
        Score1 = Score1 + U_k[i]*W_t[i]
        Score2 = Score2 + U_k[i]*U_k[i]
        Score3 = Score3 + W_t[i]*W_t[i]
        i = i+1
    if(Score3==0 or Score2==0):
        assert("user or item vector equal to 0")
    return Score1*1.0/(math.sqrt(Score2)*math.sqrt(Score3))



def cosSim(U_k, W_t):
    num = float(U_k.T*W_t)
    denom = linalg.norm(U_k)*linalg.norm(W_t)
    return 0.5+0.5*(num/denom)


