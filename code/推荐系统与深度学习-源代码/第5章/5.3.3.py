class Tree:
    def __init__(self):
        self.split_feature = None
        self.leftTree = None
        self.rightTree = None
        self.real_value_feature = True
        self.conditionValue = None
        self.leafNode = None
    def get_predict_value(self, instance):
        if self.leafNode:  
            return self.leafNode.get_predict_value()
        if not self.split_feature:
            raise ValueError("the tree is null")
        if self.real_value_feature and instance[self.split_feature] < self.conditionValue:
            return self.leftTree.get_predict_value(instance)
        elif not self.real_value_feature and instance[self.split_feature] == self.conditionValue:
            return self.leftTree.get_predict_value(instance)
        return self.rightTree.get_predict_value(instance)
    def describe(self, addtion_info=""):
        if not self.leftTree or not self.rightTree:
            return self.leafNode.describe()
        leftInfo = self.leftTree.describe()
        rightInfo = self.rightTree.describe()
        info = addtion_info+"{split_feature:"+str(self.split_feature)+",split_value:"
        +str(self.conditionValue +"[left_tree:"+leftInfo+",right_tree:"+rightInfo+"]}"
        return info
class LeafNode:
    def __init__(self, idset):
        self.idset = idset
        self.predictValue = None
    def describe(self):
        return "{LeafNode:"+str(self.predictValue)+"}"
    def get_idset(self):
        return self.idset
    def get_predict_value(self):
        return self.predictValue
    def update_predict_value(self, targets, loss):
        self.predictValue = loss.update_ternimal_regions(targets, self.idset)
def FriedmanMSE(left_values, right_values):
    weighted_n_left, weighted_n_right = len(left_values), len(right_values)
    total_meal_left, total_meal_right = sum(left_values)/float(weighted_n_left), sum(right_values)/float(weighted_n_right)
    diff = total_meal_left - total_meal_right
    return (weighted_n_left * weighted_n_right * diff * diff /
            (weighted_n_left + weighted_n_right))