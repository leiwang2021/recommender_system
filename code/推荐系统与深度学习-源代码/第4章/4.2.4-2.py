import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS

case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)

##读取数据集，使用,分隔，分别为用户id，物品id，评分，次数
def parseRating(str: String): Rating = {
  val fields = str.split(",")
  assert(fields.size == 4)
  Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
}

val ratings = spark.read.textFile(".../als/movielens_ratings.txt")
  .map(parseRating)
  .toDF()

##拆分训练集和测试集
val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

##rating：由用户-物品矩阵构成的训练集
##rank：隐藏因子的个数
##numIterations: 迭代次数
##lambda：正则项的惩罚系数
##alpha： 置信参数
val als = new ALS()
  .setRank(100)
  .setMaxIter(50)
  .setRegParam(0.01)
  .setUserCol("userId")
  .setItemCol("movieId")
  .setRatingCol("rating")
val model = als.train(training)


##rating：由用户-物品矩阵构成的训练集
##rank：隐藏因子的个数
##numIterations: 迭代次数
##lambda：正则项的惩罚系数
##alpha： 置信参数
val als = new ALS()
  .setRank(100)
  .setMaxIter(50)
  .setRegParam(0.01)
  .setUserCol("userId")
  .setItemCol("movieId")
  .setRatingCol("rating")
val model2 = als.trainImplicit(training)

##在测试集上进行预测
val predictions = model.predict(test)

##获得物品的特征
val item_feature = model.productFeatures

##获得用户的特征
val user_feature = model.userFeatures