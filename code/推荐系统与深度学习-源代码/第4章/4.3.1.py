import org.apache.spark.graphx.{GraphLoader, VertexRDD}                     
import org.apache.spark.{SparkConf, SparkContext}                           
                                                                            
object GraphRale {                                                          
  /**                                                                       
    * 数据列表的笛卡尔乘积：{1,2,3,4}=>{(1,2),(1,3),(1,4),(2,3),(2,4),(3,4）}           
    * @param input                                                          
    * @return                                                               
    */                                                                      
  def ciculate(input:List[Long]):Set[String]={                              
    var result = Set[String]()                                              
    input.foreach(x=>{                                                      
      input.foreach(y=>{                                                    
        if(x<y){                                                            
          result += s"${x}|${y}"                                            
        }else if(x>y){                                                      
          result += s"${y}|${x}"                                            
        }                                                                   
      })                                                                    
    })                                                                      
    return result;                                                          
  }                                                                         
  def twoDegree()={                                                         
    val conf = new SparkConf().setMaster("common friends").setAppName("graph")       
    val sc = new SparkContext(conf)                                
    ##输入数据为好友关系对，每一行为2个id
    val graph = GraphLoader.edgeListFile(sc,"./newtwork/grap.txt")   
    ##将形如(a,b),(a,c),(a,f)的关系对转化为a->list(b,c,f) 
    val relate: VertexRDD[List[Long]] = graph.aggregateMessages[List[Long]](
      triplet=>{                                                            
        triplet.sendToDst(List(triplet.srcId))                              
      },                                                                    
      (a,b)=>(a++b)                                                         
    ).filter(x=>x._2.length>1)                                              
    ##a->list(b,c,f)中bcd用户均有共同好友a                                                                        
    val re = relate.flatMap(x=>{                                            
      for{temp <- ciculate(x._2)}yield (temp,1)                             
    }).reduceByKey(_+_)                                                     
                                                                            
    re.foreach(println(_))                                                  
  }                                                                         
  def main(args: Array[String]): Unit = {                                   
    twoDegree()                                                             
  }                                                                         
}       