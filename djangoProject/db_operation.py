import sqlite3
from sqlite3 import OperationalError



def create_table():

    connect = sqlite3.connect('./db.sqlite3')
    cur = connect.cursor()

    try: 
        sql = """
        CREATE TABLE hmt_sysdevicelatency (
            id integer primary key autoincrement,
            SysModelName varchar(100),
            Device varchar(100),
            Latency real,
            Energy real
        )
        ;"""
        cur.execute(sql)
        print("create table success")
        return True

    except OperationalError as o:
        print(str(o))

    except Exception as e:
        print(e)
        return False

    finally:
        cur.close()
        connect.close()

def add_table_col():
    connect = sqlite3.connect('./db.sqlite3')
    cur = connect.cursor()
    
    try:
    
        cur.execute("ALTER TABLE hmt_sysmodel ADD COLUMN Infomation_ope varchar(1000)")
        connect.commit()
        
        cur.close()
        # connect.close()
    
    
        print("update table col success!")
        return True
    
    except Exception as e:
        print(str(e))
    finally:
        cur.close()
        connect.close()

def insert_data_many():

    connect = sqlite3.connect('./db.sqlite3')
    cur = connect.cursor()

    try:
        inset_sql = """
            insert into hmt_imageclassification
            (id, ModelName, Flops, Params, Energy, Accuracy, MissionName2_id, CompressRate, Storage)
            values
            (
                ?,?,?,?,?,?,?,?,?
            );
        """
        datalist = [
            (
                13, "ResNet18",557.88,11.17,2789.4,76.07,1,0,42.8
            ),
            (
                14, "ResNet18-svd",409.75,1.9,2109.61,69.05,1,90.53,4.05
            ),
            (
                15,"ResNet18-dpconv",336.86,0.89,1777.37,73.04,1,91.93,3.45
            ),
            (
                16,"ResNet18-fire",413.94,1.94,2100.64,73.22,1,82.54,7.47
            ),
            (
                17,"ResNet18-inception1",372.53,1.54,1923.27,62.56,1,86.14,5.93
            ),
            (
                18,"ResNet18-inception2",541.87,3.19,2740.26,76.58,1,71.49,12.2
            )
        ]

        connect.executemany(inset_sql, datalist)

        connect.commit()

        print("insert success!")

        return True
    except Exception as e:
        print(str(e))
    finally:
        cur.close()
        connect.close()

def update_data():

    connect = sqlite3.connect('./db.sqlite3')

    try:
        update_sql = 'update hmt_imageclassification set Accuracy = ?,  Flops = ? where id = ? and  ModelName = ?;'
        datalist = [
            (
               409.75, 1.90, 4.05, 2109.61, 5, "ResNet18-svd"
            ),
        ]

        connect.executemany(update_sql, datalist)

        connect.commit()

        print("update success!")

        return True
    except Exception as e:
        print(str(e))
    finally:
        connect.close()

def insert_data():

    connect = sqlite3.connect('./db.sqlite3')
    cur = connect.cursor()

    try:
        inset_sql = """
            insert into hmt_sysmodel
            (id, ModelName, Flops, Params, Energy, Accuracy, MissionName2_id, CompressRate, Storage)
            values
            (
                13, "resnet18-cifar100", 541.87, 3.19, 12.25, -1, 2740.26, -1,
                    "ResNet又称残差网络是由来自Microsoft Research的4位学者提出的卷积神经网络, \
                    在2015年的ImageNet大规模视觉识别竞赛（ImageNet Large Scale Visual Recognition Challenge, ILSVRC）中获得了图像分类和物体识别的优胜. \
                    残差网络的特点是容易优化, 并且能够通过增加相当的深度来提高准确率. 其内部的残差块使用了跳跃连接, 缓解了在深度神经网络中增加深度带来的梯度消失问题.", 
                
                    "Inception v2中最重要的改进是使用了Batch Normalization技术，它可以在神经网络中对每一层输入进行标准化，从而加速训练过程和提高模型的泛化能力。\
                    此外，Inception v2还使用了更小的卷积核和更多的分支结构，以进一步提高模型的表达能力。\
                    Inception2模块是Inception v2中的一个重要组件，它在Inception1模块的基础上进行了改进和优化。Inception2模块主要包括四个分支，分别是1x1卷积分支、3x3卷积分支、5x5卷积分支和3x3 max pooling分支。\
                    其中1x1卷积分支和3x3卷积分支主要用于提取特征，5x5卷积分支和3x3 max pooling分支主要用于提取多尺度的特征。四个分支的输出在通道维度上进行拼接，得到Inception2模块的输出特征图。
                    Inception v2相对于Inception v1来说，主要改进在于加入了Batch Normalization技术，这可以使模型更快地收敛和更好地泛化。此外，Inception v2还使用了更小的卷积核和更多的分支结构，\
                    以进一步提高模型的表达能力。Inception v2在2015年的ImageNet图像分类挑战中取得了非常好的成绩，成为了卷积神经网络领域的经典算法之一。"
            );
        """
        cur.execute(inset_sql)
        
        connect.commit()

        print("insert success!")

        return True
    except Exception as e:
        print(str(e))
    finally:
        cur.close()
        connect.close()

def delete_table():

    connect = sqlite3.connect('./db.sqlite3')
    cur = connect.cursor()

    try:
        del_sql = """drop table hmt_sysmodel_latency;"""


        cur.execute(del_sql)
        print("delete table success")
        return True

    except OperationalError as o:
        print(str(o))

    except Exception as e:
        print(e)
        return False

    finally:
        cur.close()
        connect.close()

if __name__ == "__main__":
    
    
    # delete_table()
    
    # add_table_col()

    # create_table()

    # insert_data()

    insert_data_many()

    # update_data()

'''
    inset_sql = """
            insert into hmt_sysmodel
            (id, SysModelName, Computation, Parameter, Storage, Latency, Energy, Accuracy, Infomation, Infomation_ope)
            values
            (
                9, "ResNet18-inception2", 541.87, 3.19, 12.25, -1, 2740.26, -1,
                    "ResNet又称残差网络是由来自Microsoft Research的4位学者提出的卷积神经网络, \
                    在2015年的ImageNet大规模视觉识别竞赛（ImageNet Large Scale Visual Recognition Challenge, ILSVRC）中获得了图像分类和物体识别的优胜. \
                    残差网络的特点是容易优化, 并且能够通过增加相当的深度来提高准确率. 其内部的残差块使用了跳跃连接, 缓解了在深度神经网络中增加深度带来的梯度消失问题.", 
                
                    "Inception v2中最重要的改进是使用了Batch Normalization技术，它可以在神经网络中对每一层输入进行标准化，从而加速训练过程和提高模型的泛化能力。\
                    此外，Inception v2还使用了更小的卷积核和更多的分支结构，以进一步提高模型的表达能力。\
                    Inception2模块是Inception v2中的一个重要组件，它在Inception1模块的基础上进行了改进和优化。Inception2模块主要包括四个分支，分别是1x1卷积分支、3x3卷积分支、5x5卷积分支和3x3 max pooling分支。\
                    其中1x1卷积分支和3x3卷积分支主要用于提取特征，5x5卷积分支和3x3 max pooling分支主要用于提取多尺度的特征。四个分支的输出在通道维度上进行拼接，得到Inception2模块的输出特征图。
                    Inception v2相对于Inception v1来说，主要改进在于加入了Batch Normalization技术，这可以使模型更快地收敛和更好地泛化。此外，Inception v2还使用了更小的卷积核和更多的分支结构，\
                    以进一步提高模型的表达能力。Inception v2在2015年的ImageNet图像分类挑战中取得了非常好的成绩，成为了卷积神经网络领域的经典算法之一。"
            );
        """
'''