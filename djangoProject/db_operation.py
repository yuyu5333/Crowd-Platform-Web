import sqlite3
from sqlite3 import OperationalError



def create_table():

    connect = sqlite3.connect('./db.sqlite3')
    cur = connect.cursor()

    try: 
        sql = """
        CREATE TABLE hmt_imagesclassification (
            id integer primary key autoincrement,
            DatasetName varchar(100),
            ModelName varchar(100),
            Computation real,
            Parameter real,
            Energy,
            Accuracy,
            CompressRate,
            Storage
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

def copy_table():
    
    connect = sqlite3.connect('./db.sqlite3')
    cur = connect.cursor()
    
    try: 
        sql = """
        INSERT INTO hmt_imagesclassification 
        (id, DatasetName, ModelName, Computation, Parameter, Energy, Storage, Accuracy, CompressRate) 
        SELECT id, Dataset, ModelName, Flops, Params, Energy, Storage, Accuracy, CompressRate  
        FROM hmt_imageclassification;"""
        cur.execute(sql)
        connect.commit()
        print("copy table success")
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
    
        cur.execute("ALTER TABLE hmt_imagesclassification ADD COLUMN Latency real")
        connect.commit()
        
        cur.close()
        # connect.close()
    
    
        print("add table col success!")
        return True
    
    except Exception as e:
        print(str(e))
    finally:
        cur.close()
        connect.close()

def delete_table_col():
    connect = sqlite3.connect()

def insert_data_many_hmt_imagesclassification():

    connect = sqlite3.connect('./db.sqlite3')
    cur = connect.cursor()

    try:
        inset_sql = """
            insert into hmt_imagesclassification
            (id, DatasetName, ModelName, Computation, Parameter, Energy, Accuracy, CompressRate, Storage, Latency)
            values
            (
                ?,?,?,?,?,?,?,?,?,?
            );
        """
        datalist = [
            # 图像分类 语义分割 目标检测 行为识别 动作检测 语音文本
            (
                37, "Food-101", "Vgg16-dpconv-inception2", 167460, 5.23, 837396.14, 53.33, 80.19, 20.11, 5207456
            ),
            # (
            #     35, "Food-101", "ResNet50-inception2", 173210, 7.07, 866080.95, 72.27, 73.28, 27.13, 7041440
            # ),
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

def insert_data_many_hmt_classdatasetmodel():
    
    connect = sqlite3.connect('./db.sqlite3')
    cur = connect.cursor()

    try:
        inset_sql = """
            insert into hmt_classdatasetmodel
            (id, ClassName, DatasetName, ModelName)
            values
            (
                ?,?,?,?
            );
        """
        datalist = [
            # 图像分类 语义分割 目标检测 行为识别 动作检测 语音文本
            (
                25, "图像分类", "Food-101", "ResNet50"
            ),
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
        update_sql = 'update hmt_imagesclassification set Latency = ? where id = ?;'
        datalist = [
            # 图像分类 语义分割 目标检测 行为识别 动作检测 语音文本
            (
                123378432, 27
            ),
            (
                121850560, 28
            ),
            (
                125028032, 29
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
            (id, DatasetName, Flops, Params, Energy, Accuracy, MissionName2_id, CompressRate, Storage)
            values
            (
                13, "resnet18-Cifar100", 541.87, 3.19, 12.25, -1, 2740.26, -1,
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

def update_table_name():
    
    connect = sqlite3.connect('./db.sqlite3')
    cur = connect.cursor()

    try:
        update_sql = """ALTER TABLE A RENAME TO hmt_classdatasetmodel ;"""


        cur.execute(update_sql)
        print("update table success")
        return True

    except OperationalError as o:
        print(str(o))

    except Exception as e:
        print(e)
        return False

    finally:
        cur.close()
        connect.close()

def test_db():
    connect = sqlite3.connect('./db.sqlite3')
    cur = connect.cursor()
    
    try:
        select_sql = """select * from hmt_imagesclassification where DatasetName = 'Cifar100' and ModelName = 'ResNet18'"""
        
        

        cur.execute(select_sql)
        
        for row in cur:
            print(row)
        
        print("test table success")
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
    
    # test_db()
    
    # delete_table()
    
    # add_table_col()

    # create_table()
    
    # copy_table()

    # insert_data()

    # insert_data_many_hmt_imagesclassification()
    
    # insert_data_many_hmt_classdatasetmodel()

    update_data()
    
    # update_table_name()

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