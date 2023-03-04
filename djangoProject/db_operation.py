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

def insert_data_many():

    connect = sqlite3.connect('./db.sqlite3')
    cur = connect.cursor()

    try:
        inset_sql = """
            insert into hmt_sysdevicelatency 
            (id, SysModelName, Device)
            values
            (
                ?,?,?
            );
        """
        datalist = [
            (
                1, "AlexNet", "Windows"
            ),
            (
                2, "AlexNet", "RaspberryPi4B"
            ),
            (
                3, "AlexNet", "JetsonNx"
            ),
            (
                4, "MobileNet", "Windows"
            ),
            (
                5, "MobileNet", "RaspberryPi4B"
            ),
            (
                6, "MobileNet", "JetsonNx"
            ),
            (
                7, "ResNet", "Windows"
            ),
            (
                8, "ResNet", "RaspberryPi4B"
            ),
            (
                9, "ResNet", "JetsonNx"
            ),
            (
                10, "VGG", "Windows"
            ),
            (
                11, "VGG", "RaspberryPi4B"
            ),
            (
                12, "VGG", "JetsonNx"
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
        update_sql = 'update hmt_sysmodel set Energy = ? where id = ? and SysModelName = ?;'
        datalist = [
            (
                260.66, 1, "AlexNet"
            ),
            (
                1671.23, 2, "MobileNet"
            ),
            (
                747.76, 3, "ResNet"
            ),
            (
                232.74, 4, "VGG"
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
            (id, SysModelName, Computation, Parameter, Storage, Latency, Energy, Accuracy, Infomation)
            values
            (
                1, "AlexNet", 738.68, 14.22, 54.26, -1, -1, -1, "Alexnet模型为8层深度网络, 由5个卷积层和3个全连接层构成, 不计LRN层和池化层. \
                    AlexNet中包含了几个比较新的技术点, 也首次在CNN中成功应用了ReLU、Dropout和LRN等Trick. 同时AlexNet也使用了GPU进行运算加速. \
                    AlexNet是2012年ImageNet竞赛冠军获得者Hinton和他的学生Alex Krizhevsky设计的, 该模型以很大\
                    的优势获得了2012年ISLVRC竞赛的冠军网络, 分类准确率由传统的 70%+提升到 80%+, 自那年之后, 深度学习开始迅速发展."
            );
        """
        cur.execute(inset_sql)

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

    # create_table()

    # insert_data()

    # insert_data_many()

    update_data()
