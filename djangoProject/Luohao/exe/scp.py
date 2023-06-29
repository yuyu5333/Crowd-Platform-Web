import paramiko
from scp import SCPClient

def scp_send_files(host, port, username, password, local_path, remote_path):
    """
    使用SCP发送文件到远程服务器

    :param host: 远程服务器主机名或IP地址
    :param port: 远程服务器端口号，默认为22
    :param username: 登录远程服务器的用户名
    :param password: 登录远程服务器的密码
    :param local_path: 本地文件或文件夹路径
    :param remote_path: 远程服务器的目标路径
    """

    # 创建SSH客户端
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # 连接远程服务器
    ssh.connect(host, port, username, password)

    # 使用SCP发送文件或文件夹
    with SCPClient(ssh.get_transport()) as scp:
        scp.put(local_path, remote_path, recursive=True)

    # 关闭SSH连接
    ssh.close()

# if __name__ == "__main__":
#     # 将以下参数替换为实际值
#     host = "example.com"
#     port = 22
#     username = "your_username"
#     password = "your_password"
#     local_path = "/path/to/local/files"
#     remote_path = "/path/to/remote/destination"

#     scp_send_files(host, port, username, password, local_path, remote_path)
