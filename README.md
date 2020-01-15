# qingdao

## 部署


### 系统环境
下述的环境需要先手动在部署机器上进行安装。

 - python 3.6.9
 https://www.python.org/downloads/release/python-369/

 - pip 19.3.1
 https://files.pythonhosted.org/packages/ce/ea/9b445176a65ae4ba22dce1d93e4b5fe182f953df71a145f557cffaffc1bf/pip-19.3.1.tar.gz

### 项目环境
在项目中提供了 makefile 文件来进行项目的部署，makefile 中编写了一些规则来执行项目的部署，其中包含 package, install, run, stop, clean 五条规则。以下是使用 makefile 文件进行项目部署的步骤。

**在能够访问到公网的环境下，克隆项目**

``
git clone https://github.com/Ferrair/qingdao.git
``

**在能够访问到公网的环境下，打包项目**

``
cd $project
``
``
make package
``

package 规则采用 pip 命令在在线的环境中下载项目依赖的离线安装包，并将整个项目打包成 zip 压缩包。在完成 package 命令后，在当前目录下会有一个hongsi_deploy.zip 的压缩包，其中包含打包好的所有项目的代码和依赖安装包

上述步骤已完成，链接: https://pan.baidu.com/s/1wwCT7SOpUznNU8rPQ6b97Q  密码:s4dw

**将压缩包拷贝至部署环境并解压**

``
unzip hongsi_deploy.zip -d hongsi_deploy
``

**安装项目**

``
make install
``

install 规则采用 pip 命令安装依赖安装包，并依据当前环境修改项目中的配置信息

**项目运行**

``
make run
``

run 规则以后台运行的方式启动项目，并将项目的日志输出重定向到项目根目录下 model_service.log 文件中，并将项目进程的 pid 保存至 pid.txt 中

**检查项目是否运行**

``
make stat
``


同时在model_service.log看到如下消息

```
 * Serving Flask app "/home/qihang/hongsi_deploy/test/src/model_controller.py"
 * Environment: production
   WARNING: Do not use the development server in a production environment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```


请求`http://127.0.0.1:5000/api/test`可以返回`Ok`


**停止项目运行**

``
make stop
``

**清理项目**

``
make clean
``