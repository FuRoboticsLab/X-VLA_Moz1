
```
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```


### Server


1. Install Conda and Install the Env
```
conda create -n XVLA python=3.10 -y
conda activate XVLA
git clone git@github.com:2toinf/X-VLA.git
cd X-VLA
pip install -r requirements.txt
```



### Client
```
curl -sSL http://10.42.0.101:8849/install.sh | bash
cd a2d_sdk
git clone git@github.com:2toinf/X-VLA.git
cd X-VLA
pip install -r eval/agibot/client_requirements.txt
source eval/agibot/robo-service.sh
```



