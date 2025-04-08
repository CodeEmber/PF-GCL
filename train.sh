#!/bin/bash
###
 # @Author       : wyx-hhhh
 # @Date         : 2024-11-25
 # @LastEditTime : 2024-11-30
 # @Description  : 
### 

# 定义模型训练命令和参数
#train_command 可以替换成你的命令
# train_command="python pytorch_main_tmall.py"
train_command="python /home/wyx/PF-GCL++/run_recbole_gnn.py --model=PFGCL --dataset=yelp"

# nohup python /home/wyx/PF-GCL++/run_recbole_gnn.py --model=PFGCL --dataset=amazon-books --lmbd_ssl=0.005 --temperature=0.05 > /dev/null 2>&1 &

# 定义 lmbd_ssl 数组
lmbd_ssl=(0.5 0.7 1.0)
# lmbd_ssl=(0.005 0.01 0.05 0.1 0.2 0.5 1.0)


# 循环遍历 lmbd_ssl 数组
for ssl in "${lmbd_ssl[@]}"; do
    echo "Training model with lmbd_ssl: $ssl"
    # 执行训练命令
    $train_command --lmbd_ssl=0.005 --temperature=$ssl
    # $train_command --lmbd_ssl=$ssl
done
   