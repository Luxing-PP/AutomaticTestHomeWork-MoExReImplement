DATA=cifar100 #cifar10
JOB=pyramidnet_moex
epoch=5
lam=0.5
prob=0.25

python MoExOnCiFAR100.py \
    --dataset ${DATA} \
    --batch_size 5 \
    --lr 0.25 \
    --epochs ${epoch} \
    --beta 1.0 \
    --lam ${lam} \
    --moex_prob ${prob}
