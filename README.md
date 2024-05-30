# Towards Timing-Driven Routing: An Efficient Learning Based Geometric Approach

![image-20240517212548512](https://img2023.cnblogs.com/blog/2505287/202405/2505287-20240517212549832-241558537.png)

For more details, please refer to the [paper at ICCAD 2023](https://mrsun0.github.io/gwsun.github.io/files/Routing_Wirelength_Timing.pdf)
## Requirements

- Pytorch >= 1.9.0
## **Train**

The model was trained using two GPUs by default, utilizing the DDP module. To initiate the training process, you can use the command "torchrun".

```shell
sudo env PATH="$PATH" CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=10002 --nproc_per_node=2 train_lambda.py --degree 10 --batch_size 2048 --weight 0.5
```

Key words:

- CUDA_VISIBLE_DEVICES: the GPUs will be used.
- master_port: The port of torchrun,  you can change it at will.
- nproc_per_node: the number of GPUs used
- degree: degree of the sample you input.
- weight: the trade-off weight between WL and Radius(PL).

## **Test**

**The well-trained checkpoint are saved at 'save/beCalled/, and you can run the python file:' inference.py' to do a inference.'**

The default value for the 'transform' parameter is 1, but for improved results, you can set it to 8 with only a slight increase in processing time.

```shell
python inference.py
```

