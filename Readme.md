# MATH8013 Final Project

### RUN
- baseline model:
~~~bash
python main_baseline.py \
 --batch-size 128 \
 --gpu 1 \
 -lr 0.01 \
 --epochs 200 \
 --network vgg16_bn
 
#**************************************************
#        *Optimal epoch: 175
#        *Optimal test accuracy: 0.93
#**************************************************
~~~
### Visualization
there may be some bug
- show network:
~~~bash
python visualization.py --load_checkpoint "Outputs/Exp3/vgg14_bn_k5" --show_network True --network vgg14_bn_k5
~~~
- visualize layer:
~~~bash
python visualization.py --load_checkpoint "Outputs/Exp3/vgg15_bn_k5" --visualized_layer "features.0" --network vgg15_bn_k5
~~~
