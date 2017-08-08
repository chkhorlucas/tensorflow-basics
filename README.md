# Basics of Tensorflow (Part I)
## Official guide
* [Getting Started with Tensorflow](https://www.tensorflow.org/get_started/get_started)
## 0. Prerequisites  
* [Install Python](https://www.python.org/downloads/) (doesn't matter 2.7 or 3.n)  

* [Install Tensorflow](https://www.tensorflow.org/install/)   

## 1. Getting started
1. Open Python in terminal.
```
$ python
```

2. Import tensorflow.
```
>>> Import tensorflow as tf
```

3. The basic components of Tensorflow:  
* Graph - a collection of ops that may be executed together as a group.  
* Operations - nodes in Tensorflow graph.  

Run below commands and you will see currently there is nothing in the graph now.
```
>>> g = tf.get_default_graph()
>>> g.get_operations()
## []
```

4. Create a constant-type tensor, which value is immutable. It will also act as a node/operation in our graph.
```
>>> input = tf.constant(1.2)
```

5. Run below commands to see information of nodes/operations in the graph.
```
>>> o = g.get_operations()
>>> o
## [<tensorflow.python.framework.ops.Operation at 0x1185005d0>]
>>> o[0].node_def
## name: "Const"
## op: "Const"
## attr {
##   key: "dtype"
##   value {
## 	type: DT_FLOAT
##   }
## }
## attr {
##   key: "value"
##   value {
## 	tensor {
##   	dtype: DT_FLOAT
##   	tensor_shape {
##   	}
##   	float_val: 1.20000004768
## 	}
##   }
## }
```

6. Key in the constant name and it will show us the tensor information but not the value of it.
```
>>> input
## <tf.Tensor 'Const:0' shape=() dtype=float32>
```

7. To view the value, we have to evaluate ```input``` by creating a session to run on it.
```
>>> sess = tf.Session()
>>> sess.run(input)
## 1.2
```

8. Create a variable-type tensor.
```
>>> weight = tf.Variable(0.8)
```

9. Let's inspect the operations in our graph again. Notice that we have more that 2 operations!
```
>>> o = g.get_operations()
>>> for ol in o: print(ol.name)
## Const
## Variable/initial_value
## Variable
## Variable/Assign
## Variable/read
```

10. Let's perform some computation and inspect on the last operation.
```
>>> output = weight * input
>>> o = g.get_operations()[-1]
>>> o.name
## 'mul'
>>> for olist in o.inputs: print(olist)
## Tensor("Variable/read:0", shape=(), dtype=float32)
## Tensor("Const:0", shape=(), dtype=float32)
```

11. To get the result of the computation, we have to first 'initialize' the variable.
```
>>> init = tf.global_variables_initializer()
>>> sess.run(init)
```

12. Now we are ready to run the operation.
```
>>> sess.run(output)
## 0.80000001
```

13. We can use Tensorboard to generate the graph.
```
>>> summary_writer = tf.summary.FileWriter('log_simple_graph', sess.graph)
```

14. Open another new session of terminal and run below command.
```
$ tensorboard --logdir=log_simple_graph
```

15. Go to localhost:6006 to view our graph.

**Conclusion:** We have perform a basic math function using Tensorflow. Reason we use such a lengthy method to do a multiplication, is to understand the concepts of the Tensorflow framework before even hands on the neural networks.
