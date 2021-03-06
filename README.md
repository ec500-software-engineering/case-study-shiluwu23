# Case-study-shiluwu23
* Case Study Project: Tensorflow

BU Google Drive link:
https://docs.google.com/document/d/1aEkK65U4RGKOq8chAGzuaRif82KWp6MWsPZ8_v-o674/edit?usp=sharing
## Brief introduction
TensorFlow is an open-source machine learning library for numerical computation and large-scale machine learning. It uses Python to provide a convenient front-end API for building applications with the framework, while executing those applications in high-performance C++.

## 1.	Technology and Platform
* Language used: C++/Python

TensorFlow is designed to support multiple client languages. Currently, Python is chosen as the first well-supported language for expressing and controlling the training of models and python supports the most features. However, the core of TensorFlow is written in a combination of highly-optimized C++ and CUDA (Nvidia's language for programming GPUs).

If the project was started today, I would still use python and C++ as the coding languages. Since C++ offers speed and performance, and python is easy to integrate and it’s helpful for building Machine Learning Applications from its inception to deployment.

*	Environment needed: Anaconda.
*	frameworks / libraries: TensorFlow, Pandas, TensorFlow-gpu, etc.
## 2.	Testing
One of the easiest ways to get started using TensorFlow Serving is with Docker. With Docker, we can manage our infrastructure in the same ways we manage your applications. Docker provides tooling and a platform to manage the lifecycle of our containers: Develop the application and its supporting components using containers. The container becomes the unit for distributing and testing our application.
### CI platform
Since TensorFlow is a cross-platform tool that can run on CPU and GPU hardware, the maintainers have set up multiple pipelines to build and test the tool on different operating system platforms and CPU/GPU configurations. Their pipelines are built using Jenkins.

MacOS, Linux,, Windows, Raspberry Pi and Android are tested on their CI.

## 3. Software architecture
* General architecture

The user level code is separated by a C API into different languages from the core runtime.

![](https://github.com/ec500-software-engineering/case-study-shiluwu23/blob/master/case%20study%20images/General%20architecture.png)


* **Client**

**Client** Defines the computation as a dataflow graph. The client creates a session, which sends the graph definition to the distributed master as a tf.GraphDef protocol buffer. When the client evaluates a node or nodes in the graph, the evaluation triggers a call to the distributed master to initiate computation.

The client initiates graph execution using a session. It has built a graph that applies weights (w) to a feature vector (x), adds a bias term (b) and saves the result in a variable (s). "/job:worker/task:0" and "/job:ps/task:0" are both tasks with worker services. "PS" stands for "parameter server": a task responsible for storing and updating the model's parameters. Other tasks send updates to these parameters as they work on optimizing the parameters. 
![](https://github.com/ec500-software-engineering/case-study-shiluwu23/blob/master/case%20study%20images/Client.png)

* **Distributed Master**

**The distributed master** prunes the graph to obtain the subgraph required to evaluate the nodes requested by the client, partitions the graph to obtain graph pieces for each participating device, and caches these pieces so that they may be re-used in subsequent steps. Since the master sees the overall computation for a step, it applies standard optimizations such as common subexpression elimination and constant folding. It then coordinates execution of the optimized subgraphs across a set of tasks.

![](https://github.com/ec500-software-engineering/case-study-shiluwu23/blob/master/case%20study%20images/Distributed%20Master.png)

* **Worker Services**

**The worker service** in each task handles requests from the master, schedules Schedule the execution of graph operations using kernel implementations appropriate to the available hardware (CPUs, GPUs, etc)., and mediates direct communication between other worker services.

![](https://github.com/ec500-software-engineering/case-study-shiluwu23/blob/master/case%20study%20images/Worker%20Services.png)

* **Kernel Implementations**

It perform the computation for individual graph operations. If it is difficult or inefficient to represent a subcomputation as a composition of operations, users can register additional kernels that provide an efficient implementation written in C++. For example, we recommend registering your own fused kernels for some performance critical operations, such as the ReLU and Sigmoid activation functions and their corresponding gradients. The XLA Compiler has an experimental implementation of automatic kernel fusion.


## 4. Defects

**No GPU support other than Nvidia and only language support**

Currently, the only supported GPUs are that of NVIDIA and the only full language support is of Python which makes it a disadvantage as there is a rise of other languages in deep learning.

**Missing Symbolic Loops**

When it comes to variable length sequences, the most required feature are the symbolic loops. However, TensorFlow does not offer this feature, but there is a workaround using finite unfolding (bucketing).

**Computation Speed**

TensorFlow lacks behind in computational speed when compared to its competitors such as Torch7, CNTK and Caffe.


## 5. Demonstration
Run the *train.py.*  This file demonstrates training a simple Convolutional Neural Network (CNN) to classify cats and dogs. The accuracy has reached 96.875%.

* Install TensorFlow  
> Python 2
➜ pip install tensorflow

> Python 3
➜ pip3 install tensorflow 

* Import TensorFlow
```import tensorflow as tf```
* Download and prepare the dataset 

Open the training dataset folder, there are 1000 pictures in all, half are dogs and half are cats.
```
ShiludeMacBook-Pro:catdog shiluwu$ cd train
ShiludeMacBook-Pro:train shiluwu$ ls
cats	dogs
ShiludeMacBook-Pro:train shiluwu$ for dir in `find ./ -maxdepth 1 -type d`;do echo -n -e "$dir\t";find $dir -type f|wc -l ;done;
./	    1000
.//dogs	     499
.//cats	     500
```
* Create the convolutional base 
* Compile and train the model 
* Evaluate the model
```
def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss, i):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    print("epoch:", str(epoch + 1) + ",i:", str(i) +
          ",accuracy:", str(acc) + ",validation_acc:", str(val_acc) + ",val_loss:", str(val_loss))
```
![](https://github.com/ec500-software-engineering/case-study-shiluwu23/blob/master/case%20study%20images/accuracy.jpg)
## Reference
https://www.tensorflow.org/tutorials/

https://stackoverflow.com/questions/35677724/tensorflow-why-was-python-the-chosen-language

https://www.infoworld.com/article/3278008/what-is-tensorflow-the-machine-learning-library-explained.html

https://data-flair.training/blogs/tensorflow-pros-and-cons/
