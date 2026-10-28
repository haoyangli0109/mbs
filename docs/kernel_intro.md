# 1. GPU的访存模式

HBM每128B为一个cache line. MMU(Memory Management Unit)先将对应的cache line放到L2 cache，然后再从L2 cache读到对应的lane. 比如:
```c++
offset = threadIdx.x * 4;  // 每个lane读取4B数据，共256B，对应2个cache line
```
上述读取模式对应的步骤为，MMU将2个cache line读取到L2 Cache，然后再将数据分配到不同的不同的lane

一个读取效率很低的例子：
```c++
// 仍然假设每个lane读取4B数据，为了读取真正有用的256B数据，MMU需要搬运64个cache line
offset = threadIdx.x * 4096;
```

上述例子中，如果64个cache line后续的数据会在后面被用到，那么提前把这些数据搬运到L2 cache是不是也不会影响后续的读取效率呢？

如果L2 Cache的size是无限大的，那么上述说法有道理。但事实上，L2 Cache的size是很小的，MI355X只有32 MB的L2 cache，而且真正运行的时候会有上百个甚至上千个wave同时在工作，它们都在争抢着32MB的L2 Cache，某个cache line还没有被用完，就可能被换出，造成浪费。

**访问HBM效率最高的方式是大位宽的连续访存。(AMD GPU最高支持的位宽是dwordx4(16B))**

# 2. 16x16x128 MFMA的layout

<img src="./images/16x16x128 layout.png" />
注：图中的每个数字代表的是lane id

从上图可以看出，MFMA的layout并不是读取最优的layout. 如果我们就按照图中所示的layout读取数据会有两个问题: (1) lane的读取不连续，(2) cache line的效率只有50%

# 3. 对B的读取

B对应于weights，由于weights是静态的，可以在inference之前进行任意操作，所以我们可以对B进行对读取有利preshuffle。MFMA的每次计算都会对应一个 16x64B的block，所以我们可以按照每16x64B block连续排布的方式进行preshuffle，如图所示:

<img src="./images/B layout.png" />

# 4. 对A的读取

A对应于activate，是在inference过程中进行dynamic quant的，所以进行overhead很大的shuffle操作就得不偿失了。此时可以利用LDS来完成行优先layout和列优先layout的转换。为了说明这个问题，我们用一个极简的样例进行演示。假设LDS的size为8 rows x 8 banks. 每个lane读取的数据为一个bank的位宽，如下图所示：

<img src="./images/LDS Swizzle.png" />

回到我们的kernel implementation

<img src="./images/A layout.png" />

# 5. Scale的读取

在MFMA指令中 scale的layout跟A和B一致，只是位宽只有一个Byte。在AMD GPU中，所有的寄存器都是4B的位宽，如果每次读一个Byte的话，效率极低。所以对scale也会进行shuffle，如下图所示:

<img src="./images/scale layout.png" />
