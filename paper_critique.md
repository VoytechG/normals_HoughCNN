1. It is unlcear how data is generated. Figure shown is not a clear visualisation of the generated data. 


2. 
 Quote: 
 We then rely on the ability of the network to generalize andtreat partial data, as is the case for occlusion in object detection.It allows the network to treat more complex situations than just 3-side corners. Regression for noisy data, which is the general case,is learned from points far enough from the edges and the corner.

 It is unclear how this is leveraged. Generated data is only a corner, how does it genereslise to more difficult situations? There are 3 planes, so 3 normals, which form 3 edge and one corner. How can they generalise? The model is not taught how to behave on various surfaces. Besides, the pointcloud density it quite large. 