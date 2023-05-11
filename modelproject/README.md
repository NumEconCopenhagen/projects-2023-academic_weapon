# Model analysis project

The model project **Dentist** consist of a dynamic optimization problem that models an agent's decision on whether or not to go to the dentist. The model is based on parameters such as level of usage of teeth, the fixed and the marginal cost of going to the dentist, a discount factor beta, and the benefits gained from the visit to the dentist. The optimization is done by brute force and it is not advised to try to run it for a life span over 15 years. To make the life span more realistic we think our agent to be a dog, taking the owner the decisions fully alturistically and feeling the ache from the teeth decay when the dog starts barking more.

We solve our model and show the best decisions of the agent and its tooth decay for different levels of beta. Also, two different extensions are added to our model. In the first one we wanted to make the teeth decay dependent on age so teeth would get worse over the life span. Secondly, we analyze the application of a subsidy on the fixed cost and on the marginal cost to check in which scenario would this subsidy be more efficient. 

The **results** of the project can be seen from running [Dentist.ipynb](Dentist.ipynb).

**Dependencies:** Apart from a standard Anaconda Python 3 installation, the project requires no further packages.
