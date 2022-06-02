# CIKM_AdaGCL
## Experiments

##### Requirements
* ogb>=1.3.3
* torch>=1.10.0
* torch-geometric>=2.0.4

##### Training
GraphSAINT
``
python saint_graph.py --epochs <epochs> --load_CL <load_CL> --par <par> --rate <rate> -topk <topk>
``
where `` <par> `` is a contrastive loss ratio. `` <rate> `` is the perturbation ratio of data augmentation. 
`` <topk> `` is the number of subgraphs involved in contrastive learning. `` <load_CL> `` is to add contrastive learning at the Nth epoch, Default is 0.

Cluster-GCN

``
python cluster_graph.py --epochs <epochs> --load_CL <load_CL> --par <par> --rate <rate>
``

GraphSAGE
``
python cluster_graph.py --epochs <epochs> --load_CL <load_CL> --par <par> --rate <rate>
``
