### Curvature-aware Graph Attention for PDEs on Manifolds
---
Deep models have recently achieved remarkable performances in solving partial differential equtions (PDEs). The previous methods are mostly focused on PDEs arising in Euclidean spaces with less emphasis on the general manifolds with rich geometry. Several proposals attempt to account for the geometry by exploiting the spatial coordinates but overlook the underlying intrinsic geometry of manifolds. In this paper, we propose a Curvature-aware Graph Attention for PDEs on manifolds by exploring the important intrinsic geometric quantities such as curvature and discrete gradient operator. It is realized via parallel transport and tensor field on manifolds. To accelerate computation, we present three curvatureoriented graph embedding approaches and derive closed-form parallel transport equations, and a sub-tree partition method is also developed to promote parameter-sharing. Our proposed curvature-aware attention can be used as a replacement for vanilla attention, and experiments show that it significantly improves the performance of the existing methods for solving PDEs on manifolds.

[Curvature-aware Graph Attention for PDEs on Manifold[paper]](https://openreview.net/forum?id=vWYLQ0VPJx&noteId=vWYLQ0VPJx)

### Requirements
---
The code doesn't need any extra dependency except for popular libraries like pytorch,numpy and some python standard libraries.

### Instructions
---
The main body of our model lies in `CURVGT.py` while some auxilary classes reside in `graphtransformers.py` and `ResNet.py`. Geometry processed documents like normal vectors, Gaussian curvature and parameters to support parallel transport are all in the folder `geometry_processed_docs`. Stuffs to support sub-tree partition mechanism are in the folder `sub_tree_partitions`. The train set and test set are in the folder `wrinkle`. The weights of the best-performing model are saved in the `best_model` directory.

### Quick Start
---
A trained model is provided. You can load the parameters and run the test function by simply typing in the command:

`python CURVGT.py --model_name CURVGT --datasets_function datasets_wave --datasets wrinkles --batch_size 20 --gpu 0 --test_pattern 0`

You can also choose to train the model by:

`python CURVGT.py --model_name CURVGT --datasets_function datasets_wave --datasets wrinkles --batch_size 20 --gpu 0 --test_pattern 1`

The following parameters can be used when running the script:

- `--model_name`: 
  - **Description**: Specifies the name of the model architecture to use.
  - **Example**: `CURVGT` 
  - **Required**: Yes

- `--datasets_function`: 
  - **Description**: Dataset loading function to use. This determines the type of PDE dataset.
  - **Options**: 
    - `datasets_wave`: Wave equation dataset
    - `datasets_isotropic-heat-equation`: Isotropic heat equation dataset
  - **Required**: Yes

- `--datasets`: 
  - **Description**: Name of the specific dataset within the chosen dataset function.
  - **Example**: `wrinkles` 
  - **Required**: Yes

- `--batch_size`: 
  - **Description**: Number of samples processed at once during training or testing.
  - **Default**: 20.

- `--gpu`: 
  - **Description**: Index of the GPU to use for computation.
  - **Default**: 0

- `--test_pattern`: 
  - **Description**: Configuration for the testing pattern or protocol.
  - **Options**: 
    - `0`: Standard testing pattern
    - `1`: Standard training pattern
  - **Default**: 0

### Further Illustration of Geometry Processed Documents
---
Documents unmentioned here are not lerveraged in our implementation. 
- `GaussianCurvature.pt`: Each node is assigned with a scalar, standing for the Gaussian curvature estimate down there. We leverage a mask (`option_mask.pt`) to decide the correct parallel transport (Euclidean, sphereic and hyperbolic). Thus three types of parallel transport are runned simultaneously and one can further optimize this to accelerate.  
- `H2frame.pt` and `HyperPT.pt`: Edge-wise parameters to support parallel transport on embedding hyperbloic spaces.
- `edge_attr.pt`: Edge-wise  parameters to support parallel transport on embedding spherical spaces.
- `PD.pt`:
- `gradMatrix.pt`: the discrete gradient operator.
- `mass.pt`: the surface area that a point takes up.
- `vec_normal.pt`, `PD.pt`: the normal vector field and principal directions at each node. The latter forms a local coordinate frame. Parallel transport should be computed based on concrete frames while feature updates are frame-independent (they are scalar).
