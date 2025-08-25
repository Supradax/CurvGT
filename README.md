### Curvature-aware Graph Attention for PDEs on Manifolds
---
Deep models have recently achieved remarkable performances in solving partial differential equtions (PDEs). The previous methods are mostly focused on PDEs arising in Euclidean spaces with less emphasis on the general manifolds with rich geometry. Several proposals attempt to account for the geometry by exploiting the spatial coordinates but overlook the underlying intrinsic geometry of manifolds. In this paper, we propose a Curvature-aware Graph Attention for PDEs on manifolds by exploring the important intrinsic geometric quantities such as curvature and discrete gradient operator. It is realized via parallel transport and tensor field on manifolds. To accelerate computation, we present three curvatureoriented graph embedding approaches and derive closed-form parallel transport equations, and a sub-tree partition method is also developed to promote parameter-sharing. Our proposed curvature-aware attention can be used as a replacement for vanilla attention, and experiments show that it significantly improves the performance of the existing methods for solving PDEs on manifolds.

[Curvature-aware Graph Attention for PDEs on Manifold[paper]](https://openreview.net/forum?id=vWYLQ0VPJx&noteId=vWYLQ0VPJx)

You can cite our paper by (.bibtex): 
```
@inproceedings{
liao2025curvatureaware,
title={Curvature-aware Graph Attention for {PDE}s on Manifolds},
author={Yunfeng Liao and Jiawen Guan and Xiucheng Li},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=vWYLQ0VPJx}
}
```

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
- `GaussianCurvature.pt`: Each node is assigned with a scalar, standing for the Gaussian curvature estimate down there. We leverage a mask (`option_mask.pt`) to decide the correct parallel transport (Euclidean, spheric and hyperbolic). Thus three types of parallel transport are runned simultaneously and one can further optimize this to accelerate. To fetch these curvature estimates $\kappa$, we use an excellent open geometry process C++ library `libigl` where these estimates along with the principal direction of curvature are also available. Only mesh documents in common formats `.obj`, `.off` etc are needed. Then the `option_mask` flags the type of space embedded in: `+1` for $\kappa>\varepsilon$ (spheric), `-1` for $\kappa < -\varepsilon$ (hyperbolic) and otherwise `0` for Euclidean.

- `H2frame.pt` and `HyperPT.pt`: Edge-wise parameters to support parallel transport on embedding hyperbloic spaces. The concrete formulae can be found in the paper and the appendix. We mention that `H2frame.pt` contain two 3D vectors at each point with positive and negative principal curvature respectively, which can also be obtained in `libigl`. `HyperPt.pt` involves the works to compute the parallel transport on $\mathbb H^2$ as mentioned in the paper. For each vertex $v$, we place it on $(0,1)$ in $\mathbb H^2$ and note that two coordinate-axes are exactly two principal directions. This observation guides us how to place $v$'s neighbors around $v$ on $\mathbb H^2$. In the sense of isometric geodesic embedding, assume two vertices $u$ and $v$ are with 3D coordinates $\mathbf x_u$ and $\mathbf x_v$, then we project $\mathbf x_v-\mathbf x_u$ back to the estimated tangent plane $T_u\mathcal M$ at $\mathbf x_u$. This tangent plane can be estimated by various ways, the most famous and simple one is via PCA(Principle Component Anaysis). Then this projected tangent vector on $\mathcal M$(or more exactly, the pseudo sphere) can be mapped to the tangent vector in $\mathbb H^2$, denoted by $\mathbf p$. With exponential map in $\mathbb H^2$, one can find a unique geodesic starting at $(0,1)$ with velocity $\mathbf p$. Consequently, the neighbor falls on this geodesic (either a line parallel to $y$-axis or a semicircle centered at the $x$-axis). We choose the one such that the hyperbolic distance equaling their Euclidean counterpart $||\mathbf x_u-\mathbf{x}_v||_2^2$. 
   
- `edge_attr.pt`: Edge-wise parameters to support parallel transport on embedding spherical spaces. The sphere case is similar to the hyperbolic one. One can get them by serveral steps: (i) find the tangent plane; (ii) estimate the tangent vector via projection; (iii) find the geodesic begining at this point with the direction along this tangent vector; (iv) determine the exact point on this geodesic such that the spherical distance equals the Euclidean counterpart.

The following ones can be gained easily by simpily calling APIs offered by `libigl`:
- `gradMatrix.pt`: the discrete gradient operator. 
- `mass.pt`: the surface area that a point takes up.
- `vec_normal.pt`, `PD.pt`: the normal vector field and principal directions at each node. The latter forms a local coordinate frame. Parallel transport should be computed based on concrete frames while feature updates are frame-independent (they are scalar).

**More Implementation Details**. We use principal directions as a coordinate chart on the surface. It may degenerate sometimes when the curvature tensor is isotropic (e.g., a sphere). But in most cases, these irregular points are rare and can be fixed via interpolations with the neighbors. Each time we compute message passing on surfaces, we project the 3D vectors onto the 2D tangent planes embedded in $\mathbb R^3$, whose these 2D coordinate components are served as inputs and outputs of the network, and transform them back to 3D ones if necessary. This approach makes the entire process working on the embedded manifold, which is independent of the 3D space it embeds in.
