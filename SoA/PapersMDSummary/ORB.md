This descriptor is one basic approach for which robustness against rotation is not present. An initial $sxs$ patch is defined for which pixel pairs (Xi,Yi) within the patch are compared given a Gaussian distribution. Meaning that the pixels chosen are more concentrated around the center than at the edges of the patch. (Xi, Yi) pairs are randomly sampled from discrete locations of a coarse polar grid. And for each i, Xi is (0,0) whereas Yi takes values on the grid until all pixels are explored exhaustively.

Once the descriptors are built, a distance measure between descriptors of different images is computed. This distance measures the number of different bits between two binary strings.

ORB
===
As an extention of BRIEF, ORB works with an orientation compensation mechanism as well as aiming to learn sampling pairs instead of working with randomly chosen pairs from a Gaussian distribution.

First, the compensation in orientation, known as measure of corner orientation, works with the moments of a patch for which the center of mass of the patch is found, a vector is then built from the corner's center to the centroid, and the arctangent between these two vectors is latter computed. The angle obtained from the computation can be used to compensate the angle change making the algorithm robust to rotation.

The latter, learning sampling pairs, works under the premises of uncorrelation and high variance of pairs. This means that the sampling pairs will be uncorrelated aiming to bring new information to the descriptor, while making the features more discriminative given the high variance of pairs. 
