# VisualEssence
## Project Structure
Below is the general project structure for CVMLP Lab's VisualEssence project. Continue reading for more information on each module involved in the project.
```bash
.
├── CNN                  # Icon discriminator, classifies singular vs. icon arrangements
    ├── DataVis              # Data visualization of CNN results and input dataset
    ├── IconGeneration       # Generates convex-hull based negative samples for CNN dataset
    └── StyleClustering      # Performs k-means clustering for icon style classification
├── FeatureExtractor     # Salient word extractor, grabs 3 most salient words from input sentence
├── Generator            # Generates icon arrangement coordinates for icon triplet inputs
    └── IconGeneration       # Generates convex-hull based samples for label generation
├── data                 # Location for all datasets
```
## CNN
Enter the VisualEssence/CNN/StyleClustering directory via command line
```bash
.
├── CNN
    ├── DataVis
        └── PixelHistogram.py    # Creates a histogram based on black pixel frequency of CNN dataset
    ├── IconGeneration
        └── ConvexHull.py        # Creates a convex-hull based negative sample with random coordinates for icon arrangements
    ├── StyleClustering
        ├── ClusteringDataGen.py # Creates, downloads, serializes k-means dataset for clustering
        └── StyleCluster.py      # Creates k-means model for clustering, includes training and inferencing functions
    ├── DiscriminatorDataGen.py  # Creates, downloads, serializes CNN dataset for classification
    ├── FileCheck.py             # Gives info re downloaded dataset and negative sample count
    ├── IconDiscriminator.py     # Creates CNN model for classification
    └── ModelValidation.py       # Generates high-scoring arrangements based on n searches through possible arrangements as scored by trained discriminator
```
## FeatureExtractor
```bash
.
├── FeatureExtractor
    └── DescriptorGen            # Performs TF-IDF feature extraction based on dataset, extracts 3 salient words from input sentences
```
## Generator
## data
