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

## Final Paper
To read the full paper, please visit [this link](https://github.com/micahreich/VisualEssence/blob/master/ve_paper_fnl.pdf).
