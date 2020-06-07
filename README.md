## CNN Training Instructions
Enter the VisualEssence/CNN/StyleClustering directory via command line
```bash
cd ../VisualEssence/CNN/StyleClustering
```
Run the following python files in the following order. First, download the style clustering dataset
```python
# Mode order is: DOWNLOAD, PICKLE
python3 ClusteringDataGen.py
```
Then train the style k-means clustering model
```python
# Mode order is: TRAIN
python3 ClusteringDataGen.py
```
Enter the VisualEssence/CNN directory via command line
```bash
cd ../VisualEssence/CNN
```
Then download the CNN dataset
```python
# Mode order is: DOWNLOAD, NEG_SAMPLE, PICKLE
python3 DiscriminatorDataGen.py
```
Then CNN training can begin
```python
# Mode order for DS is: LOAD_PICKLE
python3 IconDiscriminator.py
```
