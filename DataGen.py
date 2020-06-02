import CNN.CNNDataGen
import sys

DataGen = CNN.CNNDataGen.DatasetGenerator(int(sys.argv[1]), sys.argv[2])
DataGen.get_images(sys.argv[2])
