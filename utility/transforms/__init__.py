from .landmarks_transform import CenterInLandmarksCenterOfMass, RegisterIntoLandmarksEyes, OneHotEncodingNodes, LandmarksToPos
from .geodesic_distances import ComputeGeodesicBetweenLandmarks, ComputeSplineFromGeodesicPaths
from .fpfh import Fpfh, NormalizeFpfh
from .dataset_alignment import Procrustes
from .knn import KNNGraphLandmarks, FullyConnectedGraph
from .Age2Num import Age2Num

transforms = ["CenterInLandmarksCenterOfMass", "RegisterIntoLandmarksEyes",
              "ComputeGeodesicBetweenLandmarks", "Procrustes", "KNNGraphLandmarks", 
              "Fpfh", "OneHotEncodingNodes", "LandmarksToPos", "FullyConnectedGraph",
              "ComputeSplineFromGeodesicPaths", "Age2Num"]

__all__ = transforms
