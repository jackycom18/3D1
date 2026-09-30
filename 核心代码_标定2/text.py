import numpy
file = r"C:\python\3D_project_3\核心代码_标定2\data\H.npz"
data = numpy.load(file)
print(data['H'])