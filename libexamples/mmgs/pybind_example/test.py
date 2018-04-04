import pymmgs
import numpy as np
import sys,os
sys.path.insert(0, os.path.expanduser('~/Workspace/libigl/python'))
import pyigl as igl
import pyigl.eigen as Eigen
from iglhelpers import p2e, e2p

V= igl.eigen.MatrixXd()
F = igl.eigen.MatrixXi()
igl.read_triangle_mesh('/Users/zhongshi/1399_standing_clap_000010.obj', V, F)


# SV, SVI, SVJ, SF =  igl.eigen.MatrixXd(),  igl.eigen.MatrixXi(), igl.eigen.MatrixXi(), igl.eigen.MatrixXi()
# igl.remove_duplicate_vertices(V,F,1e-10, SV, SVI, SVJ, SF)
M = igl.eigen.MatrixXd()
igl.doublearea(V,F,M)
M = e2p(M).flatten()

# F0 = e2p(F)[np.where(M > 1e-9)[0],:]

# npl = np.load('/Users/zhongshi/1006_jump_from_wall_000003.obj.npz')
V0, F0 = e2p(V), e2p(F)


# igl.writeOBJ('1399.obj',V, F)
V,F = pymmgs.MMGS(V0, F0,0.005)
vw = igl.glfw.Viewer()
vw.data().set_mesh(p2e(V),p2e(F))
vw.launch()