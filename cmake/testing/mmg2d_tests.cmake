## =============================================================================
##  This file is part of the mmg software package for the tetrahedral
##  mesh modification.
##**  Copyright (c) Bx INP/Inria/UBordeaux/UPMC, 2004- .
##
##  mmg is free software: you can redistribute it and/or modify it
##  under the terms of the GNU Lesser General Public License as published
##  by the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  mmg is distributed in the hope that it will be useful, but WITHOUT
##  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
##  FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
##  License for more details.
##
##  You should have received a copy of the GNU Lesser General Public
##  License and of the GNU General Public License along with mmg (in
##  files COPYING.LESSER and COPYING). If not, see
##  <http://www.gnu.org/licenses/>. Please read their terms carefully and
##  use this copy of the mmg distribution only if you accept them.
## =============================================================================

###############################################################################
#####
#####         Continuous Integration
#####
###############################################################################

# Simple test: must already pass
ADD_TEST(NAME mmg2d_SimpleCircle
  COMMAND ${EXECUT_MMG2D} -v 5
  ${MMG2D_CI_TESTS}/Circle/cercle
  -out ${CTEST_OUTPUT_DIR}/mmg2d_SimpleCircle-cercle.o.meshb)


###############################################################################
#####
#####         Input/Output
#####
###############################################################################

# Binary gmsh
ADD_TEST(NAME mmg2d_binary_gmsh_2d
  COMMAND ${EXECUT_MMG2D} -v 5
  ${MMG2D_CI_TESTS}/GmshInout/cercle1.mshb
  ${CTEST_OUTPUT_DIR}/mmg2d_binary_gmsh_2d-cercle)

# Ascii gmsh
ADD_TEST(NAME mmg2d_ascii_gmsh_2d
  COMMAND ${EXECUT_MMG2D} -v 5
  ${MMG2D_CI_TESTS}/GmshInout/cercle1.msh
  ${CTEST_OUTPUT_DIR}/mmg2d_ascii_gmsh_2d-cercle)



###############################################################################
#####
#####         Isotropic cases
#####
###############################################################################
ADD_TEST(NAME mmg2d_SquareIso
  COMMAND ${EXECUT_MMG2D} -v 5
  ${MMG2D_CI_TESTS}/SquareIso/carretest
  -out ${CTEST_OUTPUT_DIR}/mmg2d_SquareIso-carretest.o.meshb)

ADD_TEST(NAME mmg2d_SquareIso_nonConstant
  COMMAND ${EXECUT_MMG2D} -v 5
  ${MMG2D_CI_TESTS}/SquareIso/non-constant
  -out ${CTEST_OUTPUT_DIR}/mmg2d_non-constant.o.meshb)

ADD_TEST(NAME mmg2d_SquareIso_nonConstant2
  COMMAND ${EXECUT_MMG2D} -v 5
  ${MMG2D_CI_TESTS}/SquareIso/non-constant-2
  -out ${CTEST_OUTPUT_DIR}/mmg2d_non-constant-2.o.meshb)

####### -nosurf option
ADD_TEST(NAME mmg2d_2squares
  COMMAND ${EXECUT_MMG2D} -msh 2 -hmax 1 -nosurf -v 5
  ${MMG2D_CI_TESTS}/2squares/2squares
  -out ${CTEST_OUTPUT_DIR}/mmg2d_2squares-2squares.o.meshb)


###############################################################################
#####
#####         Anisotropic cases
#####
###############################################################################
ADD_TEST(NAME mmg2d_SquareAniso
  COMMAND ${EXECUT_MMG2D} -v 5
  ${MMG2D_CI_TESTS}/SquareAniso/adap1
  ${CTEST_OUTPUT_DIR}/mmg2d_SquareAniso-mmg2d_SquareAniso-adap1.o.meshb)

ADD_TEST(NAME mmg2d_CircleOptimAni
  COMMAND ${EXECUT_MMG2D} -v 5 -optim -A -sol 2
  ${MMG2D_CI_TESTS}/Circle/cercle
  -out ${CTEST_OUTPUT_DIR}/mmg2d_CircleOptimAni-cercle.o.mesh)

###############################################################################
#####
#####         Mesh generation
#####
###############################################################################
ADD_TEST(NAME mmg2d_SquareGeneration
  COMMAND ${EXECUT_MMG2D} -v 5
  ${MMG2D_CI_TESTS}/SquareGeneration/carretest
  ${CTEST_OUTPUT_DIR}/mmg2d_SquareGeneration-carretest.o.meshb)

ADD_TEST(NAME mmg2d_NacaGeneration
  COMMAND ${EXECUT_MMG2D} -v 5 -hausd 0.001
  ${MMG2D_CI_TESTS}/NacaGeneration/naca
  -out ${CTEST_OUTPUT_DIR}/mmg2d_NacaGeneration-naca.o.meshb)

###############################################################################
#####
#####         Implicit domain discretization
#####
###############################################################################
#ADD_TEST(NAME mmg2d_LSDiscretization
#  COMMAND ${EXECUT_MMG2D} -v 5 -ls
#  ${MMG2D_CI_TESTS}/LSDiscretization/dom
#  -out ${CTEST_OUTPUT_DIR}/mmg2d_LSDiscretization-dom.o.meshb)
#
#ADD_TEST(NAME mmg2d_LSDiscretization2
#  COMMAND ${EXECUT_MMG2D} -v 5 -ls
#  ${MMG2D_CI_TESTS}/LSDiscretization/nacai
#  -out ${CTEST_OUTPUT_DIR}/mmg2d_LSDiscretization2-nacai.o.meshb)

ADD_TEST(NAME mmg2d_LSMultiMat
  COMMAND ${EXECUT_MMG2D} -v 5 -ls -hmin 0.005 -hmax 0.1 -hausd 0.001 -hgrad 1.3
  ${MMG2D_CI_TESTS}/LSMultiMat/multi-mat
  ${CTEST_OUTPUT_DIR}/mmg2d_LSMultiMat.o.meshb)
