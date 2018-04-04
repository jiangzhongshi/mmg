
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <tuple>

extern "C" {
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>

/** Include the mmgs library hader file */
// if the header file is in the "include" directory
// #include "libmmgs.h"
// if the header file is in "include/mmg/mmgs"
#include "mmg/mmgs/libmmgs.h"
}
#define MAX0(a,b)     (((a) > (b)) ? (a) : (b))
#define MAX3(a,b,c)  (((MAX0(a,b)) > c) ? (MAX0(a,b)) : c)

int not_main(Eigen::MatrixXd&V, Eigen::MatrixXi&F, float input_haus) {
  MMG5_pMesh      mmgMesh;
  MMG5_pSol       mmgSol;
  int             ier,k;
  /* To save final mesh in a file */
  FILE*           inm;
  /* To manually recover the mesh */
  int             np, nt, na, nc, nr, nreq, typEntity, typSol;
  int             ref, Tria[3], Edge[2], *corner, *required, *ridge;
  double          Point[3],Sol;

  fprintf(stdout,"  -- TEST MMGSLIB \n");

  /* Name and path of the mesh file */


  /** ------------------------------ STEP   I -------------------------- */
  /** 1) Initialisation of mesh and sol structures */
  /* args of InitMesh:
   * MMG5_ARG_start: we start to give the args of a variadic func
   * MMG5_ARG_ppMesh: next arg will be a pointer over a MMG5_pMesh
   * &mmgMesh: pointer toward your MMG5_pMesh (that store your mesh)
   * MMG5_ARG_ppMet: next arg will be a pointer over a MMG5_pSol storing a metric
   * &mmgSol: pointer toward your MMG5_pSol (that store your metric) */
  mmgMesh = NULL;
  mmgSol  = NULL;

  MMGS_Init_mesh(MMG5_ARG_start,
                 MMG5_ARG_ppMesh,&mmgMesh,MMG5_ARG_ppMet,&mmgSol,
                 MMG5_ARG_end);

  /* Global hausdorff value (default value = 0.01) applied on the whole boundary */
  if ( MMGS_Set_dparameter(mmgMesh,mmgSol,MMGS_DPARAM_hausd, input_haus) != 1 )
    exit(EXIT_FAILURE);

  /* Gradation control */
  if ( MMGS_Set_dparameter(mmgMesh,mmgSol,MMGS_DPARAM_hgrad, 2) != 1 )
    exit(EXIT_FAILURE);

  /** 2) Build mesh in MMG5 format */
  /** Two solutions: just use the MMGS_loadMesh function that will read a .mesh(b)
      file formatted or manually set your mesh using the MMGS_Set* functions */

  /** Manually set of the mesh */
  /** a) give the size of the mesh: 12 vertices, 12 tetra, 20 triangles, 0 edges */
  if ( MMGS_Set_meshSize(mmgMesh,V.rows(),F.rows(),0) != 1 )  exit(EXIT_FAILURE);

  /** b) give the vertices: for each vertex, give the coordinates, the reference
      and the position in mesh of the vertex */
  for (int v=0; v<V.rows(); v++)
    if ( MMGS_Set_vertex(mmgMesh,V(v,0) , V(v,1) ,V(v,2)  ,0,  v+1) != 1 )  exit(EXIT_FAILURE);

  for (int v=0; v<F.rows(); v++)
    if ( MMGS_Set_triangle(mmgMesh,F(v,0)+1 , F(v,1)+1 ,F(v,2) +1 ,3,  v+1) != 1 )  exit(EXIT_FAILURE);

  /** 4) (not mandatory): check if the number of given entities match with mesh size */
  if ( MMGS_Chk_meshData(mmgMesh,mmgSol) != 1 ) exit(EXIT_FAILURE);


  /**------------------- First wave of refinment---------------------*/



  /** ------------------------------ STEP  II -------------------------- */
  /** remesh function */
  ier = MMGS_mmgslib(mmgMesh,mmgSol);

  if ( ier == MMG5_STRONGFAILURE ) {
    fprintf(stdout,"BAD ENDING OF MMGSLIB: UNABLE TO SAVE MESH\n");
    return (ier);
  } else if ( ier == MMG5_LOWFAILURE )
    fprintf(stdout,"BAD ENDING OF MMGSLIB\n");


  /** ------------------------------ STEP III -------------------------- */
  /** get results */
  /** Two solutions: just use the MMGS_saveMesh/MMGS_saveSol functions
      that will write .mesh(b)/.sol formatted files or manually get your mesh/sol
      using the MMGS_getMesh/MMGS_getSol functions */

  /** 1) Manually get the mesh */

  /** a) get the size of the mesh: vertices, tetra, triangles, edges */
  if ( MMGS_Get_meshSize(mmgMesh,&np,&nt,&na) !=1 )  exit(EXIT_FAILURE);

  /* Table to know if a vertex is corner */
  corner = (int*)calloc(np+1,sizeof(int));

  /* Table to know if a vertex/tetra/tria/edge is required */
  required = (int*)calloc(MAX3(np,nt,na)+1 ,sizeof(int));
  /* Table to know if a coponant is corner and/or required */
  ridge = (int*)calloc(na+1 ,sizeof(int));

  nreq = 0; nc = 0;
  fprintf(stdout,"\nVertices\n%d\n",np);

  Eigen::MatrixXd new_V(np,3);
  Eigen::MatrixXi new_F(nt,3);
  for(k=1; k<=np; k++) {
    /** b) Vertex recovering */
    if ( MMGS_Get_vertex(mmgMesh,&(Point[0]),&(Point[1]),&(Point[2]),
                          &ref,&(corner[k]),&(required[k])) != 1 )
      exit(EXIT_FAILURE);
    new_V.row(k-1)<<Point[0], Point[1], Point[2];
  }

  nreq = 0;
  fprintf(stdout,"\nTriangles\n%d\n",nt);
  for(k=1; k<=nt; k++) {
    /** d) Triangles recovering */
    if ( MMGS_Get_triangle(mmgMesh,&(Tria[0]),&(Tria[1]),&(Tria[2]),
                            &ref,&(required[k])) != 1 )
      exit(EXIT_FAILURE);
    new_F.row(k-1)  << Tria[0] - 1,Tria[1] - 1,Tria[2] - 1;
  }

  nreq = 0;nr = 0;


  free(corner);
  corner = NULL;
  free(required);
  required = NULL;
  free(ridge);
  ridge    = NULL;

  /** 2) Manually get the solution */
  /** a) get the size of the sol: type of entity (SolAtVertices,...),
      number of sol, type of solution (scalar, tensor...) */

  /** 3) Free the MMGS structures */
  MMGS_Free_all(MMG5_ARG_start,
                MMG5_ARG_ppMesh,&mmgMesh,MMG5_ARG_ppMet,&mmgSol,
                MMG5_ARG_end);

  V = new_V;
  F = new_F;
  return 0;
}

namespace py = pybind11;
PYBIND11_MODULE(pymmgs, m) {
    m.doc() = R"(as a test)";

    m.def("add_any", [](py::EigenDRef<Eigen::MatrixXd> x, int r, int c, double v) { x(r,c) += v; });
    m.def("MMGS", [](const Eigen::MatrixXd&V, const Eigen::MatrixXi&F, float ih) {
      Eigen::MatrixXd nV=V;
      Eigen::MatrixXi nF=F;
      if (not_main(nV,nF, ih) != 0) return std::make_tuple(Eigen::MatrixXd(), Eigen::MatrixXi());
      return std::make_tuple(nV,nF);
    });
}