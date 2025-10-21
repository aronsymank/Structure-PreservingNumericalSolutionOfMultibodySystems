# Structure-Preserving Numerical Solution of Multibody Systems <br> (Master Thesis)
My master thesis in Scientific Computing at Technical University Berlin. I investigated and implemented several numerical solvers for constrained multibody systems with a focus on conservation of energy / power balance and constraints.   
  
The solvers are:    

**Classic Runge-Kutta Methods:**
- Gauss-Legendre Collocation
- Lobatto IIIC

**Partitioned Runge-Kutta Methods:** 
- The Lobatto IIIA-IIIB pair as described by Jay, see [here](https://epubs.siam.org/doi/10.1137/0733019)
- A variation of the classic partitioning scheme suggested by Murua, see [here](https://www.researchgate.net/publication/220261083_Partitioned_Half-Explicit_Runge-Kutta_Methods_for_Differential-Algebraic_Systems_of_Index_2)


The thesis is located in /thesis, the implementations in /src.   
The non-partitioned solvers are tested on a simple (unconstrained) mass-spring-damper system.  
The partitioned solvers are tested on a simple and double pendulum, formulated as a constrained multibody system. 
