# Performance of backend

Looks like default solver setting is slow.

First use 
```
    #optimization parameters
    max_solver_time: 1.0  # max solver itration time (ms), to guarantee real time
    max_num_iterations: 8   # max solver itrations, to guarantee real time

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.num_threads = 1;
    options.trust_region_strategy_type = ceres::DOGLEG;
```

Results: 
```
[ INFO] [1649145990.764599115, 1644397477.213071876]: AVG Iter 6.822272 time 313.818019ms Iterations : 4 solver costs: 211.566593 

time: 1644397477.134354, kf: 0 t: 7.597 -3.363 0.964 q: 0.83 -0.00 -0.01 0.56 td: 7.5ms backend_time: 334.2ms

Solver Summary (v 2.0.0-eigen-(3.3.4)-lapack-suitesparse-(5.1.2)-cxsparse-(3.1.9)-eigensparse-no_openmp) #Report for last optimization

                                     Original                  Reduced
Parameter blocks                          107                      107
Parameters                                273                      273
Effective parameters                      260                      260
Residual blocks                          1512                     1512
Residuals                                3228                     3228

Minimizer                        TRUST_REGION

Dense linear algebra library            EIGEN
Trust region strategy                  DOGLEG (TRADITIONAL)

                                        Given                     Used
Linear solver                     DENSE_SCHUR              DENSE_SCHUR
Threads                                     1                        1
Linear solver ordering              AUTOMATIC                    87,20
Schur structure                         d,d,d                    d,d,d

Cost:
Initial                          6.279035e+02
Final                            4.113649e+02
Change                           2.165386e+02

Minimizer iterations                        7
Successful steps                            7
Unsuccessful steps                          0

Time (in seconds):
Preprocessor                         0.008830

  Residual only evaluation           0.018053 (7)
  Jacobian & residual evaluation     0.065741 (7)
  Linear solver                      0.108792 (7)
Minimizer                            0.339927

Postprocessor                        0.000157
Total                                0.348914
```

Then 

First use 
```
    #optimization parameters
    max_solver_time: 1.0  # max solver itration time (ms), to guarantee real time
    max_num_iterations: 8   # max solver itrations, to guarantee real time

    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.num_threads = 1;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
```


Result

```
[ INFO] [1649146413.557104861, 1644397477.213071876]: AVG Iter 7.422947 time 274.239639ms Iterations : 8 solver costs: 312.746819 

time: 1644397477.134354, kf: 0 t: 7.616 -3.423 0.850 q: 0.83 -0.00 -0.01 0.56 td: 7.5ms backend_time: 294.6ms

Solver Summary (v 2.0.0-eigen-(3.3.4)-lapack-suitesparse-(5.1.2)-cxsparse-(3.1.9)-eigensparse-no_openmp)

                                     Original                  Reduced
Parameter blocks                          107                      107
Parameters                                273                      273
Effective parameters                      260                      260
Residual blocks                          1512                     1512
Residuals                                3228                     3228

Minimizer                        TRUST_REGION

Sparse linear algebra library    SUITE_SPARSE
Trust region strategy     LEVENBERG_MARQUARDT

                                        Given                     Used
Linear solver          SPARSE_NORMAL_CHOLESKY   SPARSE_NORMAL_CHOLESKY
Threads                                     1                        1
Linear solver ordering              AUTOMATIC                      107

Cost:
Initial                          6.127374e+02
Final                            4.104391e+02
Change                           2.022983e+02

Minimizer iterations                        8
Successful steps                            8
Unsuccessful steps                          0

Time (in seconds):
Preprocessor                         0.003552

  Residual only evaluation           0.020241 (8)
  Jacobian & residual evaluation     0.072665 (8)
  Linear solver                      0.058706 (8)
Minimizer                            0.307576

Postprocessor                        0.000140
Total                                0.311268

Termination:                      CONVERGENCE (Function tolerance reached. |cost_change|/cost: 1.884273e-07 <= 1.000000e-06)
```

Average iteration speed: 45.9ms for DOGLEG and DENSE_SCHUR 36.9ms for CHELOSKY and LM.