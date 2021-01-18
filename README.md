## RunTime in Different HardWare (RTDHW)
This is the repo of the paper _On the fair comparison of algorithms in different machines._ 
In this paper, we introduce a methodology to statistically asses the difference between the performance of two optimization algorithms executed in different machines.
The optimization problem is assumed to be a minimization problem.
In the following, we present two examples of how this methodology can be applied.
 
### Example I


In this example, we will compare a simple random initialization local search procedure with a memetic search algorithm by Benlic et al.[1] for the QAP.
Using the proposed methodology, we will statistically asses the difference in performance between these two algorithms, without having to actually execute the code of the memetic algorithm.
In this case, the memetic search algorithm is algorithm A, because this is the algorithm that is not executed and the local search algorithm is algorithm B, because this is the algorithm whose runtime is predicted.

__Step 1: Obtaining the data__

To apply the proposed methodology, we need to find certain information about the execution of the memetic algorithm.
We need the list of instances to be used in the comparison, the average objective value obtained by the memetic search algorithm and the runtime of the memetic search algorithm in each of the instances.
We list this information extracted from the article by Benlic et al.[1] in the table below.
In addition, we need to find the CPU model of the machine in which the memetic search was run (machine M<sub>1</sub>), which is "Intel Xeon E5440 2.83GHz" as specified in their article.
Finally, the machine score of this CPU, measured as PassMark <ins>single thread</ins> score is s<sub>1</sub> = 1230, which can be found out in the PassMark website.
 


| Instance   |      Runtime in seconds, t<sub>1</sub>   |  Objective value, a<sub>i</sub> |
|----------|:-------------:|------:|
tai40a                               |                                     486 |               3141222 |
tai50a                               |                                    2520 |               4945266 |
tai60a                               |                                    4050 |               7216339 |
tai80a                               |                                    3948 |              13556691 |
tai100a                              |                                    2646 |              21137728|
tai50b                               |                                      72 |             458821517 |
tai60b                               |                                     312 |             608215054 |
tai80b                               |                                    1878 |             818415043 |
tai100b                              |                                     816 |            1185996137 |
tai150b                              |                                    4686 |             499195981 |
sko100a                              |                                    1338 |                115534 |
sko100b                              |                                     390 |                152002 |
sko100c                              |                                     720 |                147862 |
sko100d                              |                                    1254 |                149584 |
sko100e                              |                                     714 |                149150 |
sko100f                              |                                    1380 |                149036 |


__Step 2: Predicting the equivalent runtime__

With the data already gathered, we need to predict the equivalent runtime of each instance for the machine in which the local search algorithm will be executed (machine M<sub>2</sub>).
To make the prediction, we need the machine score s<sub>2</sub> of this machine.
The CPU model of M<sub>2</sub> is "Intel Celeron N4100", with a PassMark <ins>single thread</ins> score of s<sub>2</sub> = 1032.
With this information, we are ready to predict the equivalent runtime in machine M<sub>2</sub> with the following formula (Equation (6) in the paper):

<img src="/github_readme_assets/images/equation_to_predict_equivalent_runtime_case_study_1.png" width="600">


where t<sub>1</sub> is substituted with the runtime of the memetic search algorithm in each instance, listed in the table above.


__Step 3: Running the experiments__

Now, we execute the local search algorithm in the instances listed in the above table, using the predicted runtimes t̂<sub>2</sub> as stopping criterion.
This execution is carried out on machine M<sub>2</sub>, and the best objective function values b̂ are listed in the table below.
Following the procedure by Benlic et al.[1], these best objective values are averaged over 10 executions.


| Instance   |      Predicted runtime, t̂<sub>2</sub>   |  Objective value, b̂<sub>i</sub> |
|----------|:-------------:|------:|
tai40a                               |                           292.09 |                               3207604 |
tai50a                               |                          1514.52 |                               5042830 |
tai60a                               |                          2434.05 |                               7393900 |
tai80a                               |                          2372.75 |                               13840668 |
tai100a                              |                          1590.25 |                               21611122 |
tai50b                               |                            43.27 |                               459986202 |
tai60b                               |                           187.51 |                               609946393 |
tai80b                               |                          1128.68 |                               824799510 |
tai100b                              |                            490.42 |                               1195646366 |
tai150b                              |                           2816.29 |                               505187740 |
sko100a                              |                            804.14 |                               153082 |
sko100b                              |                           234.39 |                               155218 |
sko100c                              |                           432.72 |                               149076 |
sko100d                              |                           753.65 |                               150568 |
sko100e                              |                           429.11 |                               150638 |
sko100f                              |                           829.38 |                               150006 |



__Step 4: Obtaining the statistic and comparing it with the corrected critical value__


Once we have all the results, we need to compute the statistic #{a<sub>i</sub> < b̂<sub>i</sub>}, which counts the number of times that a<sub>i</sub> < b̂<sub>i</sub>.
In this case, a<sub>i</sub> < b̂<sub>i</sub> happens 15 times, and therefore, #{a<sub>i</sub> < b̂<sub>i</sub>} = 15.
Now we compare the value of the statistic with the corrected critical value for a sample size (the number of instances in which a<sub>i</sub> ≠ b̂<sub>i</sub>) n = 15 and a target probability of type I error of 𝛼 = 0.05.	
The corrected critical value for this sample size and 𝛼 is Crit<sub>c</sub>(𝛼) = 3, as shown in the [corrected_critical_values.pdf](corrected_critical_values.pdf) file.
The observed statistic #{a<sub>i</sub> < b̂<sub>i</sub>} = 15 is <ins>not</ins> lower or equal to the corrected critical value Crit<sub>c</sub>(𝛼) = 3.
	


__Step 5: Conclusion__

	
Since the observed statistic is <ins>not</ins> lower or equal to the critical value, we cannot reject H<sub>0</sub>.
In this case, the conclusion is that with the amount of data that we have and the chosen target probability of type I error of 𝛼 = 0.05, we can not say that the local search algorithm has an statistically significant better performance than the memetic search algorithm.
It would <ins>not</ins> be correct to conclude that the two algorithms perform statistically significantly the same, or that the memetic search performs statistically significantly better that the local search.

It is important to note that, if we had considered the original runtimes t_1 as the stopping criterion for algorithm B in machine M<sub>2</sub> (longer than the estimated equivalent runtime t̂<sub>2</sub>), the local search would have had an unfairly longer runtime.
In other words, the comparison would have been biased towards the local search. 
-----------------------------------



### Example II



In this second example, we will compare the same simple random initialization local search procedure with an estimation of distribution algorithm (EDA) for the QAP[2].
In this case, the EDA is algorithm A, because this is the algorithm that is not executed and the local search algorithm is algorithm B, because this is the algorithm whose runtime is predicted.

__Step 1: Obtaining the data__

To apply the proposed methodology, we need to find certain information about the execution of the memetic algorithm.
We need the list of instances to be used in the comparison, the average objective value obtained by the EDA and the runtime used in each instances.
We list this information extracted from the paper[2] in the table below.
In addition, we need to find the CPU model of the machine in which the memetic search was run (machine M<sub>1</sub>), which is "AMD Ryzen 7 1800X", as specified in the paper.
Finally, the machine score of this CPU, measured as PassMark single thread score is s<sub>1</sub> = 2180, which can be found out in the PassMark website.





| Instance   |      Runtime in seconds, t<sub>1</sub>   |  Objective value, a<sub>i</sub> |
|----------|:-------------:|------:|
bur26a                               |                    1.45 |                     5432374 |
bur26b                               |                    1.45 |                     3824798 |
bur26c                               |                    1.43 |                     5427185 |
bur26d                               |                    1.44 |                     3821474 |
nug17                                |                    0.44 |                        1735 |
nug18                                |                    0.51 |                        1936 |
nug20                                |                    0.68 |                        2573 |
nug21                                |                    0.77 |                        2444 |
tai10a                               |                    0.12 |                      135028 |
tai10b                               |                    0.12 |                     1183760 |
tai12a                               |                    0.18 |                      224730 |
tai12b                               |                    0.19 |                    39464925 |
tai15a                               |                    0.31 |                      388910 |
tai15b                               |                    0.31 |                    51768918 |
tai20a                               |                    0.69 |                      709409 |
tai20b                               |                    0.68 |                   122538448 |
tai40a                               |                    5.41 |                     3194672 |
tai40b                               |                    5.41 |                   644054927 |
tai60a                               |                   19.23 |                     7367162 |
tai60b                               |                   19.21 |                   611215466 |
tai80a                               |                   50.09 |                    13792379 |
tai80b                               |                    50.1 |                   836702973 |




__Step 2: Predicting the equivalent runtime__

With the data already gathered, we need to predict the equivalent runtime of each instance for the machine in which the local search algorithm will be executed (machine M<sub>2</sub>).
To make the prediction, we need the machine score s<sub>2</sub> of this machine.
The CPU model of M<sub>2</sub> is "Intel Celeron N4100", with a PassMark single thread score of s<sub>2</sub> = 1032.
With this information, we are ready to predict the equivalent runtime in machine M<sub>2</sub> with the following formula (Equation (6) in the paper):

<img src="/github_readme_assets/images/equation_to_predict_equivalent_runtime_case_study_2.png" width="600">


where t<sub>1</sub> is substituted with the runtime of the memetic search algorithm in each instance, listed in the table above.


__Step 3: Running the experiments__

Now, we execute the local search algorithm in the instances listed in the above table, using the predicted runtimes t̂<sub>2</sub> as stopping criterion.
This execution is carried out on machine M<sub>2</sub>, and the best objective function values b̂ are listed in the table below.
Following the procedure by Arza et al.[2], these best objective values are also averaged over 20 executions.




| Instance   |      Predicted runtime, t̂<sub>2</sub>   |  Objective value, b̂<sub>i</sub> |
|----------|:-------------:|------:|
bur26a|1.57   |  5426670      |
bur26b|1.57   |  3817852      |
bur26c|1.55   |  5426795     |
bur26d|1.56   |   3821239     |
nug17| 0.48   |   1734      |
nug18| 0.55   |   1936      |
nug20| 0.74   |  2570      |
nug21| 0.84   |  2444      |
tai10a|0.13   | 135028     |
tai10b|0.13   |  1183760     |
tai12a|0.20   |  224416     |
tai12b|0.21   |  39464925     |
tai15a|0.34   |  388214     |
tai15b|0.34   |  51765268     |
tai20a|0.75   |  703482     |
tai20b|0.74   |  122455319     |
tai40a|5.87   |   3227894     |
tai40b|5.87   |   637470334     |
tai60a|20.86   |  7461354     |
tai60b|20.84   |  611833935     |
tai80a|54.35   |  13942804     |
tai80b|54.36   |  830729983      |

 

__Step 4: Obtaining the statistic and comparing it with the corrected critical value__


Once we have all the results, we need to compute the statistic #{a<sub>i</sub> < b̂<sub>i</sub>}, which counts the number of times that a<sub>i</sub> < b̂<sub>i</sub>.
In this case, a<sub>i</sub> < b̂<sub>i</sub> happens 4 times, and therefore, #{a<sub>i</sub> < b̂<sub>i</sub>} = 4.
Now we compare the value of the statistic with the corrected critical value for a sample size (the number of instances in which a<sub>i</sub> ≠ b̂<sub>i</sub>) n = 17 and a target probability of type I error of 𝛼 = 0.05.	
The corrected critical value for this sample size and 𝛼 is Crit<sub>c</sub>(𝛼) = 4, as shown in the [corrected_critical_values.pdf](corrected_critical_values.pdf) file.
The observed statistic #{a<sub>i</sub> < b̂<sub>i</sub>} = 4 is lower or equal to the corrected critical value Crit<sub>c</sub>(𝛼) = 4.


__Step 5: Conclusion__

Since the observed statistic is lower or equal to the critical value, we reject H<sub>0</sub>.
In this case, the conclusion is that with a probability of type I error of  𝛼 = 0.05, the performance of the local search procedure is statistically significantly better than the performance of the EDA.

In this case, machine M<sub>1</sub> is more powerful (in terms of computational capabilities) than machine M<sub>2</sub>. 
If we had considered the original runtimes t<sub>1</sub> as the stopping criterion for algorithm B in machine M<sub>2</sub> (shorter than the estimated equivalent runtime t̂<sub>2</sub>), it would have been more difficult for the local search to perform better than the EDA.
In that case, H<sub>0</sub> might not have been rejected.




### References

[1] Benlic, U., & Hao, J.-K. (2015). Memetic search for the quadratic assignment problem. Expert Systems with Applications, 42(1), 584-595. https://doi.org/10.1016/j.eswa.2014.08.011

[2] Arza, E., Pérez, A., Irurozki, E., & Ceberio, J. (2020). Kernels of Mallows Models under the Hamming Distance for solving the Quadratic Assignment Problem. Swarm and Evolutionary Computation, 100740. https://doi.org/10.1016/j.swevo.2020.100740

