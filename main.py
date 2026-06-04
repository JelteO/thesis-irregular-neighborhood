# main of thesis project Jelte Oldenhof
from src.evaluation.experiment import experiment_pca, experiment_if, experiment_schreyer, experiment_dominant

print("\n----- pca -----n")
df_pca, results_pca = experiment_pca()

print("\n----- isolation forest -----n")
df_if, results_if = experiment_if() 

print("\n----- schreyer ae -----n")
df_schreyer, results_schr = experiment_schreyer()

print("\n ----- dominant ----- \n")
df_dom, results_dom = experiment_dominant()