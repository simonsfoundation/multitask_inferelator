
from inferelator_ng.amusr_tfa_workflow import AMuSR_Workflow
import argparse

parser = argparse.ArgumentParser(description = 'run MTL Bsubtilis reps with ipcluster -- ipyparallel')
parser.add_argument('-cid','--cluster_id', help = 'cluster id for ipcluster', required = False, default = '')
args = vars(parser.parse_args())

cluster_id = args['cluster_id']


workflow = AMuSR_Workflow()

# Common configuration parameters
workflow.input_dir = 'data/bsubtilis_MTL'
workflow.expression_filelist = ["expression_py79.tsv", "expression_bsb1.tsv"]
workflow.meta_data_filelist = ["meta_data_py79.tsv", "meta_data_bsb1.tsv"]
workflow.tf_names_file = 'tf_names_MTL.tsv'
workflow.delTmax = 60
workflow.delTmin = 0
workflow.tau = 15
workflow.n_tasks = 2
workflow.cluster_id = cluster_id
workflow.num_bootstraps = 2
workflow.priors_filelist = ['gold_standard.tsv', 'gold_standard.tsv']
workflow.gold_standard_filelist = ['gold_standard.tsv', 'gold_standard.tsv']
workflow.output_dir = 'MTL_network'
workflow.tasks_dir = ['py79', 'bsb1']
workflow.run()
