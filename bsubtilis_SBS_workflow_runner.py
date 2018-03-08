
from inferelator_ng.multitask_sparse_blocksparse_workflow import MTL_SBS_Workflow

workflow = MTL_SBS_Workflow()
idx = 1
# Common configuration parameters
workflow.input_dir = 'data/bsubtilis_MTL'
workflow.expression_filelist = ["expression_py79.tsv", "expression_bsb1.tsv"]
workflow.meta_data_filelist = ["meta_data_py79.tsv", "meta_data_bsb1.tsv"]
workflow.tf_names_file = 'tf_names_MTL.tsv'
workflow.delTmax = 60
workflow.delTmin = 0
workflow.tau = 15
workflow.n_tasks = 2
workflow.num_bootstraps = 2
workflow.priors_filelist = ['gold_standard.tsv', 'gold_standard.tsv']
workflow.gold_standard_filelist = ['gold_standard.tsv', 'gold_standard.tsv']
workflow.output_dir = 'MTLkvs'
workflow.tasks_dir = ['py79', 'bsb1']
workflow.run()

