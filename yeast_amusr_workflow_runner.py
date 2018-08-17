from inferelator_ng.amusr_tfa_workflow import AMuSR_Workflow


workflow = AMuSR_Workflow()

# Common configuration parameters
workflow.input_dir = 'data/yeast_MTL'
workflow.expression_filelist = ['expression_scer1.tsv', 'expression_scer2.tsv', 'expression_scer3.tsv']
workflow.tf_names_file = 'tf_names.tsv'
workflow.delTmax = 60
workflow.delTmin = 0
workflow.tau = 15
workflow.n_tasks = 3
workflow.num_bootstraps = 20
workflow.priors_filelist = ['yeast-motif-prior.tsv', 'yeast-motif-prior.tsv', 'yeast-motif-prior.tsv']
workflow.gold_standard_filelist = ['gold_standard.tsv', 'gold_standard.tsv','gold_standard.tsv']
workflow.output_dir = 'MTL_yeast_motif_prior'
workflow.tasks_dir = ['scer1', 'scer2', 'scer3']
workflow.run()
