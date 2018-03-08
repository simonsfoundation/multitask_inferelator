"""
Run Multitask Network Inference with TFA-SBS.
"""
import os
import numpy as np
import pandas as pd
from . import utils
from workflow import WorkflowBase
import design_response_R
from tfa import TFA
from sparse_blocksparse import *
from results_processor import ResultsProcessor
from time import localtime, strftime
import datetime
from kvsclient import KVSClient

# Connect to the key value store service (its location is found via an
# environment variable that is set when this is started vid kvsstcp.py
# --execcmd).
kvs = KVSClient()
# Find out which process we are (assumes running under SLURM).
rank = int(os.environ['SLURM_PROCID'])

class MTL_SBS_Workflow(WorkflowBase):

    # Common configuration parameters
    input_dir = 'data'
    expression_filelist = ['expression_matrix_file.tsv', 'expression_matrix_file.tsv']
    tf_names_file = 'tf_names.tsv'
    target_genes_file = 'target_genes.tsv'
    n_tasks = 2
    prior_weight = 1
    meta_data_filelist = None
    priors_filelist = None
    gold_standard_filelist = None
    cluster_id = None
    tasks_dir = [''.join(['task_', str(k)]) for k in range(n_tasks)]
    output_dir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


    def run(self):
        """
        Execute workflow, after all configuration.
        """
        np.random.seed(self.random_seed)

        self.design_response_driver = design_response_R.DRDriver()
        self.regression_method = MT_SBS_regression()
        self.get_data()
        self.compute_activity()

        print('Calculating betas using Multitask Dirty Model')
        print(strftime("%Y-%m-%d %H:%M:%S", localtime()))

        betas = [[] for k in range(self.n_tasks)]
        rescaled_betas = [[] for k in range(self.n_tasks)]

        for idx, bootstrap in enumerate(self.get_bootstraps()):
            print('Bootstrap {} of {}'.format((idx + 1), self.num_bootstraps))
            X = []
            Y = []
            for k in range(self.n_tasks):
                task_workflow_obj = self.workflow_objs[k]
                X.append(task_workflow_obj.activity.ix[:, bootstrap[k]].transpose())
                Y.append(task_workflow_obj.response.ix[:, bootstrap[k]].transpose())

            self.regression_method.n_tasks = self.n_tasks
            self.regression_method.feature_count = len(self.tf_names)
            ownCheck = utils.own(kvs, rank, chunk=25, reset=idx!=0)
            current_betas, current_rescaled_betas = self.regression_method.run(X, Y,
                            self.target_genes, self.tf_names, kvs, rank, ownCheck, 
                            self.cluster_id, self.priors, self.prior_weight)
            for k in range(self.n
                betas[k].append(current_betas[k])
                rescaled_betas[k].append(current_rescaled_betas[k])

        print('Saving outputs')
        print(strftime("%Y-%m-%d %H:%M:%S", localtime()))

        self.emit_results(betas, rescaled_betas)


    def get_data(self):
        """
        Reads in input files and compute common data (design, response, priors...)
        """
        if self.meta_data_filelist is None:
            self.meta_data_filelist = ['metadata_dummy.tsv']*self.n_tasks

        if self.priors_filelist is None:
            self.priors_filelist = ['prior_dummy.tsv']*self.n_tasks

        if self.gold_standard_filelist is None:
            self.gold_standard_filelist = ['gs_dummy.tsv']*self.n_tasks

        workflow_objs = []

        for k in range(self.n_tasks):
            task_workflow_obj = WorkflowBase()
            task_workflow_obj.input_dir = self.input_dir
            task_workflow_obj.delTmax = self.delTmax
            task_workflow_obj.delTmin = self.delTmin
            task_workflow_obj.tau = self.tau
            task_workflow_obj.design_response_driver = self.design_response_driver
            task_workflow_obj.expression_matrix_file = self.expression_filelist[k]
            task_workflow_obj.meta_data_file = self.meta_data_filelist[k]
            task_workflow_obj.priors_file = self.priors_filelist[k]
            task_workflow_obj.gold_standard_file = self.gold_standard_filelist[k]
            task_workflow_obj.tf_names_file = self.tf_names_file
            task_workflow_obj.get_data()
            task_workflow_obj.compute_common_data()
            workflow_objs.append(task_workflow_obj)

        self.workflow_objs = workflow_objs

        tf_file = self.input_file(self.tf_names_file)
        self.tf_names = utils.read_tf_names(tf_file)

        target_genes_file = self.input_file(self.target_genes_file)
        self.target_genes = utils.read_tf_names(target_genes_file)

        self.priors = [self.workflow_objs[k].priors_data for k in range(self.n_tasks)]


    def compute_activity(self):
        """
        Compute Transcription Factor Activity
        """
        for k in range(self.n_tasks):
            print('Computing Transcription Factor Activity for Task ' + str(k))
            TFA_calculator = TFA(self.workflow_objs[k].priors_data, self.workflow_objs[k].design, self.workflow_objs[k].half_tau_response)
            self.workflow_objs[k].activity = TFA_calculator.compute_transcription_factor_activity()


    def get_bootstraps(self):
        """
        Get bootstrap indices for all tasks
        """

        bootstraps = []

        for boot in range(self.num_bootstraps):
            bootstrap = []
            for k in range(self.n_tasks):
                task_workflow_obj = self.workflow_objs[k]
                col_range = range(task_workflow_obj.response.shape[1])
                bootstrap.append([np.random.choice(col_range) for x in col_range])
            bootstraps.append(bootstrap)

        return(bootstraps)


    def emit_results(self, betas, rescaled_betas):
        """
        Output result report(s) for workflow run.
        """
        for k in range(self.n_tasks):
            task_workflow_obj = self.workflow_objs[k]
            output_dir = os.path.join(self.input_dir, self.output_dir, self.tasks_dir[k])
            os.makedirs(output_dir)
            task_workflow_obj.results_processor = ResultsProcessor(betas[k], rescaled_betas[k])
            task_workflow_obj.results_processor.summarize_network(output_dir, task_workflow_obj.gold_standard, task_workflow_obj.priors_data)
        # rank-combine hack -- assumes all priors are the same
        output_dir = os.path.join(self.input_dir, self.output_dir, 'rank-combined')
        os.makedirs(output_dir)
        task_workflow_obj.results_processor = ResultsProcessor([i for l in betas for i in l], [i for l in rescaled_betas for i in l])
        task_workflow_obj.results_processor.summarize_network(output_dir, task_workflow_obj.gold_standard, task_workflow_obj.priors_data)
