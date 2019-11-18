'''
Script adapted from https://github.com/rapidsai/notebooks-contrib

Original authors: Matthew Jones, John Zedlewski
NVIDIA
'''
import os
import sys
import copy
import uuid
import json
import time
import socket
import argparse
import itertools
import threading
import subprocess

from azureml.core import Workspace, Experiment, Environment
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.train.estimator import Estimator, Mpi
from azureml.exceptions import ComputeTargetException
from azureml.core.runconfig import RunConfiguration
from azureml.core import ScriptRunConfig
from azureml.core.authentication import InteractiveLoginAuthentication

from nyctaxi_data import download_nyctaxi_data
from nyctaxi_data import upload_nyctaxi_data

def spinner():
    while True:
        for cursor in '-\\|/':
            sys.stdout.write(cursor)
            sys.stdout.flush()
            time.sleep(0.1)
            sys.stdout.write('\b')
        if done:
            return

def print_message(msg, length=80, filler='#', pre_post=''):
    print(f'{pre_post} {msg} {pre_post}'.center(length, filler))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="./config.json")
    parser.add_argument("--cluster_name", default="RAPIDS")
    parser.add_argument("--experiment_name", default="RAPIDS0100")
    parser.add_argument("--vm_size", default="Standard_ND12S")
    parser.add_argument("--node_count", default=1)
    parser.add_argument("--min_nodes", default=0)
    parser.add_argument("--max_nodes", default=1)
    parser.add_argument("--admin_username", default="rapids")
    parser.add_argument("--admin_user_password", default="rapids")
    parser.add_argument("--admin_user_ssh_key", default="./azureml_mlads.pub")
    parser.add_argument("--ssh_id", default="./azure_mlads")

    parser.add_argument("--jupyter_token", default=uuid.uuid1().hex)
    parser.add_argument("--local_notebook_port", default="8888")
    parser.add_argument("--local_dashboard_port", default="8787")
    parser.add_argument("--local_cuxfilter_port", default="8778")

    parser.add_argument("--download_nyctaxi_data", default=False)
    parser.add_argument("--upload_nyctaxi_data", default=False)
    parser.add_argument("--nyctaxi_years", default="2016")
    parser.add_argument("--nyctaxi_src_path", default=os.getcwd())
    parser.add_argument("--nyctaxi_dst_path", default="data")
    parser.add_argument("--timeout-minutes", type=int, default=120)

    args = parser.parse_args()

    ### READ THE CONFIG FILE
    print_message('READING CONFIG')
    with open(args.config, 'r') as f:
        config = json.loads(f.read())

    subscription_id = config["subscription_id"]
    resource_group  = config["resource_group"]
    workspace_name  = config["workspace_name"]
    interactive_auth = InteractiveLoginAuthentication(tenant_id='<YOUR-TENANT-ID-HERE>')

    ### CONNECT TO WORKSPACE
    print_message(f'CONNECTING TO WORKSPACE {workspace_name}')
    workspace = Workspace(
          workspace_name=workspace_name
        , subscription_id=subscription_id
        , resource_group=resource_group
        , auth=interactive_auth
    )

    ssh_key = open(
            os.path.expanduser(args.admin_user_ssh_key)
        ).read().strip()

    ### GET OR CREATE COMPUTE TARGET
    try:
        cluster = ComputeTarget(workspace=workspace,
                                name=args.cluster_name)
    
        print_message(f'FOUND COMPUTE TARGET: {args.cluster_name}')
    
    except ComputeTargetException:
        print_message(f'TARGET "{args.cluster_name}" NOT FOUND')
        print_message("CREATING", filler='-', pre_post='>')

        if os.path.exists(os.path.expanduser(args.admin_user_ssh_key)):

            provisioning_config = (
                AmlCompute
                .provisioning_configuration(
                      vm_size=args.vm_size
                    , min_nodes=args.min_nodes
                    , max_nodes=args.max_nodes
                    , idle_seconds_before_scaledown=120
                    , admin_username=args.admin_username
                    , admin_user_password=args.admin_user_password
                    , admin_user_ssh_key=ssh_key
                )
            )
            cluster = ComputeTarget.create(
                workspace
                , args.cluster_name
                , provisioning_config
            )

            print_message('WAITING FOR COMPLETION', filler='-', pre_post='>')

            cluster.wait_for_completion(show_output=True)
            print_message(f'CLUSTER "{args.cluster_name}" CREATED')

    years = [args.nyctaxi_years]
    datastore = workspace.get_default_datastore()

    if args.download_nyctaxi_data:
        print_message('DOWNLOADING DATA')
        download_nyctaxi_data(
            years
            , args.nyctaxi_src_path
        )

    if args.upload_nyctaxi_data:
        print_message('UPLOADING DATA TO STORAGE')
        upload_nyctaxi_data(
            workspace
            , datastore
            , os.path.join(args.nyctaxi_src_path, "nyctaxi")
            , os.path.join(args.nyctaxi_dst_path, "nyctaxi"))

    n_gpus_per_node = 2 ### WE'RE USING ND6

    print_message("DECLARING ESTIMATOR")
  
    estimator = Estimator(
          source_directory='../notebooks'
        , compute_target=cluster
        , entry_script='init_dask.py'
        , script_params={
              "--datastore"        : workspace.get_default_datastore()
            , "--n_gpus_per_node"  : str(n_gpus_per_node)
            , "--jupyter_token"    : str(args.jupyter_token)
        }
        , distributed_training=Mpi(process_count_per_node=1)
        , node_count=int(args.node_count)
        , use_gpu=True
        , custom_docker_image='todrabas/mlads_rapids:fall19'
        , user_managed=True
    )

    #### SET PROPER INTERPRETER
    estimator._estimator_config.environment.python.interpreter_path = '/opt/conda/envs/rapids/bin/python'
    print_message("STARTING EXPERIMENT")
    
    experiment = Experiment(
        workspace
        , args.experiment_name
    ).submit(estimator)
    
    print()
    print_message("WAITING FOR THE HEADNODE")
    print_message("NOTE: THIS MAY TAKE SEVERAL MINUTES", filler='!')
    print_message(f"TRACK PROGRESS HERE --->>> ", filler='%')
    print_message(experiment.get_portal_url(), filler='%')

    print()
    print_message("SPINNING UP THE DASK CLUSTER")
    
    rep = 0
    done = False
    prev_status = ""
    spinning_thread = threading.Thread(target=spinner)
    spinning_thread.start()
    start_time = time.time()
    timeout_sec = args.timeout_minutes * 60

    while not "headnode" in experiment.get_metrics():
        rep += 1
        time.sleep(5)
        status = experiment.get_status()

        if status != prev_status:
            print_message(f"STATUS: {status}")
            prev_status = status

        if status == "Failed":
            details = experiment.get_details()
            print_message("FAILED TO CREATE HEADNODE", filler='!')
            print(details)
            done = True
            raise ValueError("Failed to create head node")

        elapsed = (time.time() - start_time)

        if elapsed > timeout_sec:
            done = True
            raise AssertionError("Creating head node timed out after %5.2f min" %
                             (elapsed/60))

        continue

    done = True
    spinning_thread.join()

    headnode = experiment.get_metrics()["headnode"]

    print()
    print_message("HEADNODE READY")
    print_message(f"HEADNODE IP: {headnode}")

    print_message("SETTING UP PORT FORWARDING")
    
    cmd = ("ssh -vvv -o StrictHostKeyChecking=no -N" + \
           " -i {ssh_key}" + \
           " -L 0.0.0.0:{notebook_port}:{headnode}:8888" + \
           " -L 0.0.0.0:{dashboard_port}:{headnode}:8787" + \
           " -L 0.0.0.0:{cuxfilter_port}:{headnode}:8778" + \
           " {uname}@{ip} -p {port}"
        ).format(
              ssh_key=os.path.expanduser(args.ssh_id)
            , notebook_port=args.local_notebook_port
            , dashboard_port=args.local_dashboard_port
            , cuxfilter_port=args.local_cuxfilter_port
            , headnode=headnode
            , uname=args.admin_username
            , ip=cluster.list_nodes()[0]['publicIpAddress']
            , port=cluster.list_nodes()[0]['port']
        )
    
    print_message("EXECUTING: ")
    print_message(cmd, filler='.')
    
    portforward_log_name = "portforward_log.txt"

    try:
        ssh_key
    except:
        print_message(
            "WARNING: COULD NOT FIND A VALID SSH KEY AT {path}"
            .format(path=os.path.expanduser(args.admin_user_ssh_key))
            , filler='!'
        )
        print_message(
            "WARNING: WHEN PROMPTED FOR A PASSWORD, ENTER `{password}`"
            .format(password=args.admin_user_password)
            , filler='!'
        )
    
    print_message(
        "SENDING VERBOSE PORT-FOWARDING OUTPUT TO {}"
        .format(portforward_log_name)
    )
    
    print_message(
        "NAVIGATE TO THE MICROSOFT AZURE PORTAL WHERE THE EXPERIMENT IS RUNNING"
    )
    
    print_message(
        "WHEN TRACKED METRICS INCLUDE BOTH A `jupyter` AND `jupyter-token` ENTRIES"
    )
    
    print_message(
        "THE LAB ENVIRONMENT WILL BE ACCESSIBLE ON THIS MACHINE"
    )
    
    print_message(
        """TO ACCESS THE JUPYTER LAB ENVIRONMENT, POINT YOUR BROWSER TO:
               http://{ip}:{port}/?token={token} ..."""
        .format(
            ip=socket.gethostbyname(socket.gethostname()),
            port=args.local_notebook_port,
            token=args.jupyter_token
        )
        , filler='-'
        , pre_post='>'
    )
    
    print_message(
        "TO VIEW THE DASK DASHBOARD, POINT YOUR WEB-BROWSER TO http://{ip}:{port} ..."
        .format(
              ip=socket.gethostbyname(socket.gethostname())
            , port=args.local_dashboard_port
        )
        , filler='-'
        , pre_post='>'
    )

    print_message(
        "TO VIEW THE cuXfilter DASHBOARD, POINT YOUR WEB-BROWSER TO http://{ip}:{port} ..."
        .format(
              ip=socket.gethostbyname(socket.gethostname())
            , port=args.local_cuxfilter_port
        )
        , filler='-'
        , pre_post='>'
    )

    print("TO ACQUIRE THE PATH TO YOUR DEFAULT DATASTORE, INSPECT THE TRACKED METRICS FROM THE MICROSOFT AZURE PORTAL")
    print("CANCELLING THIS SCRIPT BY USING CONTROL-C WILL NOT DECOMMISSION THE COMPUTE RESOURCES")
    print("TO DECOMMISSION COMPUTE RESOURCES, NAVIGATE TO THE MICROSOFT AZURE PORTAL AND (1) CANCEL THE RUN, (2) DELETE THE COMPUTE ASSET")
    
    portforward_log = open("portforward_out_log.txt", 'w')
    portforward_proc = (
        subprocess
        .Popen(
            cmd.split()
            , universal_newlines=True
            , stdout=subprocess.PIPE
            , stderr=subprocess.STDOUT
        )
    )

    while True:
      portforward_out = portforward_proc.stdout.readline()
      if portforward_out == '' and portforward_proc.poll() is not None:
        portforward_log.close()
        break
      elif portforward_out:
        portforward_log.write(portforward_out)
        portforward_log.flush()