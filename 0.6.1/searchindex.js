Search.setIndex({docnames:["api","api/pydra.engine","api/pydra.engine.audit","api/pydra.engine.boutiques","api/pydra.engine.core","api/pydra.engine.graph","api/pydra.engine.helpers","api/pydra.engine.helpers_file","api/pydra.engine.helpers_state","api/pydra.engine.specs","api/pydra.engine.state","api/pydra.engine.submitter","api/pydra.engine.task","api/pydra.engine.workers","api/pydra.mark","api/pydra.mark.functions","api/pydra.tasks","api/pydra.utils","api/pydra.utils.messenger","api/pydra.utils.profiler","changes","index"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,sphinx:56},filenames:["api.rst","api/pydra.engine.rst","api/pydra.engine.audit.rst","api/pydra.engine.boutiques.rst","api/pydra.engine.core.rst","api/pydra.engine.graph.rst","api/pydra.engine.helpers.rst","api/pydra.engine.helpers_file.rst","api/pydra.engine.helpers_state.rst","api/pydra.engine.specs.rst","api/pydra.engine.state.rst","api/pydra.engine.submitter.rst","api/pydra.engine.task.rst","api/pydra.engine.workers.rst","api/pydra.mark.rst","api/pydra.mark.functions.rst","api/pydra.tasks.rst","api/pydra.utils.rst","api/pydra.utils.messenger.rst","api/pydra.utils.profiler.rst","changes.rst","index.rst"],objects:{"":[[0,0,0,"-","pydra"]],"pydra.engine":[[1,2,1,"","AuditFlag"],[1,2,1,"","DockerTask"],[1,2,1,"","ShellCommandTask"],[1,2,1,"","Submitter"],[1,2,1,"","Workflow"],[2,0,0,"-","audit"],[3,0,0,"-","boutiques"],[4,0,0,"-","core"],[5,0,0,"-","graph"],[6,0,0,"-","helpers"],[7,0,0,"-","helpers_file"],[8,0,0,"-","helpers_state"],[9,0,0,"-","specs"],[10,0,0,"-","state"],[11,0,0,"-","submitter"],[12,0,0,"-","task"],[13,0,0,"-","workers"]],"pydra.engine.AuditFlag":[[1,3,1,"","ALL"],[1,3,1,"","NONE"],[1,3,1,"","PROV"],[1,3,1,"","RESOURCE"]],"pydra.engine.DockerTask":[[1,4,1,"","container_args"],[1,3,1,"","init"]],"pydra.engine.ShellCommandTask":[[1,4,1,"","cmdline"],[1,4,1,"","command_args"]],"pydra.engine.Submitter":[[1,5,1,"","close"],[1,5,1,"","submit"],[1,5,1,"","submit_workflow"]],"pydra.engine.Workflow":[[1,5,1,"","add"],[1,4,1,"","checksum"],[1,5,1,"","create_connections"],[1,4,1,"","done_all_tasks"],[1,4,1,"","graph_sorted"],[1,4,1,"","nodes"],[1,5,1,"","set_output"]],"pydra.engine.audit":[[2,2,1,"","Audit"]],"pydra.engine.audit.Audit":[[2,5,1,"","audit_check"],[2,5,1,"","audit_message"],[2,5,1,"","finalize_audit"],[2,5,1,"","monitor"],[2,5,1,"","start_audit"]],"pydra.engine.boutiques":[[3,2,1,"","BoshTask"]],"pydra.engine.core":[[4,2,1,"","TaskBase"],[4,2,1,"","Workflow"],[4,1,1,"","is_lazy"],[4,1,1,"","is_task"],[4,1,1,"","is_workflow"]],"pydra.engine.core.TaskBase":[[4,3,1,"","audit_flags"],[4,4,1,"","cache_dir"],[4,4,1,"","cache_locations"],[4,4,1,"","can_resume"],[4,4,1,"","checksum"],[4,5,1,"","checksum_states"],[4,5,1,"","combine"],[4,4,1,"","done"],[4,5,1,"","get_input_el"],[4,5,1,"","help"],[4,4,1,"","output_dir"],[4,4,1,"","output_names"],[4,5,1,"","pickle_task"],[4,5,1,"","result"],[4,5,1,"","set_state"],[4,5,1,"","split"],[4,4,1,"","version"]],"pydra.engine.core.Workflow":[[4,5,1,"","add"],[4,4,1,"","checksum"],[4,5,1,"","create_connections"],[4,4,1,"","done_all_tasks"],[4,4,1,"","graph_sorted"],[4,4,1,"","nodes"],[4,5,1,"","set_output"]],"pydra.engine.graph":[[5,2,1,"","DiGraph"]],"pydra.engine.graph.DiGraph":[[5,5,1,"","add_edges"],[5,5,1,"","add_nodes"],[5,5,1,"","calculate_max_paths"],[5,5,1,"","copy"],[5,4,1,"","edges"],[5,4,1,"","edges_names"],[5,4,1,"","nodes"],[5,4,1,"","nodes_names_map"],[5,5,1,"","remove_nodes"],[5,5,1,"","remove_nodes_connections"],[5,4,1,"","sorted_nodes"],[5,4,1,"","sorted_nodes_names"],[5,5,1,"","sorting"]],"pydra.engine.helpers":[[6,1,1,"","copyfile_workflow"],[6,1,1,"","create_checksum"],[6,1,1,"","ensure_list"],[6,1,1,"","execute"],[6,1,1,"","gather_runtime_info"],[6,1,1,"","get_available_cpus"],[6,1,1,"","get_open_loop"],[6,1,1,"","hash_function"],[6,1,1,"","hash_value"],[6,1,1,"","load_and_run"],[6,1,1,"","load_and_run_async"],[6,1,1,"","load_result"],[6,1,1,"","load_task"],[6,1,1,"","make_klass"],[6,1,1,"","output_from_inputfields"],[6,1,1,"","output_names_from_inputfields"],[6,1,1,"","print_help"],[6,1,1,"","read_and_display"],[6,1,1,"","read_and_display_async"],[6,1,1,"","read_stream_and_display"],[6,1,1,"","record_error"],[6,1,1,"","save"],[6,1,1,"","task_hash"]],"pydra.engine.helpers_file":[[7,1,1,"","copyfile"],[7,1,1,"","copyfile_input"],[7,1,1,"","copyfiles"],[7,1,1,"","ensure_list"],[7,1,1,"","get_related_files"],[7,1,1,"","hash_dir"],[7,1,1,"","hash_file"],[7,1,1,"","is_container"],[7,1,1,"","is_existing_file"],[7,1,1,"","is_local_file"],[7,1,1,"","on_cifs"],[7,6,1,"","related_filetype_sets"],[7,1,1,"","split_filename"],[7,1,1,"","template_update"]],"pydra.engine.helpers_state":[[8,7,1,"","PydraStateError"],[8,1,1,"","add_name_combiner"],[8,1,1,"","add_name_splitter"],[8,1,1,"","combine_final_groups"],[8,1,1,"","converter_groups_to_input"],[8,1,1,"","flatten"],[8,1,1,"","input_shape"],[8,1,1,"","inputs_types_to_dict"],[8,1,1,"","iter_splits"],[8,1,1,"","map_splits"],[8,1,1,"","remove_inp_from_splitter_rpn"],[8,1,1,"","rpn2splitter"],[8,1,1,"","splits"],[8,1,1,"","splits_groups"],[8,1,1,"","splitter2rpn"]],"pydra.engine.specs":[[9,2,1,"","BaseSpec"],[9,2,1,"","ContainerSpec"],[9,2,1,"","Directory"],[9,2,1,"","DockerSpec"],[9,2,1,"","File"],[9,2,1,"","LazyField"],[9,2,1,"","Result"],[9,2,1,"","Runtime"],[9,2,1,"","RuntimeSpec"],[9,2,1,"","ShellOutSpec"],[9,2,1,"","ShellSpec"],[9,2,1,"","SingularitySpec"],[9,2,1,"","SpecInfo"],[9,2,1,"","TaskHook"],[9,1,1,"","attr_fields"],[9,1,1,"","donothing"],[9,1,1,"","path_to_string"]],"pydra.engine.specs.BaseSpec":[[9,5,1,"","check_fields_input_spec"],[9,5,1,"","check_metadata"],[9,5,1,"","collect_additional_outputs"],[9,5,1,"","copyfile_input"],[9,4,1,"","hash"],[9,5,1,"","retrieve_values"],[9,5,1,"","template_update"]],"pydra.engine.specs.ContainerSpec":[[9,3,1,"","bindings"],[9,3,1,"","container"],[9,3,1,"","container_xargs"],[9,3,1,"","image"]],"pydra.engine.specs.DockerSpec":[[9,3,1,"","container"]],"pydra.engine.specs.LazyField":[[9,5,1,"","get_value"]],"pydra.engine.specs.Result":[[9,3,1,"","errored"],[9,3,1,"","output"],[9,3,1,"","runtime"]],"pydra.engine.specs.Runtime":[[9,3,1,"","cpu_peak_percent"],[9,3,1,"","rss_peak_gb"],[9,3,1,"","vms_peak_gb"]],"pydra.engine.specs.RuntimeSpec":[[9,3,1,"","container"],[9,3,1,"","network"],[9,3,1,"","outdir"]],"pydra.engine.specs.ShellOutSpec":[[9,5,1,"","collect_additional_outputs"],[9,3,1,"","return_code"],[9,3,1,"","stderr"],[9,3,1,"","stdout"]],"pydra.engine.specs.ShellSpec":[[9,3,1,"","args"],[9,5,1,"","check_fields_input_spec"],[9,5,1,"","check_metadata"],[9,3,1,"","executable"],[9,5,1,"","retrieve_values"]],"pydra.engine.specs.SingularitySpec":[[9,3,1,"","container"]],"pydra.engine.specs.SpecInfo":[[9,3,1,"","bases"],[9,3,1,"","fields"],[9,3,1,"","name"]],"pydra.engine.specs.TaskHook":[[9,3,1,"","post_run"],[9,3,1,"","post_run_task"],[9,3,1,"","pre_run"],[9,3,1,"","pre_run_task"],[9,5,1,"","reset"]],"pydra.engine.state":[[10,2,1,"","State"]],"pydra.engine.state.State":[[10,4,1,"id0","combiner"],[10,5,1,"","combiner_validation"],[10,3,1,"","final_combined_ind_mapping"],[10,3,1,"","group_for_inputs"],[10,3,1,"","group_for_inputs_final"],[10,3,1,"","groups_stack_final"],[10,4,1,"id1","inner_inputs"],[10,3,1,"","inputs_ind"],[10,4,1,"","left_combiner"],[10,4,1,"","left_combiner_all"],[10,4,1,"","left_splitter"],[10,4,1,"","left_splitter_rpn"],[10,4,1,"","left_splitter_rpn_compact"],[10,3,1,"","name"],[10,4,1,"id2","other_states"],[10,5,1,"","prepare_inputs"],[10,5,1,"","prepare_states"],[10,5,1,"","prepare_states_combined_ind"],[10,5,1,"","prepare_states_ind"],[10,5,1,"","prepare_states_val"],[10,4,1,"","right_combiner"],[10,4,1,"","right_combiner_all"],[10,4,1,"","right_splitter"],[10,4,1,"","right_splitter_rpn"],[10,5,1,"","set_input_groups"],[10,4,1,"id3","splitter"],[10,4,1,"id4","splitter_final"],[10,4,1,"id5","splitter_rpn"],[10,4,1,"id6","splitter_rpn_compact"],[10,4,1,"","splitter_rpn_final"],[10,5,1,"","splitter_validation"],[10,3,1,"","states_ind"],[10,3,1,"","states_val"],[10,5,1,"","update_connections"]],"pydra.engine.submitter":[[11,2,1,"","Submitter"],[11,1,1,"","get_runnable_tasks"],[11,1,1,"","is_runnable"]],"pydra.engine.submitter.Submitter":[[11,5,1,"","close"],[11,5,1,"","submit"],[11,5,1,"","submit_workflow"]],"pydra.engine.task":[[12,2,1,"","ContainerTask"],[12,2,1,"","DockerTask"],[12,2,1,"","FunctionTask"],[12,2,1,"","ShellCommandTask"],[12,2,1,"","SingularityTask"]],"pydra.engine.task.ContainerTask":[[12,5,1,"","bind_paths"],[12,5,1,"","binds"],[12,5,1,"","container_check"]],"pydra.engine.task.DockerTask":[[12,4,1,"","container_args"],[12,3,1,"","init"]],"pydra.engine.task.ShellCommandTask":[[12,4,1,"","cmdline"],[12,4,1,"","command_args"]],"pydra.engine.task.SingularityTask":[[12,4,1,"","container_args"],[12,3,1,"","init"]],"pydra.engine.workers":[[13,2,1,"","ConcurrentFuturesWorker"],[13,2,1,"","DaskWorker"],[13,2,1,"","DistributedWorker"],[13,2,1,"","SerialPool"],[13,2,1,"","SerialWorker"],[13,2,1,"","SlurmWorker"],[13,2,1,"","Worker"]],"pydra.engine.workers.ConcurrentFuturesWorker":[[13,5,1,"","close"],[13,5,1,"","exec_as_coro"],[13,5,1,"","run_el"]],"pydra.engine.workers.DaskWorker":[[13,5,1,"","close"],[13,5,1,"","exec_dask"],[13,5,1,"","run_el"]],"pydra.engine.workers.DistributedWorker":[[13,5,1,"","fetch_finished"],[13,3,1,"","max_jobs"]],"pydra.engine.workers.SerialPool":[[13,5,1,"","done"],[13,5,1,"","result"],[13,5,1,"","submit"]],"pydra.engine.workers.SerialWorker":[[13,5,1,"","close"],[13,5,1,"","run_el"]],"pydra.engine.workers.SlurmWorker":[[13,5,1,"","run_el"]],"pydra.engine.workers.Worker":[[13,5,1,"","close"],[13,5,1,"","fetch_finished"],[13,5,1,"","run_el"]],"pydra.mark":[[15,0,0,"-","functions"]],"pydra.mark.functions":[[15,1,1,"","annotate"],[15,1,1,"","task"]],"pydra.utils":[[18,0,0,"-","messenger"],[19,0,0,"-","profiler"]],"pydra.utils.messenger":[[18,2,1,"","AuditFlag"],[18,2,1,"","FileMessenger"],[18,2,1,"","Messenger"],[18,2,1,"","PrintMessenger"],[18,2,1,"","RemoteRESTMessenger"],[18,2,1,"","RuntimeHooks"],[18,1,1,"","collect_messages"],[18,1,1,"","gen_uuid"],[18,1,1,"","make_message"],[18,1,1,"","now"],[18,1,1,"","send_message"]],"pydra.utils.messenger.AuditFlag":[[18,3,1,"","ALL"],[18,3,1,"","NONE"],[18,3,1,"","PROV"],[18,3,1,"","RESOURCE"]],"pydra.utils.messenger.FileMessenger":[[18,5,1,"","send"]],"pydra.utils.messenger.Messenger":[[18,5,1,"","send"]],"pydra.utils.messenger.PrintMessenger":[[18,5,1,"","send"]],"pydra.utils.messenger.RemoteRESTMessenger":[[18,5,1,"","send"]],"pydra.utils.messenger.RuntimeHooks":[[18,3,1,"","resource_monitor_post_stop"],[18,3,1,"","resource_monitor_pre_start"],[18,3,1,"","task_execute_post_exit"],[18,3,1,"","task_execute_pre_entry"],[18,3,1,"","task_run_entry"],[18,3,1,"","task_run_exit"]],"pydra.utils.profiler":[[19,2,1,"","ResourceMonitor"],[19,1,1,"","get_max_resources_used"],[19,1,1,"","get_system_total_memory_gb"],[19,1,1,"","log_nodes_cb"]],"pydra.utils.profiler.ResourceMonitor":[[19,4,1,"","fname"],[19,5,1,"","run"],[19,5,1,"","stop"]],pydra:[[0,1,1,"","check_latest_version"],[1,0,0,"-","engine"],[14,0,0,"-","mark"],[16,0,0,"-","tasks"],[17,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","attribute","Python attribute"],"4":["py","property","Python property"],"5":["py","method","Python method"],"6":["py","data","Python data"],"7":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:attribute","4":"py:property","5":"py:method","6":"py:data","7":"py:exception"},terms:{"0":[0,1,4,6,8,15,18,21],"1":[1,7,8,13,18,21],"2":[0,1,15,18,21],"3":[1,18,21],"4":[18,21],"5":[6,18,19,21],"6":[18,21],"8192":7,"abstract":18,"case":[2,6],"class":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,18,19,20],"default":[1,7,9,11],"do":[1,11,18],"enum":[1,18],"final":[4,8,10,13],"float":[9,15,19],"function":[0,7,8,9,14,19,20],"import":[10,15],"int":[4,6,9,15,19],"new":[5,6,13,19],"return":[1,4,5,6,7,8,9,11,12,13,15,19,20],"true":[1,4,6,7,8,10,11,18],A:[1,4,5,6,9,10,13,18,19],As:7,If:[1,6,7,11],It:[0,10],On:7,That:[1,4],The:[0,1,4,6,7,8,9,16,18],To:16,_connect:20,_na:10,_node_wip:5,abc:6,abl:7,about:10,accept:[4,19,20],access:7,actual:[1,12],ad:[1,4,5,8,20],add:[1,4,5,20],add_edg:5,add_name_combin:8,add_name_splitt:8,add_nod:5,addit:[8,9],address:10,administr:6,afil:7,afni:7,after:[7,10],aggreg:[1,11],algorithm:8,all:[1,4,5,7,8,9,10,11,18,20],all_:20,allow:[18,20],alreadi:[6,7,20],also:[5,7,9,10,20],an:[4,6,7,9,13,20],analyz:7,ani:[1,4,5,9,11,19],annot:15,api:[13,20,21],append:18,appli:15,applic:21,ar:[1,4,5,7,8,9,10,19,20],arg:[1,3,9,12],argument:[1,9,12],arrai:20,arriv:6,associ:10,assum:[7,8],async:[1,6,11,13],asyncio:[6,13],attr:[13,20],attr_field:9,attr_typ:9,attribut:20,audit:[0,1,4,12,18],audit_check:2,audit_flag:[1,2,4,12],audit_messag:2,auditflag:[1,4,12,18],avail:[1,4,6,9,10,20],await:[1,11,13],ax:[8,10],axi:8,azur:20,b:6,backend:[1,11],base:[1,2,3,4,5,7,8,9,10,11,12,13,18,19],basespec:[1,4,9,12],basic:[4,9],been:4,befor:[1,4,11],begin:7,behavior:7,being:19,between:[5,10],big:20,bind:[9,12,20],bind_path:12,binds_path:12,bool:[1,4,6,7,8,9,11,18],boshtask:[3,20],both:4,bound:[9,12],boutiqu:[0,1,20],brik:7,build:18,built:7,c:6,cach:[4,6],cache_dir:[1,4,12],cache_loc:[1,4,6,12,20],calcul:[1,4,5,6,10,20],calculate_max_path:5,call:19,callabl:[9,12],callback:[12,19],can:[1,4,7,8,10,20],can_hardlink:7,can_resum:4,can_symlink:7,captur:6,certain:19,cf:[1,11],chang:20,check:[1,4,7,9,11,20],check_fields_input_spec:9,check_latest_vers:0,check_metadata:9,checkpoint:4,checksum:[1,4,6,20],checksum_st:4,checkum:[1,4],chunk_len:7,ci:20,cif:7,cli:[1,12],close:[1,6,11,13],cmd:6,cmdline:[1,12,20],code:[9,20],codecov:20,collect:[4,6,9,16],collect_additional_output:9,collect_messag:18,collected_path:18,combin:[4,6,8,10,20],combine_final_group:8,combiner_valid:10,command:[1,3,6,7,9,12],command_arg:[1,12,20],compact:[10,18],complet:[1,11],composit:[1,4],comput:[1,4,7,9],concept:20,concurr:[7,13,20],concurrentfutureswork:13,conda:12,config:12,conflict:20,connect:[1,4,5,8,10],consid:19,consumpt:9,cont_dim:[8,10,20],contain:[1,5,6,7,8,9,10,12,20],container:[1,9,12],container_arg:[1,12,20],container_check:12,container_info:[1,3,12],container_typ:12,container_xarg:9,containerspec:9,containertask:[1,12],content:7,context:18,contribut:20,convers:8,convert:[8,9],converter_groups_to_input:8,copi:[5,6,9,12,20],copy_related_fil:7,copyfil:7,copyfile_input:[7,9],copyfile_workflow:6,core:[0,1,12,19,20],coroutin:[1,6,11,13],correct:10,could:10,cpath:12,cpu:[6,9,20],cpu_peak_perc:9,creat:[1,2,4,5,6,7,8,10,16],create_checksum:6,create_connect:[1,4],create_new:7,create_pyscript:20,creation:20,crypto:7,cryptograph:7,cur_depth:8,current:[5,6,10,20],custom:[8,20],cwl:9,dask:[13,20],daskwork:13,data:[5,6,7,9],dataclass:[9,20],dataflow:20,decor:15,def:15,defin:7,depend:[8,10],descriptor:[3,20],design:9,dest:7,destin:7,determin:[2,10],develop:2,dict:[4,7,8,10,12,18],dictionari:[5,8,10,19],diff:7,differ:7,digraph:5,dimens:8,dimension:20,direct:5,directori:[1,2,4,6,7,9,12,20],dirpath:7,disabl:7,discuss:6,displai:6,distribut:[1,11,13,20],distributedwork:13,dmtcp:12,docker:[1,7,9,12],dockerrequir:9,dockerspec:9,dockertask:[1,12,20],document:20,doe:[5,7,19],doesn:[1,4],done:[1,4,13],done_all_task:[1,4],donoth:9,driver:7,due:8,duplic:5,dure:19,e:[7,9,10,18],each:[8,10],ecosystem:0,edg:5,edges_nam:5,edit:20,either:[5,7],element:[1,4,8,10,12,20],elements_to_remove_comb:10,elemntari:4,empti:20,enabl:[2,20],end:[2,8,19],endpoint:18,engin:[0,19,20],ensure_list:[6,7],entrypoint:[1,11],env:12,environ:[6,12],envvarrequir:9,eof:6,error:[1,6,7,8,9,11,19,20],error_path:6,especi:10,etc:[1,4],etelementri:20,evalu:10,event:6,eventloop:6,everi:[7,8,10],exampl:[6,7,15],except:[7,8,20],exec_as_coro:13,exec_dask:13,execut:[1,6,9,11,12,13,19],executor:13,exist:[1,4],exit:9,experiment:[13,20],expos:7,ext:7,extend:[1,12],extens:7,extern:12,extract:6,f:7,fact:[1,4],fail:20,failur:[7,20],fals:[1,4,6,7,8,9,11,12,13,19],far:19,featur:7,fetch_finish:13,field:[4,6,8,9,10,20],file:[4,6,9,12,18,19,20],filelist:7,filemesseng:18,filenam:[7,19],filenotfound:7,filenotfounderror:7,filesystem:[4,7,12],filetyp:7,final_combined_ind_map:10,finalize_audit:2,find:7,finish:13,first:0,fix:20,flag:[1,2,4,18,20],flatten:8,fli:20,fname:[6,7,19],folder:9,fork:16,form:[0,8],format:[18,20],found:7,fragment:12,framework:6,french:7,frequenc:19,from:[1,4,5,6,7,8,9,10,11,12,20],fulfil:9,full:[4,7,10,20],func:[12,15],functiontask:[12,15,20],futur:[1,11,13,20],g:[7,9,10],gather:18,gather_runtime_info:6,gb:19,gen_uuid:18,gener:[4,6,8,9,18],get:[1,4,5,6,8,9,10,12,13,18,19,20],get_available_cpu:[6,20],get_input_el:4,get_max_resources_us:19,get_open_loop:6,get_related_fil:7,get_runnable_task:11,get_system_total_memory_gb:19,get_valu:9,given:[6,7,9,19],graph:[0,1,4,11],graph_sort:[1,4],greater:7,group:[8,10],group_for_input:[8,10],group_for_inputs_fin:10,groups_stack:8,groups_stack_fin:10,gz:7,ha:[1,4,7,12],handl:[2,11],hard:7,hardlink:7,hash:[6,7,9,12,20],hash_dir:[7,20],hash_fil:7,hash_funct:6,hash_valu:6,have:[1,4,7,8,10],hdr:7,head:7,help:4,helper:[0,1],helpers_fil:[0,1],helpers_st:[0,1],hide_displai:6,high:19,histori:5,hlpst:10,home:7,hook:[9,18],host:7,how:[8,20],i:[9,10,18],id:19,identifi:[6,18],ignor:7,ignore_hidden_dir:7,ignore_hidden_fil:7,imag:9,img:7,imit:13,implement:[7,9,12,13,20],impos:8,improv:20,includ:[7,10],include_this_fil:7,inconsist:7,ind:[4,6,12],index:[4,21],indic:[4,8,10,20],info:19,inform:[2,6,10],inherit:[4,9],init:[1,12],initi:[1,11,20],initialworkdirrequir:9,inlinejavascriptrequir:9,inlinescriptrequir:9,inner:[8,10,20],inner_input:[8,10],inp:[8,10],input:[1,4,6,7,8,9,10,20],input_shap:8,input_spec:[1,4,9,12],inputs_ind:10,inputs_to_remov:8,inputs_types_to_dict:8,insert:5,insid:2,instal:20,instanc:[1,4,11],instead:[7,20],integ:19,intenum:18,interfac:[6,10,13,21],intern:[12,13,19],interpret:7,interv:19,intput:8,invok:9,is_contain:7,is_existing_fil:7,is_lazi:4,is_local_fil:7,is_runn:11,is_task:4,is_workflow:4,isol:12,issu:20,item:7,iter:8,iter_split:8,its:6,job:13,join:0,json:[12,19],keep:[2,7,9,10,19],kei:8,kept:20,kwarg:[1,3,4,6,9,11,12,13,18],lazi:[4,9],lazyfield:9,ld_op:18,lead:7,left_combin:10,left_combiner_al:10,left_splitt:10,left_splitter_rpn:10,left_splitter_rpn_compact:10,length:[7,8],librari:21,light:10,like:4,limit:[13,20],line:[1,6,9,12],linearli:13,link:[5,7],list:[1,4,5,6,7,8,9,10,12,20],load:[6,20],load_and_run:[6,20],load_and_run_async:6,load_result:6,load_task:6,local:12,locat:4,log:[18,19],log_nodes_cb:19,logdir:19,logger:19,look:[4,6],loop:[1,6,11,13],lpath:12,lzout:20,make_class:20,make_klass:6,make_messag:18,manag:[16,20],mandatori:9,map:[0,5,10],map_copyfil:7,map_split:8,mark:[0,5],mat:7,max_depth:8,max_job:13,maximum:[5,13],mb:19,md:20,mean:[1,4],medatada:9,mem_mb:19,memori:[9,19,20],mention:20,merg:10,messag:[2,18],message_path:18,messeng:[0,1,2,4,12,17],messenger_arg:[1,2,4,12],metadata:[6,9],meth:12,method:[5,7,20],minshal:7,mode:12,modul:[0,1,14,17,21],monitor:[1,2,12,18,19],more:[4,20],mostli:8,mount:[7,9,12],move:20,much:20,multil:20,multipl:[4,20],mutat:8,n_proc:[6,13],name:[1,4,5,6,7,8,9,10,12,20],name_prefix:6,namespac:[16,20],need:[1,4,5,9,10],nest:[8,20],network:9,neurodock:12,neuroimag:7,new_combin:10,new_edg:5,new_nod:5,new_other_st:10,newfil:7,next:20,niceman:12,nidm:18,nifti:7,nii:7,nipyp:[0,7,19],node:[1,4,5,8,9,10,12,19,20],nodes_names_map:5,none:[1,2,3,4,5,6,7,8,9,10,11,12,13,18,19,20],notat:[8,10],note:21,notebook:20,noth:[9,20],now:[18,20],num_thread:19,number:[6,8,10,13,20],numpi:20,o:9,obj:[4,6,7,11,18],object:[1,2,4,5,6,7,9,10,11,13,18],odir:2,on_cif:7,one:[4,8,13],onli:[1,7,10,12,18],open:18,openssl_sha256:7,oper:[0,7,10],opt:12,option:[1,4,8,9,12],order:[6,10],origin:[7,8,12],originalfil:7,os:[2,6,7,9,20],osx:20,other:[7,8,20],other_st:[8,10],otherwis:[7,19],outcom:4,outdir:9,outer:10,output:[1,2,4,6,7,9,10,18,20],output_dir:[4,7,9],output_file_templ:6,output_from_inputfield:6,output_nam:4,output_names_from_inputfield:6,output_spec:[1,4,6,9,12],over:[4,7,10],overwrit:4,packag:[0,20],page:21,pair:[5,7,8],parallel:13,paramet:[1,2,4,5,6,7,8,11,13,18,19],parameter:4,parametr:4,pars:[9,11],part:[7,10],partial:10,particular:[1,2,4,9],pass:20,path:[4,5,6,7,9,20],path_to_str:9,pathlib:[6,9],pathlik:[2,6,9],peak:9,pend:13,perform:[8,19],physic:9,pickl:[4,6,20],pickle_task:4,pid:19,pipelin:19,pleas:16,plugin:[1,6,11],point:[9,12,18],polish:8,poll_delai:13,pool:13,port:7,posix:7,post_run:9,post_run_task:9,pre_run:9,pre_run_task:9,preced:7,predecessor:5,prepar:10,prepare_input:10,prepare_st:10,prepare_states_combined_ind:10,prepare_states_ind:10,prepare_states_v:10,prescrib:8,present:7,presort:5,previou:[4,8,10],previous:[1,5,11],print:[4,6,18],print_help:6,printmesseng:18,prioriti:6,process:[4,6,9,10,12,18,19,20],profil:[0,17],programm:21,promis:9,promot:15,propagate_rerun:[1,4,20],proper:[6,20],properli:20,properti:[1,4,5,9,10,12,19],prov:[1,18],proven:[1,2,12,18],provid:[8,9],prune:5,pth:7,py2:7,py:[12,13],pydra:[0,20],pydrastateerror:[8,20],pyfunc:19,python:[12,13],rais:[7,20],raise_notfound:7,ram:[9,19],re:5,read:[6,7,12,20],read_and_displai:6,read_and_display_async:6,read_stream_and_displai:6,readm:20,recent:7,record:[2,19],record_error:6,recreat:4,recurr:8,recurs:[6,7],redirect:18,reduc:10,refer:5,refin:9,regard:9,regular:7,relat:7,related_filetype_set:7,releas:21,relev:10,remot:[12,18],remoterestmesseng:18,remov:[1,5,7,8,10,11,20],remove_inp_from_splitter_rpn:8,remove_nod:5,remove_node_connect:5,remove_nodes_connect:5,reorgan:20,replac:4,repo:20,report:[1,4,20],repres:[9,10],represent:[1,4],requir:[4,8,9,20],rerun:[1,4,6,11,12,13,20],reserv:16,reset:9,resourc:[1,2,18,19],resource_monitor_post_stop:18,resource_monitor_pre_start:18,resourcemonitor:19,resourcerequir:9,rest:18,restart:4,restor:6,result:[1,2,4,6,7,9,10,13,20],resum:12,retriev:4,retrieve_valu:9,return_cod:9,return_input:4,returnhelp:4,revers:8,rewrit:0,right:[10,20],right_combin:10,right_combiner_al:10,right_splitt:10,right_splitter_rpn:10,rpn2splitter:8,rpn:[8,10],rss_peak_gb:9,run:[1,4,5,6,7,9,10,11,13,19,20],run_el:13,runnabl:[1,5,11,13],runtim:[6,9],runtimehook:18,runtimespec:9,s:[8,10,13,20],same:[5,7,10],save:6,sbatch_arg:13,scalar:10,schemadefrequir:9,search:21,see:5,self:[5,10],send:[1,2,11,13,18],send_messag:18,sent:2,separ:[16,20],serialpool:13,serialwork:13,server:12,set:[1,4,6,7,9,11,13,19,20],set_input_group:10,set_output:[1,4],set_stat:4,shape:8,share:7,shell:[1,3,9,12],shellcommandrequir:9,shellcommandtask:[1,3,12,20],shelloutspec:9,shellspec:9,shelltask:[9,20],should:[7,8,10],simpl:[5,13,20],simplifi:20,singl:[7,10,20],singular:[9,12,20],singularityspec:9,singularitytask:[12,20],slurm:[13,20],slurmwork:13,small:20,so:[8,19,20],softwarerequir:9,some:20,sort:[1,4,5],sorted_nod:5,sorted_nodes_nam:5,sourc:4,spec:[0,1,4,6,7,12],specif:[1,4,6,8,9,10,12,19,20],specifi:[7,8,10,12,20],specinfo:[9,12],sphinx:20,split:[4,7,8,10],split_filenam:7,split_it:8,splits_group:8,splitter2rpn:8,splitter:[4,8,10,20],splitter_fin:10,splitter_rpn:[8,10],splitter_rpn_compact:10,splitter_rpn_fin:10,splitter_valid:10,spm:7,squar:15,stack:[8,10],stackoverflow:6,standalon:20,standard:[6,8,9,18],start:[2,5,19],start_audit:2,state:[0,1,4,8,11,12,18,20],state_field:[8,10],state_index:[4,9],states_ind:10,states_v:10,statist:19,statu:19,stderr:9,stdout:9,step:4,sto:20,stop:19,store:4,str:[1,4,6,7,9,10],stream:6,string:[6,7,9,19],strip:6,structur:[1,4,5,9],subject:7,submiss:[1,11,13],submit:[1,11,12,13],submit_workflow:[1,11],submitt:[0,1,6,20],submodul:0,subpackag:21,support:[5,6,7,20],symbol:7,symlink:7,system:[6,7,12,13,19,20],t:[1,4],take:7,task:[0,1,2,3,4,5,6,9,10,11,13,15,20],task_execute_post_exit:18,task_execute_pre_entri:18,task_hash:6,task_path:6,task_pkl:6,task_run_entri:18,task_run_exit:18,taskbas:[1,4,6,11,12],taskhook:9,templat:[7,9,16],template_upd:[7,9],test:[13,20],test_rerun:20,text:7,than:7,thei:[5,6],them:[1,5,6,11],thi:[4,5,6,7,8,9,13,19],thread:19,through:7,time:[8,9],timestamp:18,to_job:20,todo:[1,2,4,6],togeth:[4,7],total:[6,19],tp:6,track:[1,2,9,10,12,18,19],translat:8,travers:7,travi:20,truncat:18,tupl:[6,7,9,10,20],tuple2list:6,tutori:20,txt:20,type:[1,4,6,7,8,9,10,11,13,19,20],under:7,union:[1,4,9],uniqu:[1,4,6,18],unless:6,unlink:7,until:[2,6,13],unwrap:[8,10],updat:[7,9,10,15,20],update_connect:10,us:[1,4,6,7,8,10,12,13,15,20],use_hardlink:7,user:8,util:[0,1,2,4,12],val:[4,8],valid:[10,20],valu:[1,4,6,7,8,9,10,18,19,20],variabl:[8,12,20],verbos:20,version:4,virtual:9,visit:6,vms_peak_gb:9,wait:[1,11],watermark:19,we:7,were:[7,20],wf:[6,9,20],wf_path:6,what:4,whatev:6,when:[5,7,8,9,10,18,20],where:[4,6],whether:[2,4,7,13],which:[4,5,6,7,8,10],window:[7,20],within:11,without:[5,7],work:[6,12,20],worker:[0,1,11,20],workflow:[0,1,4,5,6,11,15,20],wrap:[1,12],wrapper:13,write:[1,4,6,12],written:[4,7],x:9,xor:9},titles:["Library API (application programmer interface)","pydra.engine package","pydra.engine.audit module","pydra.engine.boutiques module","pydra.engine.core module","pydra.engine.graph module","pydra.engine.helpers module","pydra.engine.helpers_file module","pydra.engine.helpers_state module","pydra.engine.specs module","pydra.engine.state module","pydra.engine.submitter module","pydra.engine.task module","pydra.engine.workers module","pydra.mark package","pydra.mark.functions module","pydra.tasks package","pydra.utils package","pydra.utils.messenger module","pydra.utils.profiler module","Release Notes","Welcome to Pydra: A simple dataflow engine with scalable semantics\u2019s documentation!"],titleterms:{"0":20,"1":20,"2":20,"3":20,"4":20,"5":20,"6":20,"function":15,"new":7,A:21,api:0,applic:0,audit:2,boutiqu:3,content:21,copi:7,core:4,dataflow:21,document:21,engin:[1,2,3,4,5,6,7,8,9,10,11,12,13,21],exist:7,file:7,graph:5,helper:6,helpers_fil:7,helpers_st:8,indic:21,interfac:0,librari:0,mark:[14,15],messeng:18,modul:[2,3,4,5,6,7,8,9,10,11,12,13,15,18,19],note:[12,20],option:7,packag:[1,14,16,17],profil:19,programm:0,pydra:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21],releas:20,s:21,scalabl:21,semant:21,simpl:21,spec:9,state:10,submitt:11,submodul:[1,14,17],subpackag:0,tabl:21,task:[12,16],util:[17,18,19],welcom:21,worker:13}})