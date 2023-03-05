import time
from mpi4py import MPI
import pickle 
from prospect.tasks import OptimizeTask, MCMCTask, AnalyseMCMCTask, TaskStatus, TaskTags

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
status = MPI.Status()

class Scheduler:
    def __init__(self, config, state):
        self.config = config
        self.next_dump_time = config['snapshot_interval']
        if state:
            # Resume from snapshot
            self.tasks = state 
        else:
            # Start from config file 
            self.tasks = {}
            self.tasks['not_started'] = {}
            self.tasks['in_progress'] = {}
            self.tasks['finished']    = {}

            if config['jobtype'] == 'profile':
                task_list = self.initialize_profile()
            elif config['jobtype'] == 'mcmc':
                task_list = self.initialize_mcmc()
            for new_task in task_list:
                self.tasks['not_started'][new_task.id] = new_task

    def delegate(self):
        workers_working = mpi_size - 1
        while len(self.tasks['not_started']) + len(self.tasks['in_progress']) > 0:
            if time.time() > self.next_dump_time:
                self.dump_snapshot()
                self.next_dump_time += self.config['snapshot_interval']

            # blocking check for workers sending something; keeps blocking the worker that sends 
            # BUG: Gets stuck from only reacting to the probe of one worker
            for idx_worker in range(mpi_size):
                #if comm.probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status): 
                if comm.iprobe(source=idx_worker, tag=MPI.ANY_TAG, status=status): 
                    source, tag = status.Get_source(), status.Get_tag()
                    print(f"Scheduler received probe of tag {tag} from worker {source}")
                    if tag == TaskTags.READY:
                        new_task = self.get_new_task()
                        if new_task:
                            print("Scheduler waiting with task ready...")
                            message = comm.recv(source=source, tag=tag, status=status) # unblock the ready worker
                            comm.isend(new_task, dest=source, tag=TaskTags.START)
                            print(f"Scheduler sent task to {source}...")

                            # Can also store start time of task for printing later...
                            new_task.assigned_worker = source
                            self.tasks['in_progress'][new_task.id] = new_task
                    elif tag == TaskTags.DONE:
                        print("Scheduler waiting...")
                        print(source, "within elif")
                        message = comm.recv(source=source, tag=tag, status=status)
                        assert(message.status == TaskStatus.FINISHED)
                        self.tasks['in_progress'].pop(message.id)
                        self.tasks['finished'][message.id] = message
                    else:
                        raise Exception('Scheduler received unknown tag from a worker.')
        else:
            # Tell everyone to stop
            while workers_working > 0:
                print(f"{workers_working=}")
                message = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                source, tag = status.Get_source(), status.Get_tag()
                if tag == TaskTags.READY:
                    comm.isend(None, dest=source, tag=TaskTags.EXIT)
                    workers_working -= 1
                    print(f"Scheduler sent exit task to {source}. {workers_working=}")
                else:
                    raise Exception('Scheduler received a non-ready call when all tasks should have been finished.')
    
    def get_new_task(self):
        for task_id, task in self.tasks['not_started'].items():
            requirements_fulfilled = True
            for required_id in task.required_task_ids:
                if required_id not in self.tasks['finished'].keys():
                    print(f"Triggered requirement = False at id={required_id}")
                    print(self.tasks['finished'].keys())
                    requirements_fulfilled = False
            if requirements_fulfilled:
                new_task = self.tasks['not_started'].pop(task_id)
                # BUG: APPARENTLY THIS DOESN'T LOAD PROPERLY WHEN SENDING NEW_TASK TO WORKER?
                """
                    Traceback (most recent call last):
                File "/Users/au566942/Documents/phd/projects/profile/prospect/prospect/prospect.py", line 48, in <module>
                    worker.work()
                File "/Users/au566942/Documents/phd/projects/profile/prospect/prospect/communication.py", line 149, in work
                    task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                File "mpi4py/MPI/Comm.pyx", line 1438, in mpi4py.MPI.Comm.recv
                File "mpi4py/MPI/msgpickle.pxi", line 341, in mpi4py.MPI.PyMPI_recv
                File "mpi4py/MPI/msgpickle.pxi", line 306, in mpi4py.MPI.PyMPI_recv_match
                File "mpi4py/MPI/msgpickle.pxi", line 152, in mpi4py.MPI.pickle_load
                File "mpi4py/MPI/msgpickle.pxi", line 141, in mpi4py.MPI.cloads
                _pickle.UnpicklingError: invalid load key, '\x00'.
                """
                new_task.load([self.tasks['finished'][required_id] for required_id in new_task.required_task_ids])
                return new_task
            else:
                print(task.required_task_ids)
        else:
            #raise Exception('Scheduler could not find a new task to start.')
            return False

    def dump_snapshot(self):
        with open(f"{self.config['output_dir']}/state.pkl", 'wb') as state_file:
            pickle.dump(self.tasks, state_file)
        with open(f"{self.config['output_dir']}/status.txt","w") as status_file:
            status_file.write("Current tasks:\n")
            status_file.write(f"Not started: {len(self.tasks['not_started'])}\n")
            status_file.write(f"In progress: {len(self.tasks['in_progress'])}\n")
            status_file.write(f"Finished: {len(self.tasks['finished'])}\n")

            status_file.write(f"\nNot started:\n")
            for task in self.tasks['not_started'].values():
                status_file.write(f"ID: {task.id}\tType: {type(task)}.\n")

            status_file.write(f"\nIn progress:\n")
            for task in self.tasks['in_progress'].values():
                status_file.write(f"ID: {task.id}\tType: {type(task)}\tWorker assigned: {task.assigned_worker}\tUptime: TIMER\n")
            
            status_file.write(f"\nFinished:\n")
            for task in self.tasks['finished'].values():
                status_file.write(f"ID: {task.id}\tType: {type(task)}\tTime taken: TIMER\n")

    def initialize_profile(self): # this shouldn't be a member of scheduler
        samples = []
        if self.config['profile']['dimension'] == '1d':
            if self.config['profile']['sampling_strategy']['type'] == 'manual':
                from sampling import ManualSampling
                samples = ManualSampling(self.config['profile']['sampling_strategy'])
            else:
                raise NotImplementedError('Only manual sampling is implemented currently.')
        else:
            raise NotImplementedError('Only 1d profiles are implemented currently.')
        task_list = []
        for sample in samples:
            task_list.append(OptimizeTask(self.config, sample))
        return task_list
    
    def initialize_mcmc(self): # this shouldn't be a member of scheduler
        task_list = []
        for idx_chain in range(self.config['mcmc']['N_chains']):
            task_list.append(MCMCTask(self.config))
        task_list.append(AnalyseMCMCTask(self.config, [task.id for task in task_list]))
        return task_list

class Worker:
    def __init__(self):
        pass
        
    def work(self):
        while True:
            comm.isend(None, dest=0, tag=TaskTags.READY)
            print(f"Worker {mpi_rank} waiting to receive...")
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            print(f"Worker {mpi_rank} received task w. tag {tag}")
            if tag == TaskTags.START:
              #  print(type(task))
                task.run()
                comm.isend(task, dest=0, tag=TaskTags.DONE)
            elif tag == TaskTags.EXIT:
                break

class Serial:
    # Run when running PROSPECT without MPI
    def __init__(self, config, state):
        self.config = config 
        self.scheduler = Scheduler(config, state)

    def run(self):
        for idx_task in range(len(self.scheduler.tasks['not_started'])):
            task = self.scheduler.tasks['not_started'].pop(idx_task)
            print(f"Running task of type {type(task)}...")
            task.run()
            self.scheduler.tasks['finished'][task.id] = task
            self.scheduler.dump_snapshot()

