from time import time 
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
status = MPI.Status()
from tasks import OptimizeTask, TaskStatus, TaskTags

class Scheduler:
    def __init__(self, config):
        self.config = config
        
        if mpi_size <= 1:
            raise Exception("PROSPECT currently only supports running with MPI and with 2 processes or more.")
        
        samples = []
        if self.config['profile']['dimension'] == '1d':
            if self.config['profile']['sampling_strategy']['type'] == 'manual':
                from sampling import ManualSampling
                samples = ManualSampling(self.config['profile']['sampling_strategy'])
            else:
                raise NotImplementedError('Only manual sampling is implemented currently.')
        else:
            raise NotImplementedError('Only 1d profiles are implemented currently.')
        
        self.tasks = {}
        self.tasks['not_started'] = {}
        for sample in samples:
            new_task = OptimizeTask(config, sample)
            self.tasks['not_started'][new_task.id] = new_task
        
        self.tasks['in_progress'] = {}
        self.tasks['finished']    = {}
    
    def delegate(self):
        # Delegate only; do no work
        while len(self.tasks['not_started']) > 0:
            message = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source, tag = status.Get_source(), status.Get_tag()
            if tag == TaskTags.READY:
                new_task_id = min([task.id for task in self.tasks['not_started'].values()]) # get this by acquisition
                new_task = self.tasks['not_started'].pop(new_task_id)
                comm.isend(new_task, dest=source, tag=TaskTags.START)
                # Can also store start time of task for printing later...
                new_task.assigned_worker = source
                self.tasks['in_progress'][new_task.id] = new_task
            elif tag == TaskTags.DONE:
                assert(message.status == TaskStatus.FINISHED)
                self.tasks['in_progress'].pop(message.id)
                self.tasks['finished'][message.id] = message
        else:
            # Tell everyone to stop, or queue analysis tasks
            pass
    
    def delegate_and_work(self):
        # Delegate and work simultaneously
        raise NotImplementedError('')
    
    def print_tasks_status(self):
        print(f"Current tasks:")
        print(f"Not started: {len(self.tasks['not_started'])}")
        print(f"In progess: {len(self.tasks['in_progess'])}")
        print(f"Finished: {len(self.tasks['finished'])}")
        print(f"\nTasks currently running:")
        for task in self.tasks['in_progess']:
            print(f"Type: {type(task)}. Worker id: {task.assigned_worker}. Uptime: TIMER")

class Worker:
    def __init__(self):
        pass
        
    def work(self):
        while True:
            comm.isend(None, dest=0, tag=TaskTags.READY)
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            if tag == TaskTags.START:
                task.run()
                comm.isend(task, dest=0, tag=TaskTags.DONE)
            elif tag == TaskTags.EXIT:
                break