import numpy as np
from .sum_tree import Sum_Tree
from abc import ABC, abstractmethod

# [s, visual_s, a, r, s_, visual_s_, done] must be this format.


class ReplayBuffer(ABC):
    def __init__(self, batch_size, capacity):
        assert isinstance(batch_size, int) and batch_size >= 0, 'batch_size must be int and larger than 0'
        assert isinstance(capacity, int) and capacity >= 0, 'capacity must be int and larger than 0'
        self.batch_size = batch_size
        self.capacity = capacity
        self._size = 0

    @abstractmethod
    def sample(self) -> list:
        pass

    @abstractmethod
    def add(self, *args) -> None:
        pass

    def is_empty(self):
        return self._size == 0

    def update(self, *args) -> None:
        pass


class StagedExperienceMechanism(ReplayBuffer):
    def __init__(self, batch_size, capacity, n, cof, gamma, agents_num=12):
        super().__init__(batch_size, capacity)
        self._data_pointer = [i for i in range(capacity)]
        self.rec = {i for i in range(capacity)}
        self._buffer = np.empty(capacity, dtype=object)
        self.n = n
        self.cof = cof
        self.gamma = gamma
        self.queues = [[] for _ in range(agents_num)]
        self.n_queues = [[] for _ in range(agents_num)]
        self._mean = -1.e30
        self.fluct = 0.

    def add(self, *args):
        '''
        change [s, s],[a, a],[r, r] to [s, a, r],[s, a, r] and store every item in it.
        '''
        # [q.append(data) for q, data in zip(self.queues, zip(*args))]
        [self._per_store(i, list(data)) for i, data in enumerate(zip(*args))]

    def _per_store(self, i, data):
        '''
        data:
            0   s           -7
            1   visual_s    -6
            2   a           -5
            3   r           -4
            4   s_          -3
            5   visual_s_   -2
            6   done        -1
        '''
        q = self.n_queues[i]
        if len(q) == 0:  
            q.append(data)
            return
        if (q[-1][-3] != data[0]).any() or (q[-1][-2] != data[1]).any():   
            if len(q) == self.n:
                self.queues[i].append(q.pop(0))
            else:
                q.clear()
            q.append(data)
            return

        if len(q) == self.n: 
            self.queues[i].append(q.pop(0))
        _len = len(q)
        for j in range(_len):   
            q[j][3] += data[3] * (self.gamma ** (_len - j))
            q[j][4:] = data[4:]
        q.append(data)
        if data[-1]:
            while q:   
                self.queues[i].append(q.pop())

    def _store_op(self, data):
        if len(self._data_pointer) == 0:
            _idx = np.unique(np.random.randint(0, self.capacity, self.capacity // 10))
            self._data_pointer.extend(_idx)
            [self.rec.add(i) for i in _idx]
        idx = self._data_pointer.pop(0)
        self.rec.remove(idx)
        self._buffer[idx] = data
        self.update_rb_after_add()

    def mark(self, r, _mean, fluct):
        for i, q in enumerate(self.queues):
            [self._store_op(tuple(data) + (r[i],)) for data in q]
        [q.clear() for q in self.queues]
        self._mean = _mean
        self.fluct = fluct

    def sample(self):
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        idx = np.random.randint(0, self._size, n_sample)
        t = self._buffer[idx]
        s, visual_s, a, r, s_, visual_s_, done, qr = [np.asarray(e) for e in zip(*t)]
        qr = np.abs(qr - self._mean)
        if qr.max() == 0:
            p = qr
        else:
            p = (1 - qr / qr.max())   # [0, 1] 与当前阶段均值差别越大，越小概率被替换，保留差值较大的经验
        p = p ** (self.cof*self.fluct) - 0.5
        rand = np.random.rand()
        _idx = idx[np.where((p + rand) > 1.)[0]]
        for i in _idx:
            l = len(self.rec)
            self.rec.add(i)
            if len(self.rec) != l:
                self._data_pointer.append(i)
        return (s, visual_s, a, r, s_, visual_s_, done)

    def get_all(self):
        return [np.asarray(e) for e in zip(*self._buffer[:self._size])]

    def update_rb_after_add(self):
        if self._size < self.capacity:
            self._size += 1

    @property
    def is_full(self):
        return self._size == self.capacity

    @property
    def size(self):
        return self._size

    @property
    def is_lg_batch_size(self):
        return self._size > self.batch_size

    @property
    def show_rb(self):
        print('RB size: ', self._size)
        print('RB capacity: ', self.capacity)
        print(self._buffer[:, np.newaxis])


class ExperienceReplay(ReplayBuffer):
    def __init__(self, batch_size, capacity):
        super().__init__(batch_size, capacity)
        self._data_pointer = 0
        self._buffer = np.empty(capacity, dtype=object)

    def add(self, *args):
        '''
        change [s, s],[a, a],[r, r] to [s, a, r],[s, a, r] and store every item in it.
        '''
        [self._store_op(data) for data in zip(*args)]

    def _store_op(self, data):
        self._buffer[self._data_pointer] = data
        self.update_rb_after_add()

    def sample(self):
        '''
        change [[s, a, r],[s, a, r]] to [[s, s],[a, a],[r, r]]
        '''
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        t = np.random.choice(self._buffer[:self._size], size=n_sample, replace=False)
        return [np.asarray(e) for e in zip(*t)]

    def get_all(self):
        return [np.asarray(e) for e in zip(*self._buffer[:self._size])]

    def update_rb_after_add(self):
        self._data_pointer += 1
        if self._data_pointer >= self.capacity:  # replace when exceed the capacity
            self._data_pointer = 0
        if self._size < self.capacity:
            self._size += 1

    @property
    def is_full(self):
        return self._size == self.capacity

    @property
    def size(self):
        return self._size

    @property
    def is_lg_batch_size(self):
        return self._size > self.batch_size

    @property
    def show_rb(self):
        print('RB size: ', self._size)
        print('RB capacity: ', self.capacity)
        print(self._buffer[:, np.newaxis])


class PrioritizedExperienceReplay(ReplayBuffer):
    '''
    This PER will introduce some bias, 'cause when the experience with the minimum probability has been collected, the min_p that be updated may become inaccuracy.
    '''

    def __init__(self,
                 batch_size: int,
                 capacity: int,
                 max_train_step: int,
                 alpha: float,
                 beta: float,
                 epsilon: float,
                 global_v: bool):
        '''
        inputs:
            max_train_step: use for calculating the decay interval of beta
            alpha: control sampling rule, alpha -> 0 means uniform sampling, alpha -> 1 means complete td_error sampling
            beta: control importance sampling ratio, beta -> 0 means no IS, beta -> 1 means complete IS.
            epsilon: a small positive number that prevents td-error of 0 from never being replayed.
            global_v: whether using the global
        '''
        assert epsilon > 0, 'epsilon must larger than zero'
        super().__init__(batch_size, capacity)
        self.tree = Sum_Tree(capacity)
        self.alpha = alpha
        self.beta = self.init_beta = beta
        self.beta_interval = (1. - beta) / max_train_step
        self.epsilon = epsilon
        self.IS_w = 1   # weights of variables by using Importance Sampling
        self.global_v = global_v
        self.reset()

    def reset(self):
        self.tree.reset()
        super().reset()
        self.beta = self.init_beta
        self.min_p = sys.maxsize
        self.max_p = np.power(self.epsilon, self.alpha)

    def add(self, *args):
        '''
        input: [ss, visual_ss, as, rs, s_s, visual_s_s, dones]
        '''
        self.add_batch(list(zip(*args)))
        # [self._store_op(data) for data in zip(*args)]

    def _store_op(self, data):
        self.tree.add(self.max_p, data)
        if self._size < self.capacity:
            self._size += 1

    def add_batch(self, data):
        data = list(data)
        num = len(data)
        self.tree.add_batch(np.full(num, self.max_p), data)
        self._size = min(self._size + num, self.capacity)

    def apex_add_batch(self, td_error, *args):
        data = list(zip(*args))
        num = len(data)
        prios = np.power(np.abs(td_error) + self.epsilon, self.alpha)
        self.tree.add_batch(prios, data)
        self._size = min(self._size + num, self.capacity)

    def sample(self, return_index=False):
        '''
        output: weights, [ss, visual_ss, as, rs, s_s, visual_s_s, dones]
        '''
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        all_intervals = np.linspace(0, self.tree.total, n_sample + 1)
        ps = np.random.uniform(all_intervals[:-1], all_intervals[1:])
        idxs, data_indx, p, data = self.tree.get_batch_parallel(ps)
        self.last_indexs = idxs
        _min_p = self.min_p if self.global_v and self.min_p < sys.maxsize else p.min()
        self.IS_w = np.power(_min_p / p, self.beta)
        if return_index:
            return data, idxs
        else:
            return data

    def get_all(self, return_index=False):
        idxs, data_indx, p, data = self.tree.get_all()
        self.last_indexs = idxs
        _min_p = self.min_p if self.global_v and self.min_p < sys.maxsize else p.min()
        self.IS_w = np.power(_min_p / p, self.beta)
        if return_index:
            return data, idxs
        else:
            return data

    def get_all_exps(self):
        return self.tree.get_all_exps()

    @property
    def is_lg_batch_size(self):
        return self._size > self.batch_size

    def update(self,
               priority,
               index=None):
        '''
        input: priorities
        '''
        assert hasattr(priority, '__len__'), 'priority must have attribute of len()'
        idxs = index if index is not None else self.last_indexs
        assert len(priority) == len(idxs), 'length between priority and last_indexs must equal'
        self.beta = min(self.beta + self.beta_interval, 1.)
        priority = np.power(np.abs(priority) + self.epsilon, self.alpha)
        self.min_p = min(self.min_p, priority.min())
        self.max_p = max(self.max_p, priority.max())
        self.tree._updatetree_batch(idxs, priority)
        # [self.tree._updatetree(idx, p) for idx, p in zip(idxs, priority)]

    def get_IS_w(self):
        return self.IS_w

    @property
    def size(self):
        return self._size


class NStepWrapper:
    def __init__(self, buffer, gamma, n, agents_num):
        '''
        gamma: discount factor
        n: n step
        agents_num: batch experience
        '''
        self.buffer = buffer
        self.n = n
        self.gamma = gamma
        self.agents_num = agents_num
        self.queue = [[] for _ in range(agents_num)]

    def add(self, *args):
        '''
        change [s, s],[a, a],[r, r] to [s, a, r],[s, a, r] and store every item in it.
        '''
        [self._per_store(i, list(data)) for i, data in enumerate(zip(*args))]

    def _per_store(self, i, data):
        '''
        data:
            0   s           -7
            1   visual_s    -6
            2   a           -5
            3   r           -4
            4   s_          -3
            5   visual_s_   -2
            6   done        -1
        '''
        q = self.queue[i]
        if len(q) == 0: 
            q.append(data)
            return
        if (q[-1][-3] != data[0]).any() or (q[-1][-2] != data[1]).any():  
            if len(q) == self.n:
                self._store_op(q.pop(0))
            else:
                q.clear()
            q.append(data)
            return

        if len(q) == self.n: 
            self._store_op(q.pop(0))
        _len = len(q)
        for j in range(_len):  
            q[j][3] += data[3] * (self.gamma ** (_len - j))
            q[j][4:] = data[4:]
        q.append(data)
        if data[-1]: 
            while q:  
                self._store_op(q.pop())

    def __getattr__(self, name):
        return getattr(self.buffer, name)


class NStepExperienceReplay(NStepWrapper):
    '''
    Replay Buffer + NStep
    [s, visual_s, a, r, s_, visual_s_, done] must be this format.
    '''

    def __init__(self, batch_size, capacity, gamma, n, agents_num):
        super().__init__(
            buffer=ExperienceReplay(batch_size, capacity),
            gamma=gamma, n=n, agents_num=agents_num
        )


class NStepPrioritizedExperienceReplay(NStepWrapper):
    '''
    PER + NStep
    [s, visual_s, a, r, s_, visual_s_, done] must be this format.
    '''

    def __init__(self, batch_size, capacity, max_episode, alpha, beta, epsilon, global_v, gamma, n, agents_num):
        super().__init__(
            buffer=PrioritizedExperienceReplay(batch_size, capacity, max_episode, alpha, beta, epsilon, global_v),
            gamma=gamma, n=n, agents_num=agents_num
        )


class EpisodeExperienceReplay(ReplayBuffer):

    def __init__(self, batch_size, capacity, agents_num):
        super().__init__(batch_size, capacity)
        self.agents_num = agents_num
        self.queue = [[] for _ in range(agents_num)]
        self._data_pointer = 0
        self._buffer = np.empty(capacity, dtype=object)

    def add(self, *args):
        '''
        change [s, s],[a, a],[r, r] to [s, a, r],[s, a, r] and store every item in it.
        '''
        [self._per_store(i, list(data)) for i, data in enumerate(zip(*args))]

    def _per_store(self, i, data):
        '''
        data:
            0   s           -7
            1   visual_s    -6
            2   a           -5
            3   r           -4
            4   s_          -3
            5   visual_s_   -2
            6   done        -1
        '''
        q = self.queue[i]
        if len(q) == 0:
            q.append(data)
            return
        if (q[-1][-3] != data[0]).any() or (q[-1][-2] != data[1]).any():
            self._store_op(q.copy())
            q.clear()
            q.append(data)
            return
        if data[-1]:
            q.append(data)
            self._store_op(q.copy())
            q.clear()
            return
        q.append(data)

    def _store_op(self, data):
        self._buffer[self._data_pointer] = data
        self.update_rb_after_add()

    def update_rb_after_add(self):
        self._data_pointer += 1
        if self._data_pointer >= self.capacity:  # replace when exceed the capacity
            self._data_pointer = 0
        if self._size < self.capacity:
            self._size += 1

    def sample(self):
        '''
        [B, (s, a, r, s', d)] => [B, time_step, N]
        '''
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        t = np.random.choice(self._buffer[:self._size], size=n_sample, replace=False)
        s, visual_s, a, r, s_, visual_s_, done = [[] for _ in range(7)]
        for i in t:
            data = [d for d in zip(*i)]
            s.append(data[0])
            visual_s.append(data[1])
            a.append(data[2])
            r.append(data[3])
            s_.append(data[4])
            visual_s_.append(data[5])
            done.append(data[6])
        return map(np.asarray, [s, visual_s, a, r, s_, visual_s_, done])

    @property
    def is_full(self):
        return self._size == self.capacity

    @property
    def size(self):
        return self._size

    @property
    def is_lg_batch_size(self):
        return self._size > self.batch_size

    @property
    def show_rb(self):
        print('RB size: ', self._size)
        print('RB capacity: ', self.capacity)
        print(self._buffer[:, np.newaxis])
