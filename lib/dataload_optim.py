import torch

class _RepeatSampler(object):

    def __init__(self, sampler):
        self.sampler = sampler
    
    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class PersistentDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        print('persistent dataloader')
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class SoftwarePipeline(object):
    
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.stream = None
    
    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        if self.stream is None:
            self.stream = torch.cuda.Stream()
        first = True
        for next_input, next_target in self.dataloader:
            with torch.cuda.stream(self.stream):
                next_target = next_target.cuda(non_blocking=True)
                if isinstance(next_input, list):
                    for i in range(len(next_input)):
                        next_input[i] = next_input[i].cuda(non_blocking=True)
                else:
                    next_input = next_input.cuda(non_blocking=True)
            if not first:
                yield input, target
            else:
                first = False
            torch.cuda.current_stream().wait_stream(self.stream)
            input = next_input
            target = next_target
        yield input, target 
            