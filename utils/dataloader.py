from concurrent.futures import ThreadPoolExecutor
import random
import torch

class DataLoader():
    def __init__(self, dataset, batch_size=1, num_workers=2, shuffle=False, drop_last=True, pin_memory=True):
        self.num_workers = num_workers
        self.dataset = dataset
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.index = 0
        
        # Shuffle indices. They also are re-shuffled every time the dataloader 
        # iterated all the dataset elements
        if self.shuffle:
            self.indices = random.sample(range(len(self.dataset)), len(self.dataset))
        else:
            self.indices = list(range(len(self.dataset)))
            
        # Dataloader length
        self.len = len(self.dataset)
        if self.batch_size != None:
            self.len = self.len//self.batch_size
        if not self.drop_last:
            self.len += 1
            
        # Start reading first batch
        if self.batch_size != None:
            indices = self.indices[:self.batch_size]
        else:
            indices = self.indices[:1]
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            self.futures = [executor.submit(self.dataset.__getitem__, i) for i in indices]
            
        
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.len
    
    def __next__(self):
        # Get current batch
        items = [f.result() for f in self.futures]
        items = [[items[i][j] for i in range(len(items))] for j in range(len(items[0]))]
        samples = [[torch.tensor(x).unsqueeze(0) if self.batch_size != None else torch.tensor(x) for x in items[i]] for i in range(len(items))]
        current_batched = [torch.cat(samples[i], dim=0) for i in range(len(items))]
        if self.pin_memory:
            for i in range(len(current_batched)):
                current_batched[i] = current_batched[i].pin_memory()
        
        # Check if the end of dataset was reached 
        if self.index == self.len:
            self.index = 0
            if self.shuffle:
                self.indices = random.sample(range(len(self.dataset)), len(self.dataset))
            raise StopIteration
        
        else:
            self.index += 1
            index_next = 0 if self.index == self.len else self.index
                
            # Start reading next batch
            batch_size = 1 if self.batch_size == None else self.batch_size
            indices = self.indices[index_next*batch_size : (index_next+1)*batch_size]
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                self.futures = [executor.submit(self.dataset.__getitem__, i) for i in indices]
            
            # Return current batch
            return current_batched
        
        
        
        
        