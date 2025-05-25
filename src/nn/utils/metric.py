class Metric():
    def __init__(self,name:str,fn,better_direction : str):
        self.name:str = name
        self.fn=fn
        assert better_direction in ["lower","higher"]
        self.better_direction = better_direction

    def __call__(self, *args, **kwds):
        return self.fn(*args,**kwds)