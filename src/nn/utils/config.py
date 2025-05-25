from functools import cached_property

class Config():
    def __init__(self,config_or_dict,name:str="",name_features:list|None=None,update_dict:dict|None=None):
        self.name = name
        if name_features is None:
            name_features = ['epochs']
        elif 'epochs' not in name_features:
            name_features = ['epochs'] + name_features
        self.name_features = name_features
        if config_or_dict is None:
            config_or_dict = Config.get_default_train_config()
        if isinstance(config_or_dict,Config):
            config_dict = config_or_dict.config_dict
        else:
            config_dict = config_or_dict
        if update_dict is not None:
            config_dict.update(update_dict)
        assert 'epochs' in config_dict
        self.config_dict = config_dict
    
    def __getitem__(self, key):
        return self.config_dict[key]
    
    def __contains__(self,key):
        return key in self.config_dict
    
    @staticmethod
    def get_default_train_config():
        return Config({
            'tt_split': 0.75, # Train-test split
            'bs': 8, # Training batch size
            'base_lr': 1*(10**-1), # Starting learning rate,
            'end_lr': 1*(10**-3), # Smallest lr you will converge to
            'epochs': 50, # epochs
            'warmup_epochs': 2 # number of warmup epochs
            }) 

    @cached_property
    def ident(self) -> str:
        return ('_'.join([self.name]+[str(self.config_dict[feature]) for feature in self.name_features])).lower()
    
    def update(self,other,name_merge="keep_base"):
        assert isinstance(other,Config) or isinstance(other,dict)
        assert name_merge in ["keep_base","keep_new","concat"]
        return Config(self.config_dict.update(other.config_dict if isinstance(other,Config) else other),self.name if name_merge == "keep_merge" or isinstance(other,dict) else (other.name if name_merge == "keep_new" else other.name+'_'+self.name))
