import torch
import torch.nn as nn 

class Activation(torch.nn.Module):
    def __init__(self,act=None) -> None:
        super().__init__()
        self._act = act
        
        upper_act_names = nn.modules.activation.__dict__.keys()
        lower_act_names = [act.lower() for act in upper_act_names]
        act_dict = dict(zip(lower_act_names, upper_act_names))

        if act is not None:
            if act in act_dict.keys():
                act_name = act_dict[act]
                self.act_func = eval("nn.modules.activation.{}()".format(
                    act_name))
            else:
                raise KeyError("{} does not exist in the current {}".format(
                    act, act_dict.keys()))

    def forward(self, x):
        if self._act is not None:
            return self.act_func(x)
        else:
            return x

class Add(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self,x,y):
        return torch.add(x,y)