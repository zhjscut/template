import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import numpy as np
import pprint
from collections import OrderedDict
import copy


class AverageMeter():
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count += n
        self.avg = self.sum / self.count
        

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lamda = np.random.beta(alpha, alpha)
    else: 
        lamda = 1
    batch_size = x.size(0)
    indices = torch.randperm(batch_size)
    x_mixed = lamda * x + (1 - lamda) * x[indices, :]
    y_a, y_b = y, y[indices]
    return x_mixed, y_a, y_b, lamda

def mixup_criterion(criterion, pred, y_a, y_b, lamda):
    """
    Cross-Entropy loss. Because the mixed label has two non-zero elements,
    so the formula of loss also contain two items
    """
    return lamda * criterion(pred, y_a) + (1 - lamda) * criterion(pred, y_b)

class My_criterion(nn.Module):
    def __init__(self):
        super(My_criterion, self).__init__()
        self.softmax = nn.Softmax(1)
        
    def forward(self, outputs_st_s, outputs_st_t):
        outputs_st_s, outputs_st_t = self.softmax(outputs_st_s), self.softmax(outputs_st_t) # [batch_size, 2*num_classes_s] respectively
        half = outputs_st_s.size(1) // 2
        outputs_Cs_s, outputs_Ct_t = outputs_st_s[:, 0:half], outputs_st_t[:, half:]    
        loss_domain_st = -torch.log(outputs_Cs_s.sum(1)).mean() - torch.log(outputs_Ct_t.sum(1)).mean()
        
        return loss_domain_st

class My_criterion_Em(nn.Module):
    def __init__(self):
        super(My_criterion_Em, self).__init__()
        self.softmax = nn.Softmax(1)
        
    def forward(self, outputs_st):
        outputs_st = self.softmax(outputs_st)
        half = outputs_st.size(1) // 2
        outputs_sum = outputs_st[:, 0:half] + outputs_st[:, half:]
        return -(outputs_sum * torch.log(outputs_sum)).sum(1).mean()
    
    
    
def check_params(model1, model2=''):
    """
    Single model version (model2 has passed nothing):
        Check parameters in models, especially used in weight-share model to verify that
        the weight is shared correctly.
        Shared weight have the same summation, while independent one are not the same.
    Double model version:
        Check parameters in two weight-share models, to verify that the weight is shared correctly.
        Shared weight have the same summation, while independent one are not the same.
    """    
    if model2 == '':
        for name, param in model1.named_parameters():
            print(name, param.abs().mean())
#             print(name, param.abs().sum())
            # print(name, param.size(), param.sum())
        return 

    # simple implementation, hard to make comparation if the model is a bit complex
    # for name, param in model1.named_parameters():
    #     print(name, param.sum())
    # for name, param in model1.named_parameters():
    #     print(name, param.sum())
    
    # complex implememtation
    params1 = model1.named_parameters()
    params2 = model2.named_parameters()        
    m1_next, m2_next = True, True
    while m1_next or m2_next:
        try:
            name, param = params1.__next__()
#             print(name, param.abs().sum())
            print(name, param.abs().mean())
        except StopIteration:
            # print('stop1')
            m1_next = False
        try:
            name, param = params2.__next__()
#             print(name, param.abs().sum())
            print(name, param.abs().mean())
        except StopIteration:
            # print('stop2')
            m2_next = False           

    
def check_model(input_size, model):
    """
    Use a random noise input tensor to flow through the entire network, to test
    if the demension of all modules is matching    
    Beacuse if there exist fc layer in the model, the H and W of test input is unique, and 
    the in_channels of fc layer is hard to obtain, so give up the idea to automatically generate
    a test input, but use a external input instead.
    Parameters
    ----------
    input_size: list or tuple
        the size of input tensor, which is [C, H, W]
    model: a subclass of nn.Module
        the model need to be checked
    Returns
    -------
    summary: OrderedDict, optional(disabled now)
        Contain informations of base modules of the model, each module has info about 
        input_shape, output_shape, num_classes, and trainable
    Usage
    -----
    model = Model(args)
    check_model(input_size, model)
    """
    def get_output_size(summary_dict, output):
        if isinstance(output, tuple): # if the model has more than one output, "output" here will be a tuple
            for i in range(len(output)): 
                summary_dict[i] = OrderedDict()
                summary_dict[i] = get_output_size(summary_dict[i],output[i])
        else:
            summary_dict['output_shape'] = list(output.size())
        return summary_dict

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            # print(module.__class__)
            module_idx = len(summary)
            # print(classes_idx)
            allow_base_modules_only = True # it control whether create summary for those middle modules
            if allow_base_modules_only:
                base_classes = ['Linear', 'Conv2d', 'Flatten', 'ReLU', 'PReLU'] # 有待添加，随着网络的变化而变化
                if class_name not in base_classes:
                    return 
            class_idx = classes_idx.get(class_name)
            if class_idx == None:
                class_idx = 0
                classes_idx[class_name] = 1
            else:
                classes_idx[class_name] += 1
            # print(type(input), type(output))
            m_key = '%s-%i (%i)' % (class_name, class_idx+1, module_idx+1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size()) # input is a tuple having an tensor element
            summary[m_key] = get_output_size(summary[m_key], output)
        
            params = 0
            if hasattr(module, 'weight'):
                params += int(torch.prod(torch.LongTensor(list(module.weight.size()))))
                if module.weight.requires_grad:
                    summary[m_key]['trainable'] = True
                else:
                    summary[m_key]['trainable'] = False
            #if hasattr(module, 'bias'):
            #  params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
    
            summary[m_key]['num_params'] = params # not take bias into consideration
            pprint.pprint({m_key: summary[m_key]})
            # print(m_key, ":", summary[m_key]) # print info of each module in one line
        if not isinstance(module, nn.Sequential) and \
            not isinstance(module, nn.ModuleList) and \
            not (module == model): # make sure "module" is a base module, such conv, fc and so on
            hooks.append(module.register_forward_hook(hook)) # hooks is used to record added hook for removing them later
  
    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(1,*in_size)) for in_size in input_size]
    else:
        x = Variable(torch.rand(1,*input_size)) # 1 is batch_size dimension to adapt the model's structure

    # create properties
    summary = OrderedDict()
    classes_idx = {}
    hooks = []
    # register hook
    model.apply(register_hook) # 递归地去给每个网络组件挂挂钩（不只是conv, fc这种底层组件，上面的Sequential组件也会被访问到）
    # make a forward pass
    output = model(x)    
    # output, _ = model(x)
    # remove these hooks
    for h in hooks:
        h.remove()
    # pprint.pprint(summary)
    print('output size:', output.size())
    print('Check done.')
    # return summary        
    
def check_grad(model1, model2=''):
    """
    Single model version (model2 has passed nothing):
        Check parameters in models, especially used in weight-share model to verify that
        the weight is shared correctly.
        Shared weight have the same summation, while independent one are not the same.
    Double model version:
        Check parameters in two weight-share models, to verify that the weight is shared correctly.
        Shared weight have the same summation, while independent one are not the same.
    """    
    minimum = 10086
    maximum = 0
    simple_sum = 0
    simple_count = 0
    if model2 == '':
        for name, param in model1.named_parameters():
#             print(name, param.grad.abs().sum())
#             print(name, param.grad.abs().mean())
            if param.grad.abs().mean() < minimum:
                minimum = param.grad.abs().mean()
            elif param.grad.abs().mean() > maximum:
                maximum = param.grad.abs().mean()
            simple_sum += param.grad.abs().mean()
            simple_count += 1
        simple_mean = simple_sum / simple_count
#         print('min:', minimum, 'max:', maximum, 'mean:', simple_mean)
        return simple_mean

    # simple implementation, hard to make comparation if the model is a bit complex
    # for name, param in model1.named_parameters():
    #     print(name, param.sum())
    # for name, param in model1.named_parameters():
    #     print(name, param.sum())
    
    # complex implememtation
    params1 = model1.named_parameters()
    params2 = model2.named_parameters()        
    m1_next, m2_next = True, True
    while m1_next or m2_next:
        try:
            name, param = params1.__next__()
#             print(name, param.grad.abs().sum())
            print(name, param.grad.abs().mean())            
        except StopIteration:
            # print('stop1')
            m1_next = False
        try:
            name, param = params2.__next__()
            print(name, param.sum())
        except StopIteration:
            # print('stop2')
            m2_next = False         

            
def analyze_output(outputs, labels='', need_softmax=False, others=''):
    """
    Obtain the mininum, maximum and medium number in softmax output of correct label in a batch
    Parameters
    ----------
    outputs: Tensor or Variable
        the output tensor, with size [batch_size, num_classes]
    labels: Tensor or Variable
        the label of a batch, with size [batch_size]
    need_softmax: bool, optional
        Whether need to do softmax on outputs (default: False)
    Returns
    -------
    minimum: scalar
    maximum: scalar
    medium: scalar
    Usage
    -----
    outputs = model(inputs)
    minimum, maximum, medium = analyze_output(outputs, labels)
    """
    if need_softmax == True:
        softmax = nn.Softmax(1)
        outputs = softmax(outputs)
    if others=='SymNet':
        softmax = nn.Softmax(1)
        outputs_tmp = softmax(outputs) # [batch_size, 2*num_classes]
        half = outputs_tmp.size(1) // 2
        outputs = outputs_tmp[:, 0:half] + outputs_tmp[:, half:] # [batch_size, num_classes]
        
    if labels == '': # for unsupervised learning
        return outputs.min(), outputs.max(), outputs.contiguous().view(-1).median()
    else:
        indices = torch.LongTensor(list(range(outputs.size(0))))
        correct_outputs = outputs[indices, labels] # [batch_size]
        # there is a error if not use contiguous()
    #     return correct_outputs.min(), correct_outputs.max(), correct_outputs.median()
        return correct_outputs.min(), correct_outputs.max(), correct_outputs.median(), outputs.min(), outputs.max(), outputs.contiguous().view(-1).median()   
    
    
def drop_msecond(time_):
    """Drop the m second string in the datetime string
    Parameters
    ----------
    time_: datetime.datetime, like "datetime.datetime(2015, 4, 7, 4, 30, 3, 628556)"
    
    Output
    ------
    time_wo_msecond: string without m second part, like "2015-04-07 04:30:03"
    """
    return str(time_).split('.')[0]

def time_delta2str(time_delta):
    """Transform datetime.timedelta variable to needed string
    Parameters
    ----------
    time_delta: datetime.timedelta, like "datetime.timedelta(0, 767, 125697)"
    
    Output
    ------
    time_str: string in the format we need, like
    """
    # str(time_delta) is like "0:12:47.125697"
    h, m, s = str(time_delta).split('.')[0].split(':')
    return '{}h{}m{}s'.format(h, m, s)