import torch

model = torch.load('./cache/nnet/CornerNet_Squeeze/CornerNet_Squeeze_500000.pkl')
model2 = torch.load('./cache/nnet/CornerNet_Squeeze_traffic_light_v1/CornerNet_Squeeze_traffic_light_0.pkl')




# print("-----------------------------Model1's state_dict------------------------------------:")
# for param_tensor in model:
#     print(param_tensor, "\t", model[param_tensor].size())


# print("-----------------------------Model2's state_dict------------------------------------:")
# for param_tensor in model2:
#     print(param_tensor, "\t", model[param_tensor].size())

# model['module.tl_heats.0.1.weight'] = model2['module.tl_heats.0.1.weight']
# model['module.tl_heats.0.1.bias'] = model2['module.tl_heats.0.1.bias']
# model['module.tl_heats.1.1.weight'] = model2['module.tl_heats.1.1.weight']
# model['module.tl_heats.1.1.bias'] = model2['module.tl_heats.1.1.bias']
# model['module.br_heats.0.1.weight'] = model2['module.br_heats.0.1.weight']
# model['module.br_heats.0.1.bias'] = model2['module.br_heats.0.1.bias']
# model['module.br_heats.1.1.weight'] = model2['module.br_heats.1.1.weight']
# model['module.br_heats.1.1.bias'] = model2['module.br_heats.1.1.bias']

# print("-----------------------------Model2's state_dict------------------------------------:")
# print(model2['module.tl_heats.0.1.weight'].size())
# print( model2['module.tl_heats.0.1.bias'].size())
# print(model2['module.tl_heats.1.1.weight'].size())
# print(model2['module.tl_heats.1.1.bias'].size())
# print(model2['module.br_heats.0.1.weight'].size())
# print(model2['module.br_heats.0.1.bias'].size())
# print(model2['module.br_heats.1.1.weight'].size())
# print(model2['module.br_heats.1.1.bias'].size())



# torch.save(model,'./cache/nnet/pretrained_one_category/CornerNet_Squeeze_traffic_light_pretrained_one.pkl')

                                 
module.tl_heats.0.1.weight       torch.Size([80, 256, 1, 1])                       
module.tl_heats.0.1.bias         torch.Size([80])
module.tl_heats.1.1.weight       torch.Size([80, 256, 1, 1])
module.tl_heats.1.1.bias         torch.Size([80])
module.br_heats.0.1.weight       torch.Size([80, 256, 1, 1])
module.br_heats.0.1.bias         torch.Size([80])
module.br_heats.1.1.weight       torch.Size([80, 256, 1, 1])
module.br_heats.1.1.bias         torch.Size([80])

# torch.Size([1, 256, 1, 1])
# torch.Size([1])
# torch.Size([1, 256, 1, 1])
# torch.Size([1])
# torch.Size([1, 256, 1, 1])
# torch.Size([1])
# torch.Size([1, 256, 1, 1])
# torch.Size([1])