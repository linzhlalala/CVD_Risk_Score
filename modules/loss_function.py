from multiprocessing import reduction
import torch
import torch.nn as nn
import torch.nn.functional as F

# class FocalLoss(nn.Module):
#     "Non weighted version of Focal Loss"    
#     def __init__(self, alpha=.25, gamma=2):
#         super(FocalLoss, self).__init__()        
#         self.alpha = alpha
#         self.gamma = gamma
            
#     def forward(self, inputs, targets):
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')        
#         targets = targets.type(torch.long)        
#         at =  targets*(1-2*self.alpha) + self.alpha
#         pt = torch.exp(-BCE_loss)        
#         #print(at.shape,pt.shape, BCE_loss.shape)
#         F_loss = at*(1-pt)**self.gamma * BCE_loss        
#         return F_loss.mean()

class Mix_Loss(nn.Module):
    def __init__(self):
        super(Mix_Loss, self).__init__()
        # self.focal_loss = FocalLoss()

    def mt_supervised(self, pred, target, divide = 5):
        # The categorical changed to ordinal
        loss_temp = []
        
        c_p = pred
        c_y = target
        # ordinal can also be done in binary mode
        #print(c_p.shape,c_y.shape)
        #loss_temp.append(weights[idx]*F.binary_cross_entropy_with_logits(c_p,c_y))
        loss_temp.append(F.binary_cross_entropy_with_logits(c_p,c_y))
        loss = sum(loss_temp)
        return loss
    
    def forward(self, pred, target, divide = 5):
        # supervised loss
        pred = pred.view(-1,2,350)
        loss_s1 = self.mt_supervised(pred[:,0,:],target)
        loss_s2 = self.mt_supervised(pred[:,1,:],target)        
        #print(loss_s1, loss_s2)
        loss = loss_s1 + loss_s2
        return loss

if __name__ == '__main__':
    x = torch.rand(36,434)
    y = torch.ones(18,434)
    criterion = Mix_Loss()
    print(criterion(x,y))


    
    # def mt_supervised_weight(self, pred, target, divide = 5):
        
    #     # The categorical
    #     loss_temp = []
    #     preds = [pred[:,:100], pred[:,100:200], pred[:,200:300], pred[:,300:400],pred[:,400:431],pred[:,431],pred[:,432],pred[:,433]]
    #     for idx, column in enumerate(preds):
    #         c_p = column
    #         c_y = target[:,idx]
    #         if idx < divide:
    #             temp = weights[idx]*F.cross_entropy(c_p,c_y.long(),reduction = 'none')
    #             if idx == 0:
    #                 penal = torch.exp(6*(c_y/100 + torch.full(c_y.shape,0.2).to(c_y)))
    #                 temp = penal*temp
    #         else:
    #             temp = weights[idx]*F.binary_cross_entropy_with_logits(c_p,c_y,reduction = 'none')
    #         loss_temp.append(temp)

    #     loss = sum(loss_temp)
    #    return torch.mean(loss)
        # # the contra loss    
        # # conditioning     
        # better = loss_s1 - loss_s2
        # y = z.clone()
        # y.retain_grad()
        # #print(torch.mean(z,1))
        # for i in range(better.shape[0]):
        #     if better[i] < 0: # 1 is better, need swap
        #         y[[2*i, 2*i+1]] = z[[2*i+1,2*i]]
        #         #pred[i,0,:] , pred[i,1,:] = pred[i,1,:], pred[i,0,:]
        #         #print(i)                
        # #print(torch.mean(z,1))
        # z = y.view(-1,2,384)        
        # z1 = z[:,0,:]
        # z2 = z[:,1,:]
        # loss2 = F.mse_loss(z1,z2.detach())

        # #loss1 = torch.mean(self.mt_supervised(pred[:,0,:],target))
        #target_double = target.unsqueeze(1).expand(-1,2,-1).flatten(start_dim=0, end_dim=1)
