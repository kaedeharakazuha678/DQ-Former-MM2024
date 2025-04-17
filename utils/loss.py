import torch
import torch.nn as nn
import torch.nn.functional as F
from .InfoNCE import InfoNCE

# samples_per_class = [4710,1743,683,1109,268,271,1205]
# weights = [1.0 / n for n in samples_per_class]
# normalized_weights = [w / sum(weights) for w in weights]
# self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(normalized_weights).to('cuda'))

class MSELoss:
    def __init__(self):
        self.criterion = nn.MSELoss()
    def get_loss(self, output, target):
        return self.criterion(output, target)
    

class CrossEntropyLoss:
    def __init__(self):    
        self.criterion = nn.CrossEntropyLoss()      
    def get_loss(self, output, target):
        return self.criterion(output, target)
    

class LabelSmoothingCELoss(nn.Module):
    def __init__(self, epsilon=0.1):
        super(LabelSmoothingCELoss, self).__init__()
        self.epsilon = epsilon
        
    def get_loss(self, output, target):
        C = output.size(1)
        log_preds = F.log_softmax(output, dim=1)
        loss = -(1.0 - self.epsilon) * log_preds.gather(1, target.unsqueeze(-1))
        loss = loss.sum(-1) / C
        loss = loss + self.epsilon * log_preds.mean()
        
        return loss.mean()
    

class MELDSmoothedCrossEntropyLoss:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        samples_per_class = [4710,1743,683,1109,268,271,1205]
        weights = [1.0 / n for n in samples_per_class]
        normalized_weights = torch.tensor([w / sum(weights) for w in weights])
        self.class_weights = normalized_weights

    def get_loss(self, output, target):
        # Compute standard cross entropy
        log_preds = torch.nn.functional.log_softmax(output, dim=-1)
        nll_loss = torch.nn.functional.nll_loss(log_preds, target, weight=self.class_weights, reduction='mean')
        
        # Compute the smooth loss
        n_classes = output.size(-1)
        loss = -log_preds.mean(dim=-1)
        smooth_loss = loss.mean()
        
        # Combine the two loss types
        final_loss = (1 - self.epsilon) * nll_loss + self.epsilon * smooth_loss
        return final_loss


class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def get_loss(self, input, target, pad_mask=None):
        # input: (batch_size, seq_len, num_classes)
        # target: (batch_size, seq_len, num_classes)
        input = F.log_softmax(input, dim=-1)
        target = F.softmax(target, dim=-1)
        loss = self.kl_loss(input, target)
        if pad_mask is not None:
            loss.masked_fill_(pad_mask, 0.)
        return loss


class JSdivergenceLoss(nn.Module):
    def __init__(self):
        super(JSdivergenceLoss, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def get_loss(self, p, q, pad_mask=None):
        # p: (batch_size, seq_len, num_classes)
        # q: (batch_size, seq_len, num_classes)
        m = 0.5 * (p + q)
        loss = 0.5 * self.kl_loss(p, m) + 0.5 * self.kl_loss(q, m)
        if pad_mask is not None:
            loss.masked_fill_(pad_mask, 0.)
        return loss

        
#region Knowledge Distillation Losses (KD Losses)      
class Softlabel_KDLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(Softlabel_KDLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def get_loss(self, student_logits, teacher_logits):
        student_logits = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_logits = F.softmax(teacher_logits / self.temperature, dim=-1)
        return self.criterion(student_logits, teacher_logits)
    
class NST_KDLoss(nn.Module):
    # Neural Style Transfer Knowledge Distillation Loss
    # 1. we use squared maximum mean discrepancy (MMD) (Gretton et al., 2012) which uses the kernel trick to measure 
    # and minimize the distance between the activation patterns of student neurons and the teacher neurons
    # 2. We chose to use a polynomial kernel k[s; t] =(s⊤t + c)d with d = 2 and c = 0 due to its superior performance 
    # over other kernels in (Huang and Wang, 2017)’s experiments. 
    # Kernel: polynomial kernel
    # d = 2 and c = 0 by default
    def __init__(self, d=2, c=0):
        super(NST_KDLoss, self).__init__()
        self.d = d
        self.c = c

    def polynomial_kernel(self, x1, x2):
        return (torch.dot(x1, x2) + self.c) ** self.d

    def squared_MMD(self, student_features, teacher_features):
        batch_size, seq_len, dimension = student_features.size(0), student_features.size(1), student_features.size(2)
        loss_all = torch.zeros((batch_size, ))
        teacher_features_broadcast = teacher_features.unsqueeze(1).expand(-1, seq_len, -1)
        for n in range(batch_size):
            xx, yy, zz = 0, 0, 0
            for i in range(dimension):
                for j in range(dimension):
                    # calculate all the possible combinations of the student features: polynomial_kernel(X[n,:,i], X[n,:,j])
                    xx += self.polynomial_kernel(student_features[n, :, i], student_features[n, :, j])

                    # calculate all the possible combinations of the teacher features broadcast: polynomial_kernel(Y[n,:,i], Y[n,:,j])
                    yy += self.polynomial_kernel(teacher_features_broadcast[n, :, i], teacher_features_broadcast[n, :, j])
                    
                    # calculate all the possible combinations of the student and teacher features: polynomial_kernel(X[n,:,i], Y[n,:,j])
                    zz += self.polynomial_kernel(student_features[n, :, i], teacher_features_broadcast[n, :, j])
            
            # print(xx, yy, zz, loss_all)
            
            loss_all[n] = (xx  + yy  - 2 * zz) / dimension ** 2

        return loss_all.mean()

    def forward(self, student_features, teacher_features):
        return self.squared_MMD(student_features, teacher_features)


class MMD:
    def __init__(self, kernel='gaussian', kernel_params=None):
        """
        Initialize the MMD object.

        Args:
            kernel (str): Kernel function to use ('gaussian' or 'polynomial').
            kernel_params (dict): Parameters for the kernel function.
        """
        self.kernel = kernel
        self.kernel_params = kernel_params

    def _gaussian_kernel(self, x, y):
        gamma = self.kernel_params.get('gamma', 1.0)
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)

        x = x.view(x_size, 1, dim)
        y = y.view(1, y_size, dim)

        xy = torch.norm(x - y, dim=2)
        xy = xy * xy / (2.0 * gamma * gamma)

        gram_matrix = torch.exp(-xy)
        return gram_matrix

    def _polynomial_kernel(self, x, y):
        degree = self.kernel_params.get('degree', 2)
        coef = self.kernel_params.get('coef', 1)
        const = self.kernel_params.get('const', 1)

        return (coef * torch.mm(x, y.t()) + const) ** degree

    def get_loss(self, x, y):
        """
        Compute the Maximum Mean Discrepancy (MMD) between two distributions.

        Args:
            x (torch.Tensor): Samples from the first distribution.
            y (torch.Tensor): Samples from the second distribution.

        Returns:
            torch.Tensor: MMD between x and y.
        """
        if self.kernel == 'gaussian':
            xx = self._gaussian_kernel(x, x)
            yy = self._gaussian_kernel(y, y)
            xy = self._gaussian_kernel(x, y)
        elif self.kernel == 'polynomial':
            xx = self._polynomial_kernel(x, x)
            yy = self._polynomial_kernel(y, y)
            xy = self._polynomial_kernel(x, y)
        else:
            raise ValueError("Unsupported kernel type. Use 'gaussian' or 'polynomial'.")

        mmd = (torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy))/x.size(-1)**2
        return mmd


class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, student, teacher):
        batch_size = int(student.size()[0])
        kernels = self.guassian_kernel(student, teacher, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
   
# contrastive learning loss        
class CRD_KDLoss(nn.Module):
    # Contrastive Representation Distillation (CRD)
    # Just like before, given input audio a and text x, t(a) ∈ Rd and s(x) ∈ R|x|×d are teacher and student 
    # representations respectively. From the joint distribution of teacher and student representations, we 
    # get one positive pair for every N (batch-size) negative pairs that are drawn from the product of the marginal
    # distributions. To maximize the lower bound of the mutual information between s(x) and t(a), we have to minimize 
    # the following loss function:
    # L = -log(exp(s(x)⊤t(a)/τ) / (exp(s(x)⊤t(a)/τ) + ∑exp(s(x)⊤t(a_neg)/τ)))
    # where τ is the temperature parameter and a_neg is the negative sample of a.
    
    def __init__(self, tau=1.0):
        super(CRD_KDLoss, self).__init__()
        self.tau = tau

    def forward(self, s_x, t_a):
        batch_size, seq_len, _ = s_x.size()

        # Calculate dot product of student and teacher representations
        dot_product_pos = torch.sum(s_x * t_a.unsqueeze(1), dim=2) / self.tau  # unsqueeze to match dimensions

        # Calculate dot product for negative samples
        negative_samples = t_a[torch.randperm(batch_size), :]  # permute the batch to get negative samples
        dot_product_neg = torch.sum(s_x * negative_samples.unsqueeze(1), dim=2) / self.tau

        # Calculate numerator and denominator terms for the loss
        numerator = torch.exp(dot_product_pos)
        denominator = numerator + torch.sum(torch.exp(dot_product_neg), dim=0)

        # Compute the loss
        loss = -torch.log(numerator / denominator).mean()

        return loss
      

class InfoNCE_KDLoss(nn.Module):
    def __init__(self, info_temp) -> None:
        super().__init__()
        self.info_temp = info_temp
        self.infoNCE = InfoNCE(temperature=info_temp, reduction='mean', negative_mode=None)
        
    def get_loss(self, student_features, teacher_features):
        '''
        student_features: (batch_size, embedding_size)
        teacher_features: (batch_size, embedding_size)
        explanation:
            We aim to learn a good multimodal representation by maximizing the mutual information between the student and teacher representations.
            the student_features and teacher_features are the output of the text encoder(eg. bert-large) and audio encoder (eg. whisper) respectively.
            if the text modality of sample a is the query, then the corresponding audio modality is the key, 
            and other samples' audio modality in the same mini-batch are negative keys.
        '''
        query = student_features
        positive_key = teacher_features
        return self.infoNCE(query, positive_key)

#endregion


#region Contrastive Learning Losses (CL Losses)

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def get_loss(self, features, labels):
        # Calculate pairwise Euclidean distances
        distances = F.pairwise_distance(features.unsqueeze(1), features.unsqueeze(0), p=2)

        # Generate a mask for positive pairs (same labels)
        mask_positive = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))

        # Generate a mask for negative pairs (different labels)
        mask_negative = ~mask_positive

        # Extract positive and negative distances
        distances_positive = distances[mask_positive].view(features.size(0), -1)
        distances_negative = distances[mask_negative].view(features.size(0), -1)

        # Calculate the contrastive loss
        loss_positive = torch.mean(distances_positive)
        loss_negative = torch.mean(torch.clamp(self.margin - distances_negative, min=0))

        # margin function: max(margin - d(a, b), 0)
        # 较小的 margin 可能会导致相对较近的同类别样本之间的相似度也会被惩罚，而较大的 margin 则可能使模型对于不同类别样本的区分更为明显。

        # Total contrastive loss
        total_loss = loss_positive + loss_negative

        return total_loss
    

class ContrastiveLossCosineSimilarity(torch.nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLossCosineSimilarity, self).__init__()
        self.margin = margin

    def forward(self, features, labels):
        # Normalize the feature vectors
        features_normalized = F.normalize(features, p=2, dim=1)

        # Calculate cosine similarity
        similarities = torch.mm(features_normalized, features_normalized.t())

        # Generate a mask for positive pairs (same labels)
        mask_positive = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))

        # Generate a mask for negative pairs (different labels)
        mask_negative = ~mask_positive

        # Extract positive and negative similarities
        similarities_positive = similarities[mask_positive].view(features.size(0), -1)
        similarities_negative = similarities[mask_negative].view(features.size(0), -1)

        # Calculate the contrastive loss
        loss_positive = torch.mean(similarities_positive)
        loss_negative = torch.mean(torch.clamp(self.margin - similarities_negative, min=0))

        # Total contrastive loss
        total_loss = loss_positive + loss_negative

        return total_loss

#endregion