import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, use_cosine_distance=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.use_cosine_distance = use_cosine_distance

    def forward(self, anchor, positive, negative):
        # Нормализация эмбеддингов
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        if self.use_cosine_distance:
            # Используем косинусное расстояние
            distance_positive = 1 - F.cosine_similarity(anchor, positive, dim=1)
            distance_negative = 1 - F.cosine_similarity(anchor, negative, dim=1)
        else:
            # Используем евклидово расстояние
            distance_positive = torch.norm(anchor - positive, dim=1)
            distance_negative = torch.norm(anchor - negative, dim=1)

        # Вычисляем Triplet Loss
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()