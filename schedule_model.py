import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from constants import NUM_EMPLOYEES, NUM_DAYS


class ScheduleModel(nn.Module):
    """
    Нейронная сеть для генерации расписания сотрудников.

    Архитектура:
    - Входной слой (fc1): Преобразует входные данные (NUM_EMPLOYEES * NUM_DAYS) в 512 нейронов.
    - Скрытый слой (fc2): Уменьшает размерность до 256 нейронов.
    - Выходной слой (fc3): Генерирует предсказания для каждого сотрудника и дня (3 класса: выходной, утренняя смена, вечерняя смена).
    - Dropout: Регуляризация с вероятностью 30% для предотвращения переобучения.
    - Активация: ReLU для добавления нелинейности.

    Методы:
    - forward(x): Определяет, как данные проходят через модель.
        Вход:
            x (torch.Tensor): Входные данные размером (batch_size, NUM_EMPLOYEES * NUM_DAYS).
        Выход:
            torch.Tensor: Предсказания размером (batch_size, NUM_EMPLOYEES, NUM_DAYS, 3).
    """
    def __init__(self) -> None:
        super(ScheduleModel, self).__init__()
        self.fc1 = nn.Linear(NUM_EMPLOYEES * NUM_DAYS, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, NUM_EMPLOYEES * NUM_DAYS * 3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.view(-1, NUM_EMPLOYEES, NUM_DAYS, 3)
