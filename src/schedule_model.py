import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .constants import NUM_EMPLOYEES, NUM_DAYS


class ScheduleModel(nn.Module):
    """
    Нейронная сеть для генерации расписания сотрудников.

    Архитектура:
    - Входной слой (fc1): Преобразует входные данные в 16 нейронов.
    - Выходной слой (fc2): Генерирует предсказания для каждого сотрудника и дня (3 класса: выходной, утренняя смена, вечерняя смена).
    - Dropout: Регуляризация с вероятностью 15% для предотвращения переобучения.
    - Активация: ReLU используется для добавления нелинейности.
    """
    def __init__(self, input_size, output_size) -> None:
        super(ScheduleModel, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, output_size)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x: Tensor) -> Tensor:
        """
        Прямой проход через модель.

        Вход:
            x (torch.Tensor): Входные данные размером (batch_size, input_size).
        Выход:
            torch.Tensor: Предсказания размером (batch_size, input_size, 3).
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.view(-1, self.input_size, 3)