import numpy as np
from typing import Optional
from .constants import NUM_EMPLOYEES, NUM_DAYS


class ScheduleGenerator:
    """
    Класс для генерации расписания сотрудников с учетом заданных правил.
    """

    def __init__(
            self,
            num_employees: int = NUM_EMPLOYEES,
            num_days: int = NUM_DAYS
    ) -> None:
        """
        Инициализация генератора расписания.

        :param num_employees: Количество сотрудников.
        :param num_days: Количество дней.
        """
        self.num_employees: int = num_employees
        self.num_days: int = num_days
        self.schedule: Optional[np.ndarray] = None

    def generate_schedule(self, shuffle_rows: bool = False) -> np.ndarray:
        """
        Генерация расписания для сотрудников.

        :param shuffle_rows: Если True, строки расписания будут перемешаны.
        :return: Сгенерированное расписание в виде numpy-массива.
        """
        adjusted_num_days = ((self.num_days + 6) // 7) * 7  # Округление вверх до кратного 7
        schedule = np.zeros((self.num_employees, adjusted_num_days), dtype=int)

        # 1. Генерация расписания для первого сотрудника
        for week_start in range(0, adjusted_num_days, 7):
            week_end = min(week_start + 7, adjusted_num_days)
            if (week_start // 7) % 2 == 0:
                schedule[0, week_start:week_end] = 1
            else:
                schedule[0, week_start:week_end] = 2
            if week_end - week_start >= 2:  # Выходные
                schedule[0, week_end - 2:week_end] = 0

        # 2. Копирование первого сотрудника
        schedule[1] = schedule[0]

        # 3. Копирование первого и второго, инвертирование для третьего и четвертого
        for emp in [2, 3]:
            inverted_schedule = schedule[0 if emp == 2 else 1].copy()
            inverted_schedule[inverted_schedule == 1] = 3
            inverted_schedule[inverted_schedule == 2] = 1
            inverted_schedule[inverted_schedule == 3] = 2
            schedule[emp] = inverted_schedule

        # 4. Копирование 1-4 сотрудников и сдвиг вправо на 2 дня
        for emp in range(4, min(8, self.num_employees)):
            base_emp = emp % 4
            shifted_schedule = np.roll(schedule[base_emp], shift=2)
            if np.all(shifted_schedule[:2] == 0):
                shifted_schedule[:2] = 0
            schedule[emp] = shifted_schedule

        # 5. Генерация остальных сотрудников
        for emp in range(8, self.num_employees):
            base_emp = emp - 4
            shifted_schedule = np.roll(schedule[base_emp], shift=2)
            for day in range(adjusted_num_days - 5):
                if np.all(shifted_schedule[day:day + 5] != 0):
                    shifted_schedule[day + 5] = 0  # Добавляем выходной
            schedule[emp] = shifted_schedule

        # Обрезаем до исходной длины
        schedule = schedule[:, :self.num_days]

        # Если нужно перемешать строки
        if shuffle_rows:
            np.random.shuffle(schedule)

        self.schedule = schedule
        return schedule

    @staticmethod
    def add_noise_to_schedule(
        schedule: np.ndarray,
        noise_level: float = 0.05
    ) -> np.ndarray:
        """
        Добавляет шум в расписание, изменяя случайные смены.

        :param schedule: numpy-массив расписания.
        :param noise_level: Доля данных, в которые вносится шум.
        :return: Расписание с шумом.
        """
        noisy_schedule = schedule.copy()
        num_employees, num_days = noisy_schedule.shape
        num_changes = int(noise_level * num_employees * num_days)

        for _ in range(num_changes):
            emp = np.random.randint(0, num_employees)
            day = np.random.randint(0, num_days)
            noisy_schedule[emp, day] = np.random.choice([0, 1, 2])  # Выходной, утренняя, вечерняя

        return noisy_schedule
