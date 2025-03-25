import numpy as np
from typing import Dict
from .constants import (
    NUM_DAYS,
    SHIFT_EVENING,
    SHIFT_MORNING,
    DAY_OFF,
)


class ScheduleValidator:
    """
    Класс для проверки расписания сотрудников на соответствие заданным правилам.
    """

    def __init__(self, schedule: np.ndarray):
        """
        Инициализация валидатора.

        :param schedule: Расписание сотрудников в виде numpy-массива.
                         Строки — сотрудники, столбцы — дни.
        """
        self.schedule: np.ndarray = schedule

    def check_min_two_days_off_per_week(self) -> bool:
        """
        Проверяет, чтобы у каждого сотрудника было минимум два выходных на неделе.

        :return: True, если правило выполняется, иначе False.
        """
        for employee in self.schedule:
            for week in range(0, NUM_DAYS, 7):
                if employee[week:week + 7].shape[0] <= 5:
                    continue
                if np.sum(employee[week:week + 7] == DAY_OFF) < 2:
                    return False
        return True

    def check_consecutive_days_off(self) -> bool:
        """
        Проверяет, чтобы выходные шли подряд.

        :return: True, если правило выполняется, иначе False.
        """
        for employee in self.schedule:
            for week in range(0, NUM_DAYS, 7):
                days_off = employee[week:week + 7] == DAY_OFF
                if (np.sum(days_off) >= 2 and not any(
                        np.all(days_off[i:i + 2])
                        for i in range(len(days_off) - 1))):
                    return False
        return True

    def check_min_two_employees_per_shift(self) -> bool:
        """
        Проверяет, чтобы на каждой смене было минимум два сотрудника.

        :return: True, если правило выполняется, иначе False.
        """
        for day in range(self.schedule.shape[1]):
            shifts = self.schedule[:, day]
            if (np.sum(shifts == SHIFT_MORNING) < 2 or
                    np.sum(shifts == SHIFT_EVENING) < 2):
                return False
        return True

    def check_no_morning_after_evening(self) -> bool:
        """
        Проверяет, чтобы после вечерней смены не было утренней.

        :return: True, если правило выполняется, иначе False.
        """
        for employee in self.schedule:
            if any(
                employee[i] == SHIFT_MORNING and
                employee[i - 1] == SHIFT_EVENING
                for i in range(1, len(employee))
            ):
                return False
        return True

    def check_weekly_shift_rotation(self) -> bool:
        """
        Проверяет, чтобы смены чередовались каждую неделю.

        :return: True, если правило выполняется, иначе False.
        """
        for employee in self.schedule:
            for week in range(0, NUM_DAYS - 7, 7):
                week_shifts = employee[week:week + 7]
                next_week_shifts = employee[week + 7:week + 14]
                if np.array_equal(week_shifts, next_week_shifts):
                    return False
        return True

    def check_max_five_consecutive_work_days(self) -> bool:
        """
        Проверяет, чтобы у сотрудника не было более 5 рабочих дней подряд.

        :return: True, если правило выполняется, иначе False.
        """
        for employee in self.schedule:
            work_streak = 0
            for day in employee:
                if day in [SHIFT_EVENING, SHIFT_MORNING]:
                    work_streak += 1
                    if work_streak > 5:
                        return False
                else:
                    work_streak = 0
        return True

    def validate_all_rules(self) -> Dict[str, bool]:
        """
        Проверяет расписание на соответствие всем правилам.

        :return: Словарь с результатами проверки для каждого правила.
        """
        return {
            "Минимум два выходных на неделе":
                self.check_min_two_days_off_per_week(),
            "Выходные идут подряд":
                self.check_consecutive_days_off(),
            "Минимум 2 сотрудника на смене":
                self.check_min_two_employees_per_shift(),
            "Нет утренней после вечерней":
                self.check_no_morning_after_evening(),
            "Смена утро/вечер каждую неделю":
                self.check_weekly_shift_rotation(),
            "Не более 5 рабочих дней подряд":
                self.check_max_five_consecutive_work_days()
        }