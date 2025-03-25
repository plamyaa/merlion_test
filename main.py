import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import sys
sys.path.append('./src')

from src.constants import (
    NUM_DAYS,
    NUM_EMPLOYEES,
    OUTPUT_FILE,
)
from src.schedule_generator import ScheduleGenerator
from src.schedule_validator import ScheduleValidator
from src.schedule_model import ScheduleModel


def main():
    # 1. Генерация расписания
    schedule_generator = ScheduleGenerator(num_employees=NUM_EMPLOYEES, num_days=NUM_DAYS)
    generated_schedule = schedule_generator.generate_schedule(shuffle_rows=True)
    schedule_df = pd.DataFrame(
        generated_schedule,
        columns=[f'Day_{i+1}' for i in range(NUM_DAYS)]
    )

    # Сохранение расписания
    schedule_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Расписание сохранено в файл: {OUTPUT_FILE}")

    # 2. Подготовка данных для обучения
    schedule = schedule_df.values
    noisy_schedule = ScheduleGenerator.add_noise_to_schedule(schedule, noise_level=0.50)

    # Преобразование данных
    x = noisy_schedule.reshape(noisy_schedule.shape[0], -1)
    y = schedule.reshape(schedule.shape[0], -1)

    # Разделение на тренировочные и тестовые данные
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Преобразование в тензоры
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 3. Создание и обучение модели
    output_size = x_train.shape[1] * 3  # 180 * 3
    model = ScheduleModel(input_size=x_train.shape[1], output_size=output_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Обучение
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)  # (240, 180, 3)
        # Меняю форму y_train на (batch_size, 180)
        loss = criterion(outputs.view(-1, 3), y_train.view(-1))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    # Сохранение модели
    torch.save(model.state_dict(), 'shift_scheduler_model.pth')
    print("Модель сохранена в файл: shift_scheduler_model.pth")

    # 4. Оценка модели на тестовых данных
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)  # (60, 180, 3)
        y_pred = torch.argmax(y_pred, dim=2).numpy()  # (60, 180)

    # 5. Вычисление метрик
    y_pred_flat = y_pred.flatten()  # (10800,)
    y_test_flat = y_test.flatten()  # (10800,)

    accuracy = accuracy_score(y_test_flat, y_pred_flat)
    precision = precision_score(y_test_flat, y_pred_flat, average='weighted')
    recall = recall_score(y_test_flat, y_pred_flat, average='weighted')
    f1 = f1_score(y_test_flat, y_pred_flat, average='weighted')

    print("\nМетрики на тестовых данных:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 6. Проверка расписания
    validator = ScheduleValidator(schedule)
    validation_results = validator.validate_all_rules()

    print("\nРезультаты проверки расписания:")
    for rule, passed in validation_results.items():
        status = True if passed else False
        print(f"{rule}: {status}")

    # Дозапись в файл с расписанием
    if all(validation_results.values()):  # Если все проверки пройдены успешно
        schedule_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        print(f"Новое расписание успешно дозаписано в файл: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()