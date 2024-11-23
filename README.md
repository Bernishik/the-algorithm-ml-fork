This project open sources some of the ML models used at Twitter.

Currently these are:

1. The "For You" Heavy Ranker (projects/home/recap).

2. TwHIN embeddings (projects/twhin) https://arxiv.org/abs/2202.05387


This project can be run inside a python virtualenv. We have only tried this on Linux machines and because we use torchrec it works best with an Nvidia GPU. To setup run

`./images/init_venv.sh` (Linux only).

The READMEs of each project contain instructions about how to run each project.

____
## Опис проекту

Цей проєкт призначений для дослідження роботи компонента Heavy Ranker з відкритого коду рекомендаційних систем Х
Данний репозиторій є форком оригінального репозиторію https://github.com/twitter/the-algorithm-ml
З доданим власним кодом/конфігами для проведення дослідження даного компоненту
____
## Структура репозиторію

Структура повністю відповідає оригінальному репозиторію за вийнятком коду написаного мною для проведення дослідження, а саме:
- main.py: Основний файл з реалізацією використання натренованої моделі та підрахунку фінального скора
- README.md: Документація проекту.
- projects/home/recap/config/local_prod.yaml: конфіги для роботи з моделлю
- projects/home/recap/script/create_random_data.sh: скрипт для генерації вхідних данних для моделі
- projects/home/recap/script/run_local.sh: скрипт для тренування моделі
____
## Особливості

- Запуск компоненти з мінімальними налаштуваннями від користувача
- Генерація випадкових даних для подальшого викристання
- Легкий запуск налаштованого треннування моделі
- Доданий функціонал підрахунку скору по вагам та вихідним данним

____
## Як зібрати і запустити

### Системні вимоги
Операційна система Linux(бажано ubuntu)

Nvidia GPU з СUDA ядрами(робота виконана на відеокарті RTX 4070 TI SUPER)



### Інструкції
1. Клонувати репозиторій:

   `git clone https://github.com/Bernishik/the-algorithm-ml-fork`

   `cd the-algorithm-ml-fork`
2. Встановити Python 3.10

   `sudo apt update`

   `sudo apt install python3.10`

   `sudo apt install python3.10-venv`

   `sudo apt-get install python3-tk`
3. Опціонально: змінити в файлі images/init_venv.sh PYTHONBIN на шлях до директорії python

   `nano ./images/init_venv.sh`

   9 стрічку: "PYTHONBIN=$(which python3.10)"

   замінити на 'PYTHONBIN="path-to-python"' де path-to-python це шлях до bin директорії python
4. Запустити налаштування віртуального середовища з встановленням необхідних залежностей

   `./images/init_venv.sh`
5. Активувати віртуальне середовище

   `source $HOME/tml_venv/bin/activate`

6. Встановелння залежностей

   
   `pip install matplotlib==3.8 --no-deps`

   `pip install seaborn --no-deps`

   `pip install pillow --no-deps`

   `pip install cycler --no-deps`

   `pip install kiwisolver --no-deps`

   `pip install scipy --no-deps`
   
6. Генерація випадкових даних

   `projects/home/recap/script/create_random_data.sh`
7. Запуск тренування моделі на згенерованих даних

   `projects/home/recap/script/run_local.sh`

8. Запуск оцінки score

   `python main.py`

### Додаткові команди
1. Генерація випадкових даних

   `projects/home/recap/script/create_random_data.sh`
2. Запуск тренування моделі на згенерованих даних

   `projects/home/recap/script/run_local.sh`
3. Запуск оцінки score

   `python main.py`
4. Активація віртуального середовища

   `source $HOME/tml_venv/bin/activate`
5. Деактивація віртуального середовища

   `source /home/bernish/tml_venv/bin/deactivate`
