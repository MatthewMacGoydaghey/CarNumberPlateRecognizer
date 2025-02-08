<<<<<<< README.md
Установка python: sudo apt install python-pip -y;

Установка зависимсотей: pip install -r requirements.txt

Запуск сервера: python server.py


Запрос:
localhost:5005/verify_license_plate?plate_number=M960EH197
car_photo: car.jpg


Ответы:


"status": true,
"message": 'Номер соответствует входной строке'


"status": false,
"message": 'Номер не соответствует входной строке или не распознан'