from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import easyocr
import numpy as np

app = Flask(__name__)
port_number = 5005
CaPD_model = YOLO('CaPD_model.pt')
reader = easyocr.Reader(['en', 'ru'])

@app.route('/verify_license_plate', methods=['GET'])

def RecognizePlateNumber():
    args = request.args
    plateNumber = args.get('plate_number')
    if (not plateNumber):
        return jsonify({'status': False, 'message': 'Поле plate_number необходимо'}), 401
    if len(plateNumber) > 6 and plateNumber[6] != ' ': plateNumber = plateNumber[:6] + ' ' + plateNumber[6:]
    plateNumber = translate_and_uppercase(plateNumber)
    if 'car_photo' not in request.files:
        return jsonify({'status': False, 'message': 'Поле car_photo необходимо'}), 401
    car_photo = request.files['car_photo']
    if car_photo.filename == '':
        return jsonify({'status': False, 'message': 'Фото авто отсутствует'}), 401
    if car_photo.mimetype not in ['image/jpeg', 'image/png']:
        return jsonify({'status': False, 'message': 'Файл должен быть в формате jpg или png'}), 401
    
    image = cv2.imdecode(np.frombuffer(car_photo.read(), np.uint8), cv2.IMREAD_COLOR)
    detected_objects = []
    results = CaPD_model(image)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            if class_id in [0, 1]:
                detected_objects.append({
                    'class': 'Car' if class_id == 0 else 'Plate',
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence
                })
    if not any(obj['class'] == 'Car' for obj in detected_objects):
        return jsonify({'status': False, 'message': 'Машина не найдена на фото'}), 401
    if not any(obj['class'] == 'Plate' for obj in detected_objects):
        return jsonify({'status': False, 'message': 'Номерной знак не найден на фото'}), 401
    
    for obj in detected_objects:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{obj['class']} {obj['confidence']:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    for obj in detected_objects:
        if obj['class'] == 'Plate':
            x1, y1, x2, y2 = map(int, obj['bbox'])
            plate_image = image[y1:y2, x1:x2]
            table = np.array([(i / 255.0) ** 1.0 * 255 for i in range(256)]).astype("uint8")
            threshold_value = 0
            allowedSymbols = '0123456789ABВCEHKMМOPTТXY'
            firstRecognizedIteration = 0
            recognized_text = ""
            rangeCount = 90
            for i in range(rangeCount):
                if (i == 0):
                    result = reader.readtext(plate_image, allowlist=allowedSymbols)
                gamma_adjusted_image = cv2.LUT(plate_image, table)
                gray = cv2.cvtColor(gamma_adjusted_image, cv2.COLOR_BGR2GRAY)
                _, thresh_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
                if (i <= 45 and i > 0):
                    blur = cv2.GaussianBlur(thresh_image, (3, 3), 2, 2)
                    result = reader.readtext(blur, allowlist=allowedSymbols)
                if (i > 45):
                    result = reader.readtext(thresh_image, allowlist=allowedSymbols)
                if (result):
                    recognized_text = ' '.join([text[1] for text in result])
                    if (len(recognized_text) > 7):
                        translatedUpperCased_text = translate_and_uppercase(recognized_text)
                        processed_text = process_string(translatedUpperCased_text.strip())
                        if (processed_text == plateNumber):
                           return jsonify({'status': True, 'message': 'Номер успешно распознан и соответсвует заявленному'}), 200
                        #for _, text in enumerate(result):
                        # confidence = text[2]
                        # print(f"Итерация: {i}, Текст: {processed_text}, {recognized_text} Confidence: {confidence:.2f}")
                    if (i > 1 and firstRecognizedIteration == 0):
                        firstRecognizedIteration = i
                if (i == 45):
                   threshold_value = firstRecognizedIteration * 6
                   rangeCount = rangeCount - firstRecognizedIteration
                if (rangeCount == i): break
                threshold_value += 6
    return ({'status': False, 'message': "Номер на фото не распознан или не соответствует заявленному"}), 200


def process_string(input_string):
    num_prefix = ''
    if len(input_string) > 0 and input_string[0].isdigit():
        num_prefix += input_string[0]
    if len(input_string) > 1 and input_string[1].isdigit():
        num_prefix += input_string[1]
    if len(input_string) > 2 and input_string[2].isdigit():
        num_prefix += input_string[2]
    if len(num_prefix) > 0 and len(input_string) > len(num_prefix) and input_string[len(num_prefix)] == ' ':
        post_space = num_prefix
        input_string = input_string[len(num_prefix) + 1:]
    else:
        post_space = ''
    if input_string and input_string[0] == '1':
        input_string = input_string[1:]
    if input_string and input_string[0] == '8':
        input_string = 'B' + input_string[1:]
    if len(input_string) > 2 and input_string[0] == 'E' and any(c.isalpha() for c in input_string[1:]) and any(c.isdigit() for c in input_string[1:]):
        input_string = input_string[1:]
    replacements = {
        '4': 'A', '0': 'O', '1': 'K',
        '7': 'Y', '8': 'B', '9': 'M',
        '5': 'E', '6': 'K'
    }
    if len(input_string) > 5:
        for i in [0, 4, 5]:
            if input_string[i] in replacements:
                input_string = input_string[:i] + replacements[input_string[i]] + input_string[i + 1:]
    if len(input_string) >= 4:
        for i in range(1, 4):
            if input_string[i] in ['T', 'Т']:
                input_string = input_string[:i] + '1' + input_string[i + 1:]
            elif input_string[i] == 'A':
                input_string = input_string[:i] + '4' + input_string[i + 1:]
            elif input_string[i] in ['O', 'О']:
                input_string = input_string[:i] + '0' + input_string[i + 1:]
            elif input_string[i] in ['B', 'В']:
                input_string = input_string[:i] + '8' + input_string[i + 1:]
    if len(input_string) > 6 and ' ' not in input_string[:7]:
        input_string = input_string[:6] + ' ' + input_string[6:]
    pre_space = input_string.strip()
    full_string = pre_space + (' ' + post_space if post_space else '')
    full_list = list(full_string)
    for i in range(6, len(full_string)):
        if full_list[i] in ['K', 'К']:
            full_list[i] = '6'
        elif full_list[i] in ['E', 'Е']:
            full_list[i] = '5'
        elif full_list[i] in ['T', 'Т']:
            full_list[i] = '7'
        elif full_list[i] in ['P']:
            full_list[i] = '2'
        full_string = ''.join(full_list)
    if len(full_string) > 10:
       full_string = full_string[:10]
    return full_string
    

def translate_and_uppercase(input_string):
    transliteration_dict = {
        'А': 'A', 'В': 'B', 'У': 'U', 'Х': 'Kh', 'М': 'M',
        'Н': 'N', 'О': 'O', 'Т': 'T', 'Е': 'E', 'Р': 'R',
        'С': 'S', 'К': 'K'
    }
    input_string = input_string.upper()
    translated_string = ''
    for char in input_string:
        if char in transliteration_dict:
            translated_string += transliteration_dict[char]
        else:
            translated_string += char
    return translated_string


app.run(debug=True, host='localhost', port=port_number)