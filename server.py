from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import easyocr
import numpy as np

app = Flask(__name__)
CaPDmodel = YOLO('CaPD_model.pt')
reader = easyocr.Reader(['en', 'ru'])

@app.route('/verify_license_plate', methods=['GET'])

def RecognizePlateNumber():
    args = request.args
    plateNumber = args.get('number')
    if (not plateNumber):
        return jsonify({'status': False, 'message': 'Поле number необходимо'}), 401
    if len(plateNumber) > 6 and plateNumber[6] != ' ': plateNumber = plateNumber[:6] + ' ' + plateNumber[6:]
    plateNumber = translate_and_uppercase(plateNumber)
    if 'photo' not in request.files:
        return jsonify({'status': False, 'message': 'Поле photo необходимо'}), 401
    photo = request.files['photo']
    if photo.filename == '':
        return jsonify({'status': False, 'message': 'Фото авто отсутствует'}), 401
    
    image = cv2.imdecode(np.frombuffer(photo.read(), np.uint8), cv2.IMREAD_COLOR)
    detected_objects = []
    results = CaPDmodel(image)
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

    for obj in detected_objects:
        print(f"Обнаружен объект: {obj['class']}, Координаты: {obj['bbox']}, Вероятность: {obj['confidence']:.2f}")
    if not any(obj['class'] == 'Car' for obj in detected_objects):
        return jsonify({'status': False, 'message': 'Машина не найдена на фото'}), 401
    if not any(obj['class'] == 'Plate' for obj in detected_objects):
        return jsonify({'status': False, 'message': 'Номерной знак не найден на фото'}), 401
    
    for obj in detected_objects:
        x1, y1, x2, y2 = map(int, obj['bbox'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{obj['class']} {obj['confidence']:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Detected Objects', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for obj in detected_objects:
        if obj['class'] == 'Plate':
            x1, y1, x2, y2 = map(int, obj['bbox'])
            plate_image = image[y1:y2, x1:x2]
            table = np.array([(i / 255.0) ** 1.0 * 255 for i in range(256)]).astype("uint8")
            threshold_value = 0
            recognized_text = ""
            for i in range(45):
                if i > 0:
                    gamma_adjusted_image = cv2.LUT(plate_image, table)
                    gray = cv2.cvtColor(gamma_adjusted_image, cv2.COLOR_BGR2GRAY)
                    _, thresh_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
                    blur = cv2.GaussianBlur(thresh_image, (3, 3), 2, 2)
                    result = reader.readtext(blur, allowlist='0123456789ABВCEHKMМOPTТXY')
                else:
                    result = reader.readtext(plate_image, allowlist='0123456789ABВCEHKMМOPTТXY')
                if result:
                    recognized_text = ' '.join([text[1] for text in result])
                    translatedUpperCased_text = translate_and_uppercase(recognized_text)
                    processed_text = process_string(translatedUpperCased_text.strip())
                    if (processed_text == plateNumber):
                        return jsonify({'status': True, 'message': 'Номер успешно распознан и соответсвует заявленному'}), 200
                    print(f"Итерация {i}: Распознанный номер: {processed_text}")
                
                threshold_value += 6

    return ({'status': False, 'message': "Номер на фото не распознан или не соответствует заявленному"}), 200


def process_string(input_string):
    if input_string and input_string[0] == '1':
        input_string = input_string[1:]
    if input_string and input_string[0] == '8':
        input_string = 'B' + input_string[1:]
    replacements = {
        '4': 'A', '0': 'O', '1': 'K', '7': 'Y', '8': 'B', '9': 'M', '5': 'E'
    }
    if len(input_string) >= 7:
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
    pre_space, post_space = input_string.split(' ', 1) if ' ' in input_string else (input_string, '')
    if post_space:
        if post_space[0] in ['B', 'В']:
            post_space = post_space[1:]
        if post_space and post_space[0] == 'E':
            if len(post_space) < 3 or (len(post_space) >= 4 and post_space[3] != ' '):
                post_space = '1' + post_space[1:]
            else:
                post_space = post_space[1:]
        for char in post_space:
            if char.isalpha() and not char.isdigit():
                if len(post_space) < 3 or (len(post_space) >= 4 and post_space[3] != ' '):
                    post_space = '1' + post_space[1:]
                else:
                    post_space = post_space.replace(char, '', 1)
                break
        if len(post_space) > 3:
            post_space = post_space[:3]
    return pre_space + (' ' + post_space if post_space else '')


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


if __name__ == '__main__':
    app.run(debug=True)