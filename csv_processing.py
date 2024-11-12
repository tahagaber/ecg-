import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

model_path = "model.keras"  
csv_path = "ptbdb_normal.csv"  

if not os.path.exists(model_path):
    print("خطأ: ملف النموذج غير موجود.")
else:
    model = load_model(model_path)

if not os.path.exists(csv_path):
    print("خطأ: ملف بيانات المريض غير موجود.")
else:
    # تحميل بيانات المريض من ملف CSV
    patient_data = pd.read_csv(csv_path)

    if isinstance(patient_data, pd.DataFrame) and not patient_data.empty:
        print("شكل بيانات المريض:", patient_data.shape)

        patient_data_selected = patient_data.iloc[:, :187]

        total_elements = patient_data_selected.values.size
        num_samples = total_elements // 187

        if total_elements % 187 == 0:
            patient_data_processed = patient_data_selected.values.reshape(num_samples, 187, 1, 1)

            predictions = model.predict(patient_data_processed)

            class_names = ['Normal', 'Atrial Fibrillation', 'Premature Ventricular Contractions', 'Ventricular Tachycardia', 'Bradycardia']
            predicted_classes = [class_names[np.argmax(prediction)] for prediction in predictions]

            most_common_class = np.unique(predicted_classes, return_counts=True)
            most_common_index = np.argmax(most_common_class[1])  # الحصول على الفهرس للفئة الأكثر شيوعًا
            most_common_disease = most_common_class[0][most_common_index]  # اسم المرض الأكثر شيوعًا

            print(f"نوع المرض الأكثر شيوعًا: {most_common_disease}")

            prediction_counts = dict(zip(most_common_class[0], most_common_class[1]))
            print("توزيع التنبؤات:", prediction_counts)
        else:
            print(f"عدد العناصر {total_elements} لا يمكن تقسيمه على 187.")
    else:
        print("خطأ: بيانات المريض غير صحيحة أو فارغة.")
