import os
import cv2
import albumentations as A

# Diretórios
input_dir = '../../fotos_treino'  # Diretório de entrada com as imagens originais
output_dir = 'fotos_aumentadas'  # Diretório de saída para as imagens aumentadas

# Cria o diretório de saída se não existir
os.makedirs(output_dir, exist_ok=True)

# Função para carregar imagem
def load_image(file_path):
    return cv2.imread(file_path)

# Função para salvar imagem
def save_image(file_path, image):
    cv2.imwrite(file_path, image)

# Aumento de dados com Albumentations
def augment_image(image, augmenter):
    augmented = augmenter(image=image)
    return augmented['image']

# Transformações de aumento de dados
augmenter = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=45, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
    A.Blur(blur_limit=5, p=0.1),
    A.CLAHE(clip_limit=4.0, p=0.2),
    A.RandomGamma(p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2),
    A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.2),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.1)
])

# Processamento das imagens
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_dir, filename)
        image = load_image(image_path)
        for i in range(10):  # Criar 10 variações de aumento por imagem
            augmented_image = augment_image(image, augmenter)
            augmented_image_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_aug_{i}.jpg')
            save_image(augmented_image_path, augmented_image)