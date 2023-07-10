# Prueba Final - Limpieza de Datos
Tema: Creacion de una CNN para reconocimiento de tipos de carnes

Maestria: Ciberseguridad

Estudiante: Andres Fernando Cardenas Ponce - 1805369194

Dataset: https://drive.google.com/file/d/1Z5DJ-MVS1TQV1kow9mIFWTec-ZdOLRLF/view?usp=sharing

# Metodologia
En esta práctica final, se utilizó un modelo de red neuronal convolucional (CNN) para la clasificación de imágenes. La metodología seguida consistió en los siguientes pasos:

1. Lectura y preprocesamiento de datos: Se cargaron las imágenes de entrenamiento y prueba, y se realizaron operaciones de preprocesamiento, como la redimensionado de las imágenes a un tamaño específico y la normalización de los valores de píxeles.

2. Diseño de la arquitectura de la CNN: Se definió la estructura de la red neuronal convolucional, que consta de capas convolucionales, capas de agrupación, capas de activación y capas completamente conectadas. La arquitectura se diseñó de manera que pudiera aprender automáticamente las características y patrones relevantes de las imágenes.

3. Entrenamiento del modelo: Se realizó el entrenamiento del modelo CNN utilizando los datos de entrenamiento. Durante el entrenamiento, se ajustaron los pesos y los sesgos de la red neuronal mediante la minimización de una función de pérdida, utilizando un algoritmo de optimización.

4. Evaluación del modelo: Se evaluó el rendimiento del modelo utilizando los datos de prueba. Se calculó el porcentaje de aciertos, que indica la precisión de la clasificación del modelo en las imágenes no vistas durante el entrenamiento.

# Resultados
El modelo CNN logró un alto rendimiento en la clasificación de imágenes. El porcentaje de aciertos obtenido en los datos de prueba fue del 90.12%, lo que indica que el modelo fue capaz de clasificar correctamente la gran mayoría de las imágenes no vistas previamente.

Además, en los datos de entrenamiento, el modelo CNN alcanzó un porcentaje de aciertos del 98.43%. Esto demuestra que la red neuronal pudo aprender eficazmente los patrones y características de las imágenes de entrenamiento y generalizar ese conocimiento para clasificar correctamente nuevas instancias.

La capacidad de la CNN para extraer automáticamente características relevantes de las imágenes a través de las capas convolucionales y aprender la representación óptima de los datos contribuyó al éxito de la clasificación de imágenes.

# Conclusiones 
En base a los resultados obtenidos, se puede concluir que el modelo CNN es altamente efectivo para la clasificación de imágenes. Su capacidad para aprender características relevantes de manera automática y su habilidad para generalizar y clasificar correctamente nuevas instancias lo convierten en una elección prometedora.

El uso de una arquitectura de red neuronal convolucional permite aprovechar la estructura y los patrones visuales presentes en las imágenes, lo que mejora significativamente la precisión de la clasificación.

Sin embargo, es importante tener en cuenta que el rendimiento del modelo está influenciado por la calidad y cantidad de datos de entrenamiento. En futuras investigaciones, sería recomendable aumentar la cantidad de datos disponibles y mejorar la diversidad y representatividad de las imágenes en cada clase para obtener resultados aún más precisos.
