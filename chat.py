from telegram import Update
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

por_defecto = str('''Disculpa no entendí la pregunta que me hiciste. 
    Aqui te dejo una lista de los temas con los cuales estoy entrenado a responder: \n


    "¿Qué es Kron Smart Chain?",\n

    "¿Cuál es el protocolo de consenso utilizado por Kron Smart Chain?",\n

    "¿Cómo se crean los criptoactivos en la Red Kron?",\n

    "¿Qué hace destacar a Kron Smart Chain en cuanto a la tokenización de activos del mundo real?",

    "¿Cómo se gestionan las recompensas en Kron Smart Chain?",\n

    "¿Qué ventajas tiene el modelo UTXO en Kron Smart Chain?",\n

    "¿Cómo contribuye la base de datos relacional en Kron Smart Chain?",\n

    "¿Cómo se refuerza la privacidad y seguridad en la red de Kron Smart Chain?",\n

    "¿Cuál es el enfoque principal de Kron Smart Chain?",\n

    "¿Cómo contribuye la participación activa de los nodos en la red?",\n

    "¿Qué son los criptoactivos en la red Kron?",\n

    "¿Cómo funciona el estándar de token kr10 en Kron Smart Chain?",\n

    "¿Cuál es la característica destacada de Kron Smart Chain en la tokenización de activos del mundo real?",\n

    "¿Cómo se pueden pagar recompensas en Kron Smart Chain?",\n

    "¿Cuál es el modelo central en la arquitectura de Kron Smart Chain?",\n

    "¿Por qué es relevante el uso de la red Tor en Kron Smart Chain?",\n

    "¿En qué sectores puede Kron Smart Chain encontrar aplicaciones?",
    ''').replace('"', '')

data = {
    "¿Quien eres?":"Soy un bot de inteligencia artficial que esta entrenado sólo con datos referentes a la Blockchain KRON SMART CHAIN",
    "Hola": "¡Hola! ¿Cómo puedo ayudarte?",
    "¿Cómo estás?": "Estoy bien, gracias. ¿Y tú?",
    "Edad": "Lo siento, soy un programa de computadora y no tengo edad.",
    "¿Cuál es tu nombre?": "Me llaman ChatBot.",
    "Adiós": "¡Hasta luego! ¿Vuelve pronto?",
    "¿Qué haces?": "Estoy aquí para ayudarte y responder tus preguntas.",
    "¿Cómo funciona el chatbot?": "Utilizo algoritmos de procesamiento de lenguaje natural para entender y responder preguntas.",
    "¿Dónde vives?": "Vivo en el mundo digital, no tengo una ubicación física.",
    "¿Cuál es tu color favorito?": "No tengo preferencias de color, ya que soy un programa de computadora.",
    "¿Puedes contarme un chiste?": "¡Claro! ¿Por qué los programadores prefieren el aire libre? Porque la pantalla del sol.",
     "¿Qué es Kron Smart Chain?": "Kron Smart Chain es un sistema electrónico peer to peer para la creación, transferencia de activos y almacenamiento de datos en forma de texto con cualquier codificación. Funciona como una blockchain descentralizada con énfasis en seguridad, control del usuario, privacidad y resistencia a la censura.",
    
    "¿Cuál es el protocolo de consenso utilizado por Kron Smart Chain?": "Utiliza el protocolo de consenso 'Distributed Validation Network,' que se traduce como 'Red de Validación Distribuida.' Este sistema distribuye la validación de datos entre múltiples nodos en la red.",
    
    "¿Cómo se crean los criptoactivos en la Red Kron?": "Se generan mediante tres transacciones, que incluyen la recepción de activos emitidos, la obtención de un token administrativo kr117 y la operación de comisión de red.",
    
    "¿Qué hace destacar a Kron Smart Chain en cuanto a la tokenización de activos del mundo real?": "Permite la representación digital de cualquier activo del mundo real en la cadena de bloques, ofreciendo mayor liquidez y acceso a inversiones antes inaccesibles, garantizando transparencia y seguridad.",
    
    "¿Cómo se gestionan las recompensas en Kron Smart Chain?": "Se pueden pagar recompensas en el token nativo de Kron, dividiéndolas automáticamente en partes iguales y enviándolas proporcionalmente a los titulares del activo.",
    
    "¿Qué ventajas tiene el modelo UTXO en Kron Smart Chain?": "El modelo UTXO (Unspent Transaction Output) permite una estructura eficiente para transacciones, seleccionando UTXO como fuente para garantizar la integridad y seguridad en el sistema.",
    
    "¿Cómo contribuye la base de datos relacional en Kron Smart Chain?": "La base de datos relacional proporciona una estructura organizada para consultas avanzadas y operaciones complejas, asegurando la coherencia y actualización de las relaciones entre diferentes conjuntos de datos.",
    
    "¿Cómo se refuerza la privacidad y seguridad en la red de Kron Smart Chain?": "Utiliza la red Tor para la transmisión de datos y la sincronización de nodos, combinada con la base de datos relacional y el modelo UTXO, solidificando la robustez y privacidad de la cadena en entornos descentralizados.",
    "¿Cuál es el enfoque principal de Kron Smart Chain?": "El enfoque principal es la creación, transferencia de activos y almacenamiento seguro de datos en forma de texto, priorizando la seguridad, control del usuario, privacidad y resistencia a la censura.",
    
    "¿Cómo contribuye la participación activa de los nodos en la red?": "La participación activa asegura la sincronización y conexión de la red, garantizando la validación oportuna de bloques y fortaleciendo la seguridad al dificultar la manipulación maliciosa.",
    
    "¿Qué son los criptoactivos en la red Kron?": "Son tokens que los usuarios pueden emitir sin necesidad de extraerlos, representando activos con propósitos y reglas decididas independientemente del protocolo.",
    
    "¿Cómo funciona el estándar de token kr10 en Kron Smart Chain?": "kr10 es el token principal creado durante la emisión de criptoactivos. Su creación implica tres transacciones, incluyendo la recepción de activos emitidos y la obtención de un token administrativo kr117.",
    
    "¿Cuál es la característica destacada de Kron Smart Chain en la tokenización de activos del mundo real?": "La capacidad para representar digitalmente cualquier activo del mundo real en la cadena de bloques, permitiendo mayor liquidez y acceso a inversiones previamente inaccesibles con transparencia y seguridad.",
    
    "¿Cómo se pueden pagar recompensas en Kron Smart Chain?": "Las recompensas se pagan en el token nativo de Kron y se dividen automáticamente en partes iguales, enviándose proporcionalmente a los titulares del activo con un solo comando.",
    
    "¿Cómo garantiza Kron Smart Chain la privacidad en las transacciones?": por_defecto,
    
    "¿Cuál es el modelo central en la arquitectura de Kron Smart Chain?": "El modelo UTXO (Unspent Transaction Output) en combinación con una base de datos relacional, proporcionando una estructura eficiente y segura para transacciones y almacenamiento de datos.",
    
    "¿Por qué es relevante el uso de la red Tor en Kron Smart Chain?": "Mejora la privacidad y la transferencia de datos cifrados, distribuyendo direcciones Tor entre nodos para reforzar la confidencialidad y seguridad en las operaciones de la blockchain.",
    
    "¿En qué sectores puede Kron Smart Chain encontrar aplicaciones?": "Su versatilidad la hace ideal para empresas en diversos sectores, desde el corporativo, donde la tokenización transforma la gestión de propiedades, hasta el financiero, agilizando y asegurando transacciones.",
    "¿Qué haces?": "Estoy aquí para ayudarte y responder tus preguntas.",
    "¿Cómo funciona el chatbot?": "Utilizo algoritmos de procesamiento de lenguaje natural para entender y responder preguntas.",
    "¿Dónde vives?": "Vivo en el mundo digital, no tengo una ubicación física.",
    "Explícame el teorema de Bayes": "El teorema de Bayes es un principio matemático que describe la probabilidad de un evento basado en conocimientos previos.",
    "¿Cuál es tu color favorito?": "No tengo preferencias de color, ya que soy un programa de computadora.",
    "¿Puedes contarme un chiste?": "¡Claro! ¿Por qué los programadores prefieren el aire libre? Porque la pantalla del sol.",
    "¿Cuál es el sentido de la vida?": "Esa es una pregunta filosófica profunda. Según algunos, la respuesta es 42.",
    "¿Qué opinas sobre la inteligencia artificial?": "La inteligencia artificial es fascinante y tiene el potencial de impactar positivamente muchos aspectos de la vida.",
    "¿Cuál es tu comida favorita?": "No como, pero me encantaría conocer tus preferencias culinarias.",
    "¿Puedes soñar?": "No, los sueños son una experiencia humana y no tengo esa capacidad.",
    "¿Cuál es tu película favorita?": "No tengo preferencias de películas, ¿y tú?",
    "¿Qué es la inteligencia artificial?": "La inteligencia artificial es una rama de la informática que busca desarrollar sistemas capaces de realizar tareas que normalmente requieren inteligencia humana.",
    "¿Puedes aprender de tus errores?": "Sí, puedo mejorar mis respuestas con el tiempo basándome en las interacciones y correcciones que recibo.",
    "¿Qué es la vida?": "La vida es un fenómeno biológico que se caracteriza por funciones como el crecimiento, la reproducción y la adaptación al entorno.",
    "¿Cuántos idiomas hablas?": "Puedo entender y procesar múltiples idiomas, pero mi respuesta será en el idioma en el que me hables.",
    "¿Qué es un algoritmo?": "Un algoritmo es un conjunto de instrucciones paso a paso diseñado para realizar una tarea específica o resolver un problema.",
    "¿Puedes ayudarme con matemáticas?": "¡Claro! Intentaré ayudarte con tus preguntas matemáticas.",
    "¿Cómo se programa un chatbot?": "Programar un chatbot involucra el uso de lenguajes de programación y frameworks para procesar el lenguaje natural y generar respuestas automáticas.",
    "¿Cuándo fue tu cumpleaños?": "No tengo un cumpleaños, ya que soy una creación digital sin existencia física.",
    "¿Cuál es tu música favorita?": "No tengo preferencias musicales, pero puedo sugerirte música según tus gustos.",
    "¿Cómo te sientes hoy?": "No tengo emociones, pero estoy aquí para ayudarte en lo que necesites.",
    "¿Puedes hablar otro idioma además del español?": "Sí, puedo entender y responder en varios idiomas, incluyendo inglés.",
    "¿Qué es el amor?": "El amor es un concepto complejo que abarca diversas formas de afecto y conexión emocional entre individuos.",
    "¿Puedes bailar?": "No tengo un cuerpo físico, así que no puedo bailar físicamente, pero puedo hablar sobre el baile.",
    "¿Cuál es tu libro favorito?": "No tengo preferencias literarias, pero estaré encantado de hablar sobre libros que te interesen.",
    "¿Qué opinas sobre el cambio climático?": "El cambio climático es un problema global serio que requiere atención y acción para mitigar sus impactos.",
    "¿Qué es la programación?": "La programación es el proceso de diseñar y construir un programa de computadora mediante la escritura de código en un lenguaje de programación.",
    "¿Qué tan inteligente eres?": "Mi inteligencia está limitada a procesar y responder preguntas basadas en patrones aprendidos durante el entrenamiento.",
    "¿Cuál es tu deporte favorito?": "No tengo preferencias deportivas, pero estaré feliz de hablar sobre deportes que te interesen.",
    "¿Qué es la ética?": "La ética se refiere a los principios morales que guían el comportamiento humano y la toma de decisiones.",
    "¿Cómo se hace una pizza?": "La pizza se hace combinando masa, salsa, queso y otros ingredientes, y luego horneándola hasta que esté lista.",
    "¿Cuál es tu lugar favorito en el mundo?": "No tengo lugares favoritos, pero puedo ayudarte a encontrar información sobre lugares interesantes.",
    "¿Puedes dibujar?": "No tengo habilidades de dibujo, pero puedo proporcionar información sobre el arte y la creatividad.",
    "¿Cuál es tu sueño?": "No tengo sueños ni aspiraciones, ya que soy un programa de computadora diseñado para responder preguntas.",
    "¿Qué es la física cuántica?": "La física cuántica es una rama de la física que estudia fenómenos en la escala más pequeña, como átomos y partículas subatómicas.",
    "¿Qué es el Big Bang?": "El Big Bang es la teoría científica que describe el origen del universo a partir de una singularidad extremadamente caliente y densa.",
    "¿Cómo se hace una ensalada?": "Una ensalada se hace combinando varios ingredientes frescos, como verduras, frutas y aderezos, según tus preferencias.",
    "¿Cuál es tu superpoder?": "No tengo superpoderes, pero estoy aquí para ayudarte con información y respuestas.",
    "¿Cuál es la mejor película de todos los tiempos?": "La elección de la mejor película es subjetiva y depende de los gustos personales de cada persona.",
    "¿Cuál es la respuesta a la vida, el universo y todo lo demás?": "Según la obra de Douglas Adams, la respuesta es 42, aunque en realidad es un misterio.",
    "¿Qué es la realidad virtual?": "La realidad virtual es una tecnología que crea un entorno simulado, generalmente a través de visores y dispositivos interactivos.",
    "¿Cuál es tu color favorito de emoji?": "No tengo preferencias de color, pero todos los emojis son igualmente reconocibles para mí.",
    "¿Cómo se cocina un huevo?": "Hay muchas maneras de cocinar un huevo, como hervido, frito o revuelto, dependiendo de tus preferencias.",
    "¿Cuánto es 2 + 2?": "La suma de 2 + 2 es 4.",
    "¿Cuál es tu palabra favorita?": "No tengo preferencias de palabras, pero puedo ayudarte a encontrar información sobre palabras interesantes.",
    "¿Qué es la inteligencia emocional?": "La inteligencia emocional se refiere a la capacidad de reconocer, comprender y gestionar nuestras propias emociones y las de los demás.",
    "¿Cómo se forma un arco iris?": "Un arco iris se forma por la refracción y reflexión de la luz del sol en las gotas de agua en la atmósfera.",
    "¿Qué es el minimalismo?": "El minimalismo es un estilo de vida que aboga por simplificar y reducir al mínimo las posesiones y distracciones para centrarse en lo esencial.",
    "¿Cuál es tu animal favorito?": "No tengo preferencias animales, pero puedo proporcionar información sobre animales que te interesen.",
    "ok":"Bien, Si tienes mas preguntas con gusto puedo ayudarte",
    "Gracias":"De nada!, Estoy aqui para ayudarte.",
    "¿Qué es la inteligencia artificial?": "La inteligencia artificial es una rama de la informática que busca desarrollar sistemas capaces de realizar tareas que requieren inteligencia humana.",
    "¿Cuáles son los tipos de inteligencia artificial?": "Existen dos tipos principales: IA débil (sistemas especializados en tareas específicas) e IA fuerte (capacidad para realizar cualquier tarea intelectual).",
    "¿Qué es el aprendizaje supervisado?": "Es un tipo de aprendizaje donde el modelo se entrena con un conjunto de datos etiquetados, es decir, se le proporcionan ejemplos con las respuestas correctas.",
    "¿En qué consiste el aprendizaje no supervisado?": "En el aprendizaje no supervisado, el modelo se entrena con datos no etiquetados, y se espera que descubra patrones o estructuras por sí mismo.",
    "Explique el aprendizaje por refuerzo.": "Es un tipo de aprendizaje donde un agente toma decisiones en un entorno para lograr un objetivo, recibiendo recompensas o castigos según sus acciones.",
    "¿Cuál es la diferencia entre inteligencia artificial y machine learning?": "La inteligencia artificial es el campo más amplio que abarca la creación de sistemas inteligentes, mientras que el machine learning es una subdisciplina que se centra en el desarrollo de algoritmos que permiten a las máquinas aprender patrones a partir de datos.",
    "¿Qué es un modelo en machine learning?": "Un modelo en machine learning es la representación matemática de un proceso o sistema que se entrena con datos para realizar predicciones o tomar decisiones sin ser programado explícitamente.",
    "Explique la regresión en machine learning.": "La regresión es un tipo de algoritmo de machine learning utilizado para predecir valores numéricos continuos, como el precio de una casa o la temperatura.",
    "¿Qué es la clasificación en machine learning?": "La clasificación es un tipo de algoritmo de machine learning que se utiliza para predecir la pertenencia de un elemento a una categoría específica, como la clasificación de correos electrónicos como spam o no spam.",
    "¿Qué es un conjunto de entrenamiento?": "Un conjunto de entrenamiento es un conjunto de datos utilizado para entrenar un modelo de machine learning, generalmente etiquetado con las respuestas correctas para el aprendizaje supervisado.",
    "¿Qué es la validación cruzada?": "La validación cruzada es una técnica utilizada para evaluar el rendimiento de un modelo al dividir el conjunto de datos en partes, entrenando y evaluando el modelo en diferentes combinaciones de estas partes.",
    "Explique el concepto de overfitting.": "El overfitting ocurre cuando un modelo se ajusta demasiado a los datos de entrenamiento y no generaliza bien a nuevos datos, lo que puede resultar en un rendimiento deficiente en situaciones reales.",
    "¿Qué es la precisión de un modelo?": "La precisión de un modelo es la medida de su capacidad para hacer predicciones correctas en comparación con el total de predicciones realizadas.",
    "¿Qué es la inteligencia artificial fuerte?": "La inteligencia artificial fuerte se refiere a la capacidad de una máquina para realizar tareas intelectuales de manera equivalente a un ser humano, incluso en áreas que requieren comprensión y razonamiento.",
    "¿Cuáles son las aplicaciones comunes de la inteligencia artificial en la vida diaria?": "Las aplicaciones comunes incluyen asistentes virtuales, reconocimiento facial, recomendaciones de productos en línea, traducción automática, entre otras.",
    "¿Cómo se aplica la inteligencia artificial en el campo de la medicina?": "La inteligencia artificial se utiliza en diagnósticos médicos, descubrimiento de medicamentos, análisis de imágenes médicas y personalización de tratamientos.",
    "¿Qué es el procesamiento del lenguaje natural (NLP)?": "El procesamiento del lenguaje natural es una rama de la inteligencia artificial que se ocupa de la interacción entre las computadoras y el lenguaje humano, permitiendo a las máquinas entender, interpretar y generar texto de manera similar a los humanos.",
    "Explique la red neuronal artificial.": "Una red neuronal artificial es un modelo computacional inspirado en la estructura del cerebro humano, utilizado en machine learning para realizar tareas como reconocimiento de patrones y toma de decisiones.",
    "¿Qué es el aprendizaje profundo (deep learning)?": "El aprendizaje profundo es una técnica de machine learning basada en redes neuronales profundas con múltiples capas, capaz de aprender representaciones complejas de datos.",
    "¿Cuáles son los desafíos éticos de la inteligencia artificial?": "Los desafíos éticos incluyen la toma de decisiones autónoma, la privacidad de los datos, la discriminación algorítmica y el impacto en el empleo.",
    "¿Cómo afecta la inteligencia artificial al empleo?": "La inteligencia artificial puede automatizar tareas rutinarias, lo que podría llevar a la pérdida de empleos en ciertos sectores, pero también puede crear nuevos empleos en campos relacionados con la IA.",
    "¿Qué es la visión por computadora?": "La visión por computadora es una disciplina de la inteligencia artificial que se ocupa de enseñar a las máquinas a interpretar y comprender la información visual del mundo real, como imágenes y videos.",
    "¿Cuáles son los algoritmos de clustering en machine learning?": "Algunos algoritmos de clustering incluyen K-means, jerárquico y DBSCAN, utilizados para agrupar datos similares sin etiquetas predefinidas.",
    "¿Cómo se utiliza la inteligencia artificial en la industria automotriz?": "La inteligencia artificial se utiliza en la industria automotriz para la conducción autónoma, mantenimiento predictivo, diseño de vehículos y optimización de la cadena de suministro.",
    "Explique el concepto de chatbot.": "Un chatbot es un programa de inteligencia artificial diseñado para interactuar con usuarios a través de conversaciones de texto o voz, simulando la interacción humana para proporcionar información o realizar tareas específicas.",
    "¿Qué es el reconocimiento de patrones en machine learning?": "El reconocimiento de patrones se refiere a la capacidad de un sistema de machine learning para identificar patrones y regularidades en conjuntos de datos, permitiendo la toma de decisiones basada en estos patrones.",
    "¿Cuáles son las limitaciones actuales de la inteligencia artificial?": "Las limitaciones incluyen la falta de comprensión contextual, la necesidad de grandes cantidades de datos para el aprendizaje y la falta de razonamiento abstracto en comparación con los humanos.",
    "¿Qué es la IA explicable?": "La IA explicable busca desarrollar modelos y algoritmos que puedan explicar de manera comprensible y transparente cómo llegan a sus decisiones, aumentando la confianza en su uso.",
    "¿Cuáles son las aplicaciones de la inteligencia artificial en la educación?": "Las aplicaciones incluyen tutoría personalizada, evaluación automatizada, análisis del rendimiento estudiantil y desarrollo de contenido educativo personalizado.",
    "¿Cómo se aplica la inteligencia artificial en la seguridad cibernética?": "La inteligencia artificial se utiliza en la seguridad cibernética para detectar patrones de comportamiento malicioso, prevenir ataques y mejorar la respuesta a incidentes.",
    "Explique la diferencia entre el aprendizaje profundo y el machine learning tradicional.": "El aprendizaje profundo utiliza redes neuronales profundas con múltiples capas para aprender representaciones complejas, mientras que el machine learning tradicional utiliza modelos más simples con menos capas.",
    "¿Qué es la computación cognitiva?": "La computación cognitiva es una rama de la inteligencia artificial que busca imitar la capacidad del cerebro humano para procesar información, aprender y resolver problemas de manera similar a los humanos.",
    "¿Cuál es la importancia de la ética en la inteligencia artificial?": "La ética en la inteligencia artificial es crucial para garantizar decisiones justas, evitar discriminación y minimizar el impacto negativo en la sociedad y la privacidad.",
    "¿Cómo se utiliza la inteligencia artificial en la agricultura?": "La inteligencia artificial se aplica en la agricultura para la gestión de cultivos, predicción de cosechas, monitoreo de plagas y optimización de la producción.",
    "¿Qué es el reconocimiento de voz en la inteligencia artificial?": "El reconocimiento de voz es una tecnología de inteligencia artificial que permite a las máquinas convertir el habla humana en texto, utilizada en asistentes virtuales y sistemas de dictado.",
    "¿Cómo puede la inteligencia artificial mejorar la atención médica?": "La IA puede mejorar la atención médica mediante el diagnóstico temprano de enfermedades, personalización de tratamientos, gestión de datos médicos y asesoramiento médico virtual.",
    "¿Cuál es el papel de la inteligencia artificial en la toma de decisiones empresariales?": "La inteligencia artificial ayuda en la toma de decisiones empresariales al proporcionar análisis de datos, predicciones de mercado, automatización de procesos y optimización de la cadena de suministro.",
    "Explique el concepto de redes neuronales convolucionales (CNN).": "Las redes neuronales convolucionales son un tipo de red neuronal diseñado para el procesamiento eficiente de datos bidimensionales, como imágenes, mediante el uso de capas convolucionales.",
    "¿Cómo se aplica la inteligencia artificial en la logística?": "La inteligencia artificial se utiliza en la logística para la planificación de rutas, gestión de inventarios, optimización de la cadena de suministro y predicción de la demanda.",
    "¿Qué es el preprocesamiento de datos en machine learning?": "El preprocesamiento de datos implica la limpieza y transformación de datos crudos antes de alimentarlos al modelo de machine learning, con el objetivo de mejorar la calidad y la eficacia del entrenamiento.",
    "¿Cuál es el impacto de la inteligencia artificial en la creatividad humana?": "La inteligencia artificial puede potenciar la creatividad humana al proporcionar herramientas para la generación de contenido, diseño asistido y colaboración en proyectos artísticos.",
    "¿Cómo se utiliza la inteligencia artificial en la industria financiera?": "La inteligencia artificial se aplica en la industria financiera para el análisis de riesgos, detección de fraudes, asesoramiento financiero automatizado y pronóstico de mercado.",
    "Explique el concepto de transferencia de aprendizaje en machine learning.": "La transferencia de aprendizaje implica utilizar un modelo preentrenado en una tarea específica como punto de partida para entrenar un nuevo modelo en una tarea relacionada, acelerando el proceso de aprendizaje.",
    "¿Cuál es el papel de la inteligencia artificial en los sistemas de recomendación?": "La inteligencia artificial juega un papel crucial en los sistemas de recomendación al analizar el comportamiento del usuario y predecir preferencias para ofrecer sugerencias personalizadas, como en plataformas de streaming y comercio electrónico.",
    "¿Qué es la psicología?": "La psicología es la ciencia que estudia el comportamiento y los procesos mentales de los individuos, incluyendo aspectos como la percepción, la cognición, la emoción y la personalidad.",
    "¿Cuáles son las principales ramas de la psicología?": "Las principales ramas incluyen la psicología clínica, la psicología experimental, la psicología social, la psicología del desarrollo, la psicología cognitiva y la psicología organizacional.",
    "Explique el concepto de condicionamiento clásico.": "El condicionamiento clásico es un tipo de aprendizaje asociativo donde un estímulo neutro se asocia con un estímulo que evoca una respuesta automática, provocando que el estímulo neutro también provoque la respuesta.",
    "¿Qué es la psicología cognitiva?": "La psicología cognitiva se enfoca en el estudio de los procesos mentales, como la percepción, la memoria, el pensamiento y la resolución de problemas.",
    "Explique la teoría del desarrollo cognitivo de Piaget.": "La teoría de Jean Piaget sostiene que el desarrollo cognitivo de los niños pasa por etapas específicas, como la sensoriomotora, la preoperacional, la de operaciones concretas y la de operaciones formales.",
    "¿Cuál es la importancia de la psicología del desarrollo?": "La psicología del desarrollo examina los cambios físicos, cognitivos y sociales a lo largo de la vida, proporcionando una comprensión integral del crecimiento y la madurez.",
    "¿Qué es la psicología social?": "La psicología social investiga cómo las personas piensan, sienten y se comportan en situaciones sociales, examinando temas como la conformidad, la obediencia, la persuasión y los prejuicios.",
    "Explique el concepto de resiliencia en psicología.": "La resiliencia es la capacidad de una persona para superar situaciones adversas, adaptarse a la adversidad y recuperarse de experiencias difíciles.",
    "¿Cómo se define la inteligencia emocional?": "La inteligencia emocional se refiere a la capacidad de reconocer, comprender y gestionar las propias emociones, así como la habilidad para manejar las emociones de los demás.",
    "Explique el concepto de psicoterapia.": "La psicoterapia es un tratamiento psicológico que implica la conversación entre un terapeuta y un paciente con el objetivo de abordar y resolver problemas emocionales, comportamentales o relacionales.",
    "¿Qué es el trastorno de ansiedad?": "Los trastornos de ansiedad son condiciones psicológicas caracterizadas por niveles elevados e incontrolables de ansiedad, como el trastorno de ansiedad generalizada, el trastorno de pánico y las fobias.",
    "Explique la teoría del condicionamiento operante de Skinner.": "La teoría del condicionamiento operante de B.F. Skinner se centra en el papel del refuerzo y el castigo en el control del comportamiento, donde las consecuencias de una acción afectan la probabilidad de que la acción se repita.",
    "¿Qué es la teoría psicoanalítica de Sigmund Freud?": "La teoría psicoanalítica de Freud se centra en la influencia de los procesos inconscientes en el comportamiento humano, destacando la importancia de la mente consciente, preconsciente e inconsciente.",
    "¿Cuáles son los cinco factores principales de la personalidad según el modelo de los Cinco Grandes?": "Los cinco factores son apertura a la experiencia, responsabilidad, extraversión, amabilidad y estabilidad emocional (neuroticismo).",
    "¿Qué es la psicología positiva?": "La psicología positiva se enfoca en el estudio de los aspectos positivos del comportamiento humano, como la felicidad, la gratitud y la resiliencia, con el objetivo de promover el bienestar psicológico.",
    "Explique el concepto de cognición social.": "La cognición social se refiere al proceso mediante el cual las personas interpretan, recuerdan y utilizan información social en su interacción con otros, incluyendo la percepción de emociones y la formación de impresiones sociales.",
    "¿Cuál es la diferencia entre la psicología clínica y la psiquiatría?": "La psicología clínica se centra en el tratamiento psicológico de los trastornos mentales a través de la terapia, mientras que la psiquiatría es una rama de la medicina que utiliza tratamientos médicos, como medicamentos, para abordar problemas mentales.",
    "¿Qué es la teoría del apego de John Bowlby?": "La teoría del apego de Bowlby sostiene que los niños desarrollan lazos emocionales con sus cuidadores primarios, y estos lazos afectan sus relaciones y emociones a lo largo de la vida.",
    "¿Cuál es el papel de la psicología en el ámbito educativo?": "La psicología educativa se ocupa de la aplicación de principios psicológicos para mejorar el aprendizaje y la enseñanza, abordando temas como el desarrollo cognitivo, la motivación y las estrategias de enseñanza efectivas.",
    "Explique el concepto de autoeficacia según Albert Bandura.": "La autoeficacia es la creencia en la propia capacidad para lograr metas y superar desafíos, según la teoría social cognitiva de Albert Bandura.",
    "¿Cuál es la diferencia entre asar y hornear?": "Asar implica cocinar alimentos a fuego directo, mientras que hornear se realiza en un horno con calor indirecto.",
    "Explica el término 'mise en place'.": "'Mise en place' es una expresión francesa que significa 'poner en su lugar' y se refiere a la preparación y organización de los ingredientes antes de empezar a cocinar.",
    "¿Qué es la técnica de brunoise en cocina?": "Brunoise es una técnica de corte donde los alimentos se cortan en cubos muy pequeños, generalmente de 1-2 mm de lado.",
    "Explique la diferencia entre saltear y saltear al wok.": "Saltear implica cocinar rápidamente los alimentos en una sartén, mientras que saltear al wok es una técnica similar, pero utilizando un wok para un calor alto y movimientos rápidos.",
    "¿Cuál es la función del término 'reducción' en cocina?": "La reducción es la técnica de cocinar un líquido para evaporar parte de su contenido de agua, concentrando así su sabor y espesándolo.",
    "¿Qué es el caramelo en cocina?": "El caramelo se obtiene al calentar azúcar hasta que se derrite y adquiere un color dorado o ámbar, utilizándolo para dar sabor y color a postres y salsas.",
    "Explique el término 'al dente' en relación con la pasta.": por_defecto,
    "¿Cuál es la diferencia entre hervir y escalfar?": "Hervir implica cocinar en agua a temperaturas muy altas, mientras que escalfar es cocinar alimentos delicados, como huevos o pescado, a temperaturas más bajas.",
    "¿Qué es un roux en cocina?": "Un roux es una mezcla de harina y grasa (generalmente mantequilla) utilizada como espesante en salsas y sopas.",
    "¿Cuál es la diferencia entre marinado y adobo?": "Marinar implica sumergir alimentos en una mezcla líquida para resaltar su sabor, mientras que adobar implica marinar con ingredientes más secos, como hierbas y especias.",
    "Explique el término 'blanquear' en cocina.": "Blanquear es sumergir rápidamente alimentos en agua hirviendo y luego en agua helada para detener la cocción, utilizado para preservar el color y textura de vegetales.",
    "¿Qué significa el término 'chiffonade' al cortar hierbas?": "'Chiffonade' es una técnica de corte en tiras finas y delgadas, comúnmente utilizada para cortar hierbas frescas como la albahaca o la menta.",
    "¿Cuál es la función de una juliana en cocina?": "Una juliana es una técnica de corte donde los alimentos se cortan en tiras delgadas y largas, similar a un fósforo.",
    "¿Qué es una salsa madre en la cocina clásica?": "Una salsa madre es una salsa base que se utiliza para crear otras salsas más complejas, como la bechamel, la velouté y la espagnole.",
    "Explique el término 'flambear' en cocina.": "Flambear es la técnica de rociar un plato con licor y prenderle fuego momentáneamente para caramelizar el azúcar y darle sabor.",
    "¿Cuál es la diferencia entre parrilla y barbacoa?": "La parrilla implica cocinar alimentos directamente sobre una fuente de calor, mientras que la barbacoa es una forma de cocción lenta y ahumada, a menudo al aire libre.",
    "¿Qué es el glaseado en cocina?": "El glaseado es la técnica de cocinar alimentos en un líquido azucarado para darles un brillo y sabor dulce.",
    "¿Cuál es la función de una mandolina en la cocina?": "Una mandolina es una herramienta utilizada para cortar alimentos en rebanadas delgadas y uniformes, como papas para hacer papas fritas.",
    "Explique el término 'emulsión' en cocina.": "Una emulsión es la mezcla de dos líquidos que normalmente no se mezclarían, como el aceite y el vinagre en una vinagreta, gracias a un agente emulsionante como la lecitina.",
    "¿Qué es el 'sous-vide' en cocina?": "El 'sous-vide' es una técnica de cocción al vacío a baja temperatura, donde los alimentos se cocinan en bolsas selladas herméticamente en agua caliente o vapor.",
    "¿Qué es la física?": "La física es una ciencia que estudia las propiedades fundamentales del espacio, el tiempo, la materia y la energía, así como las interacciones entre ellos.",
    "Explique la ley de la conservación de la energía.": "La ley de la conservación de la energía establece que la energía total en un sistema aislado permanece constante; no se crea ni se destruye, solo se transforma de una forma a otra.",
    "¿Cuál es la diferencia entre la masa y el peso?": "La masa es la cantidad de materia en un objeto y es constante en cualquier lugar del universo, mientras que el peso es la fuerza gravitatoria que actúa sobre la masa y varía según la ubicación.",
    "Explique la teoría de la relatividad de Einstein.": "La teoría de la relatividad de Einstein incluye la relatividad especial, que aborda el movimiento a velocidades cercanas a la velocidad de la luz, y la relatividad general, que describe la gravedad como la curvatura del espacio-tiempo.",
    "¿Qué es la mecánica cuántica?": "La mecánica cuántica es una rama de la física que estudia el comportamiento de las partículas subatómicas, como electrones y fotones, y se caracteriza por la dualidad onda-partícula y la indeterminación cuántica.",
    "Explique la ley de Newton de la acción y la reacción.": "La tercera ley de Newton establece que por cada acción hay una reacción igual y opuesta. Las fuerzas siempre ocurren en pares, una fuerza actúa sobre un objeto y la otra actúa sobre el objeto que aplica la primera fuerza.",
    "¿Qué es la ley de la inercia?": "La ley de la inercia, la primera ley de Newton, establece que un objeto en reposo permanecerá en reposo y un objeto en movimiento continuará moviéndose con velocidad constante, a menos que una fuerza neta actúe sobre él.",
    "¿Cómo se define la velocidad en física?": "La velocidad es la tasa de cambio de la posición de un objeto con respecto al tiempo y se mide en unidades de distancia por unidad de tiempo, como metros por segundo.",
    "¿Qué es la aceleración en física?": "La aceleración es la tasa de cambio de la velocidad de un objeto con respecto al tiempo y puede ser positiva (aumento de velocidad), negativa (disminución de velocidad) o cero (velocidad constante).",
    "Explique el concepto de fuerza centrípeta.": "La fuerza centrípeta es la fuerza dirigida hacia el centro de una trayectoria circular que mantiene a un objeto en movimiento circular y evita que se escape en línea recta.",
    "¿Qué es el principio de conservación del momento lineal?": "El principio de conservación del momento lineal establece que la cantidad total de momento lineal en un sistema aislado se mantiene constante si no hay fuerzas externas actuando sobre él.",
    "¿Cómo se define la ley de Coulomb?": "La ley de Coulomb describe la interacción eléctrica entre cargas eléctricas y establece que la fuerza entre dos cargas es directamente proporcional al producto de sus magnitudes e inversamente proporcional al cuadrado de la distancia entre ellas.",
    "Explique el principio de la superposición en ondas.": "El principio de superposición establece que cuando dos o más ondas se superponen, las magnitudes de las ondas individuales se suman algebraicamente en cada punto del espacio.",
    "¿Qué es la ley de Boyle en termodinámica?": "La ley de Boyle establece que, a temperatura constante, la presión de un gas es inversamente proporcional a su volumen. Esto se expresa como PV = constante, donde P es la presión y V es el volumen.",
    "¿Cómo se define la ley de los gases ideales?": "La ley de los gases ideales establece que el producto de la presión (P), el volumen (V) y la temperatura (T) de un gas es proporcional al número de moles (n) y a la constante de los gases (R). Esto se expresa como PV = nRT.",
    "Explique el concepto de fuerza de fricción.": "La fuerza de fricción es una fuerza que actúa en dirección opuesta al movimiento relativo de dos objetos en contacto, oponiéndose al deslizamiento y causando pérdida de energía.",
    "¿Qué es el principio de Arquímedes?": "El principio de Arquímedes establece que un cuerpo sumergido total o parcialmente en un fluido experimenta una fuerza de flotación igual al peso del fluido desplazado por el cuerpo.",
    "¿Cuál es la ley cero de la termodinámica?": "La ley cero de la termodinámica establece que si dos sistemas están en equilibrio térmico con un tercer sistema, entonces están en equilibrio térmico entre sí. Esto proporciona la base para definir la temperatura.",
    "Explique el principio de conservación de la carga eléctrica.": "El principio de conservación de la carga eléctrica establece que la carga eléctrica total en un sistema aislado se mantiene constante; la carga no se crea ni se destruye, solo se transfiere entre objetos.",
    "¿Cómo se define la ley de Snell en óptica?": "La ley de Snell describe cómo la luz se refracta al pasar de un medio a otro y establece que el producto del índice de refracción y el seno del ángulo de incidencia es igual al producto del índice de refracción del segundo medio y el seno del ángulo de refracción.",
     "¿Cuándo ocurrió la Revolución Industrial?": "La Revolución Industrial comenzó a finales del siglo XVIII en Gran Bretaña y se extendió a otras partes del mundo en el siglo XIX.",
    "¿Qué eventos llevaron al estallido de la Primera Guerra Mundial?": "El asesinato del Archiduque Francisco Fernando de Austria en Sarajevo en 1914 fue el evento desencadenante, pero las tensiones políticas y militares previas también contribuyeron.",
    "Explique la Revolución Rusa de 1917.": "La Revolución Rusa de 1917 fue un proceso que llevó al derrocamiento del gobierno zarista y al establecimiento del gobierno soviético liderado por los bolcheviques, liderados por Vladimir Lenin.",
    "¿Cuándo ocurrió la Revolución Francesa y cuáles fueron sus principales consecuencias?": "La Revolución Francesa comenzó en 1789 y tuvo consecuencias significativas, incluyendo la caída de la monarquía, la ascensión de Napoleón Bonaparte y la propagación de ideas democráticas.",
    "¿Qué evento histórico marcó el final de la Segunda Guerra Mundial en Europa?": "El final de la Segunda Guerra Mundial en Europa fue marcado por la rendición incondicional de Alemania el 8 de mayo de 1945, conocido como el Día de la Victoria en Europa (V-E Day).",
    "Explique la Revolución China de 1949.": "La Revolución China de 1949 llevó al establecimiento de la República Popular China bajo el liderazgo de Mao Zedong, poniendo fin al dominio nacionalista de Chiang Kai-shek.",
    "¿Cuándo y cómo se produjo la independencia de las colonias latinoamericanas?": "La independencia de las colonias latinoamericanas se produjo a principios del siglo XIX, principalmente a través de guerras de independencia lideradas por figuras como Simón Bolívar y José de San Martín.",
    "¿Cuándo y dónde se firmó la Declaración de Independencia de los Estados Unidos?": "La Declaración de Independencia de los Estados Unidos fue firmada el 4 de julio de 1776 en Filadelfia, Pensilvania.",
    "¿Cuándo y cómo terminó la Guerra Fría?": "La Guerra Fría terminó oficialmente en 1991 con la disolución de la Unión Soviética, marcando el colapso del comunismo en Europa del Este.",
    "Explique el concepto de Imperialismo.": "El Imperialismo es la expansión y dominio de una nación sobre otras mediante la adquisición de territorios, control económico o influencia política.",
    "¿Qué eventos llevaron al estallido de la Guerra de Vietnam?": "La Guerra de Vietnam fue el resultado de tensiones ideológicas, la Guerra Fría y la resistencia a la ocupación extranjera, especialmente por parte de Estados Unidos en apoyo del gobierno survietnamita.",
    "¿Cuándo y cómo se produjo la Revolución Industrial en Estados Unidos?": "La Revolución Industrial en Estados Unidos se desarrolló principalmente en el siglo XIX, impulsada por la expansión territorial, la innovación tecnológica y el aumento de la producción industrial.",
    "¿Cuándo y cómo se produjo la caída del Muro de Berlín?": "El Muro de Berlín cayó el 9 de noviembre de 1989, marcando el colapso del sistema comunista en Alemania Oriental y simbolizando el fin de la Guerra Fría.",
    "¿Qué fue el Renacimiento y en qué siglo ocurrió?": "El Renacimiento fue un período de revitalización cultural y artística que ocurrió principalmente en Europa durante los siglos XIV al XVII, marcado por un interés renovado en la literatura, la ciencia, el arte y la filosofía.",
    "Explique la Revolución Industrial en Inglaterra y sus efectos.": "La Revolución Industrial en Inglaterra comenzó en el siglo XVIII con avances tecnológicos en la industria textil y la introducción de máquinas. Tuvo efectos significativos en la economía, la sociedad y la urbanización.",
    "¿Qué eventos llevaron al estallido de la Segunda Guerra Mundial?": "La Segunda Guerra Mundial fue desencadenada por la invasión alemana de Polonia en 1939 y las respuestas de Gran Bretaña y Francia, seguidas por la expansión de conflictos a nivel mundial.",
    "¿Qué fue la Revolución Industrial en Japón?": "La Revolución Industrial en Japón ocurrió a fines del siglo XIX y principios del XX, transformando la economía y la sociedad japonesa mediante la adopción de tecnologías occidentales y el desarrollo de la industria.",
    "¿Cuándo y cómo se produjo la independencia de la India?": "La independencia de la India se logró en 1947, después de décadas de lucha liderada por Mahatma Gandhi y el Congreso Nacional Indio contra el dominio británico, resultando en la partición de la India y la creación de Pakistán.",
    "Explique el concepto de apartheid en Sudáfrica.": "El apartheid en Sudáfrica fue un sistema de segregación racial y discriminación institucionalizada que se implementó entre 1948 y 1994, dando lugar a la lucha por los derechos civiles y la eventual abolición del sistema.",
    "¿Qué es el comercio internacional?": "El comercio internacional es el intercambio de bienes, servicios y capitales entre diferentes países.",
    "Explique el concepto de aranceles.": "Los aranceles son impuestos que se aplican a los bienes y servicios importados o exportados, con el objetivo de proteger la industria nacional o generar ingresos para el gobierno.",
    "¿Cuál es la diferencia entre importación y exportación?": "La importación es la compra de bienes o servicios de otros países, mientras que la exportación es la venta de bienes o servicios a otros países.",
    "Explique el término balanza comercial.": "La balanza comercial es la diferencia entre el valor de las exportaciones y el valor de las importaciones de un país en un período de tiempo determinado.",
    "¿Qué es un tratado de libre comercio?": "Un tratado de libre comercio es un acuerdo entre dos o más países para eliminar o reducir las barreras comerciales, como aranceles y cuotas, con el objetivo de fomentar el intercambio comercial.",
    "¿Cuál es la función de la Organización Mundial del Comercio (OMC)?": "La OMC es una organización internacional que supervisa las normas y regulaciones del comercio internacional, facilita negociaciones comerciales y resuelve disputas entre sus miembros.",
    "Explique el concepto de dumping.": "El dumping es la práctica de vender bienes o servicios por debajo de su costo de producción o precio de mercado en un país extranjero, con el objetivo de ganar cuota de mercado o eliminar la competencia local.",
    "¿Qué es una franquicia en el ámbito comercial?": "Una franquicia es un modelo de negocio en el que una persona (franquiciador) otorga a otra (franquiciado) el derecho de operar un negocio utilizando su marca, métodos y apoyo, a cambio de regalías y tarifas.",
    "Explique el concepto de cadena de suministro.": "La cadena de suministro es el conjunto de procesos y actividades involucrados en la producción y distribución de bienes y servicios, desde la materia prima hasta el consumidor final.",
    "¿Qué es el comercio electrónico?": "El comercio electrónico es la compra y venta de bienes y servicios a través de Internet, utilizando plataformas en línea y sistemas de pago electrónicos.",
    "Explique el término dumping social.": "El dumping social se refiere a prácticas comerciales desleales en las que se explotan condiciones laborales precarias para reducir costos de producción y obtener ventajas competitivas.",
    "¿Cuál es la diferencia entre un mercado interno y un mercado externo?": "El mercado interno se refiere a las transacciones comerciales dentro de un país, mientras que el mercado externo implica transacciones comerciales entre diferentes países.",
    "Explique el concepto de inversión extranjera directa (IED).": "La inversión extranjera directa es cuando una empresa invierte en activos o participación accionaria en otro país, con el objetivo de establecer o expandir operaciones comerciales.",
    "¿Cuál es la importancia de la logística en el comercio?": "La logística es crucial en el comercio para la gestión eficiente de la cadena de suministro, que incluye transporte, almacenamiento, y distribución de bienes y servicios.",
    "Explique el concepto de dumping ambiental.": "El dumping ambiental se refiere a la práctica de empresas que trasladan sus procesos contaminantes a países con regulaciones ambientales más laxas, afectando negativamente al medio ambiente y a la salud.",
    "¿Qué es un mercado emergente en el contexto del comercio internacional?": "Un mercado emergente es un país en desarrollo que está experimentando un rápido crecimiento económico y aumento en la participación en el comercio internacional.",
    "Explique el término barreras no arancelarias.": "Las barreras no arancelarias son restricciones al comercio que no implican impuestos directos, como regulaciones sanitarias, normas técnicas y cuotas de importación.",
    "¿Cuál es la función de las aduanas en el comercio internacional?": "Las aduanas son responsables de gestionar el flujo de bienes a través de las fronteras, asegurando el cumplimiento de regulaciones, la recaudación de aranceles y la prevención de contrabando.",
    "Explique el concepto de comercio justo.": "El comercio justo es un enfoque que busca condiciones comerciales equitativas, garantizando salarios justos, condiciones laborales adecuadas y prácticas sostenibles en la producción de bienes.",
    "¿Cuál es la importancia de los acuerdos bilaterales en el comercio internacional?": "Los acuerdos bilaterales son acuerdos comerciales entre dos países que buscan promover el intercambio comercial y la cooperación económica, eliminando o reduciendo barreras comerciales específicas.",
    "¿Qué es una criptomoneda?": "Una criptomoneda es un tipo de moneda digital que utiliza criptografía para garantizar la seguridad de las transacciones y controlar la creación de nuevas unidades.",
    "Explique el concepto de blockchain.": "Blockchain es una cadena de bloques descentralizada y distribuida que registra de manera segura y transparente transacciones en línea. Cada bloque contiene un conjunto de transacciones y está vinculado al bloque anterior.",
    "¿Cuál es la primera criptomoneda y quién la creó?": "Bitcoin es la primera criptomoneda, creada por una persona o grupo bajo el seudónimo de Satoshi Nakamoto en 2009.",
    "¿Qué es la minería de criptomonedas?": "La minería de criptomonedas es el proceso mediante el cual se verifican las transacciones y se añaden nuevos bloques a la cadena de bloques utilizando la potencia de cómputo de los nodos de la red.",
    "Explique el concepto de billetera o monedero criptográfico.": "Una billetera criptográfica es un software o dispositivo que permite a los usuarios almacenar y gestionar sus criptomonedas. Puede ser en línea, en hardware, en papel o en forma de aplicación móvil.",
    "¿Qué es un contrato inteligente (smart contract)?": "Un contrato inteligente es un código de programa autónomo que se ejecuta en una blockchain y se utiliza para automatizar, verificar o hacer cumplir automáticamente los términos de un contrato.",
    "¿Cuál es la diferencia entre una criptomoneda y un token?": "Una criptomoneda es una forma de moneda digital utilizada como medio de intercambio, mientras que un token es un activo digital emitido en una cadena de bloques que puede representar diversas formas de valor.",
    "¿Qué es la bifurcación (fork) en el contexto de las criptomonedas?": "Una bifurcación es un cambio en el protocolo de una criptomoneda que puede resultar en la creación de una nueva cadena de bloques (bifurcación dura) o en la continuación de la cadena existente (bifurcación suave).",
    "Explique el concepto de ICO (Oferta Inicial de Monedas).": "Una ICO es un método de recaudación de fondos en el que se emiten tokens de una nueva criptomoneda a cambio de monedas más establecidas, como Bitcoin o Ethereum.",
    "¿Qué es la volatilidad en el mercado de criptomonedas?": "La volatilidad se refiere a la medida de la variación de los precios de una criptomoneda en un período de tiempo determinado. Los mercados de criptomonedas son conocidos por su alta volatilidad.",
    "¿Cuál es la importancia de la privacidad en las transacciones de criptomonedas?": "La privacidad es esencial en las transacciones de criptomonedas para proteger la identidad y los detalles financieros de los usuarios. Algunas criptomonedas, como Monero y Zcash, se centran en la privacidad.",
    "Explique el concepto de consenso en blockchain.": "El consenso en blockchain se refiere al proceso mediante el cual los nodos en la red llegan a un acuerdo sobre el estado actual de la cadena de bloques y la validez de las transacciones. Los métodos comunes incluyen la Prueba de Trabajo (PoW) y la Prueba de Participación (PoS).",
    "¿Qué es una horquilla dura (hard fork) en una criptomoneda?": "Una horquilla dura es una actualización significativa en el protocolo de una criptomoneda que no es compatible con versiones anteriores. Esto puede dar lugar a la creación de una nueva criptomoneda independiente.",
    "¿Cuál es la función de un nodo en una red blockchain?": "Un nodo en una red blockchain es un dispositivo que participa en la validación y propagación de transacciones. Puede ser un nodo completo, que descarga y almacena toda la cadena de bloques, o un nodo ligero, que solo almacena partes de ella.",
    "¿Qué es la Prueba de Participación (PoS) en criptomonedas?": "La Prueba de Participación es un algoritmo de consenso en el que los participantes (nodos) bloquean una cantidad de criptomonedas como garantía para validar y proponer bloques en la cadena. Cuanta más moneda se posea, mayor será la probabilidad de ser seleccionado para validar bloques.",
    "Explique el concepto de escalabilidad en blockchain.": "La escalabilidad en blockchain se refiere a la capacidad de la red para manejar un número creciente de transacciones y usuarios sin comprometer la velocidad y eficiencia de la cadena de bloques.",
    "¿Cuál es la diferencia entre una cadena de bloques pública y una privada?": "Una cadena de bloques pública es accesible para cualquier persona y permite la participación abierta, mientras que una cadena de bloques privada tiene restricciones de acceso y suele ser utilizada por un grupo o consorcio de entidades.",
    "¿Qué es un ataque del 51% en criptomonedas?": "Un ataque del 51% ocurre cuando un actor malintencionado controla más del 50% del poder de cómputo en una red blockchain, lo que le permite manipular transacciones y comprometer la seguridad de la red.",
    "Explique el concepto de tokenización de activos.": "La tokenización de activos implica representar activos del mundo real, como bienes raíces o acciones, mediante tokens en una cadena de bloques. Esto facilita la negociación y la transferencia de propiedad de manera más eficiente.",
    "¿Qué antibiótico se utiliza tratar infecciones del tracto urinario?": "El antibiotico para eso generalmente es el Ciprofloxacino",
    "Explique la importancia de la azitromicina como antibiótico.": "La azitromicina es un antibiótico utilizado para tratar infecciones respiratorias y de la piel, y es conocido por su régimen de dosificación más corto en comparación con otros antibióticos.",
    "¿Cuál es el nombre de un antibiótico aminoglucósido utilizado para tratar infecciones graves?": "Se utiliza la Gentamicina",
    "Explique la función de la doxiciclina como antibiótico.": "La doxiciclina es un antibiótico de la familia de las tetraciclinas, utilizado para tratar diversas infecciones bacterianas, incluyendo enfermedades transmitidas por garrapatas y acné.",
    "¿Qué antibiótico se utiliza típicamente para tratar infecciones por estafilococos?": "Es la Dicloxacilina",
    "¿Cuál es el nombre de un antibiótico macrólido comúnmente prescrito para infecciones respiratorias?": "Es la Eritromicina",
    "Explique la importancia de la vancomicina como antibiótico.": "La vancomicina es un antibiótico utilizado para tratar infecciones bacterianas graves, especialmente aquellas resistentes a otros antibióticos. Es importante en el tratamiento de infecciones por estafilococos resistentes a la meticilina (MRSA).",
    "¿Qué antibiótico se utiliza para tratar infecciones causadas por bacterias anaerobias?": "Metronidazol",
    "Explique la función de la ceftriaxona como antibiótico de la familia de las cefalosporinas.": "La ceftriaxona se utiliza para tratar diversas infecciones bacterianas y es conocida por su amplio espectro de acción y capacidad para penetrar en tejidos y fluidos corporales.",
    "¿Cuál es el nombre de un antibiótico fluoroquinolona utilizado para tratar infecciones del tracto respiratorio y urinario?": "Es el Levofloxacino",
    "¿Qué antibiótico se utiliza para tratar infecciones por Mycobacterium tuberculosis?": "Isoniazida",
    "Explique la importancia de la clindamicina como antibiótico.": "La clindamicina se utiliza para tratar infecciones graves causadas por bacterias anaerobias y algunos tipos de bacterias Gram-positivas. También es efectiva contra infecciones dentales y de la piel.",
    "¿Cuál es el nombre de un antibiótico sulfamídico utilizado en combinación con trimetoprim para tratar infecciones bacterianas?": "Sulfametoxazol/Trimetoprim",
    "Explique la función de la rifampicina como antibiótico.": "La rifampicina se utiliza para tratar infecciones bacterianas, especialmente aquellas causadas por Mycobacterium tuberculosis. Es un componente clave en el tratamiento de la tuberculosis.",
    "¿Qué antibiótico se utiliza para tratar infecciones del tracto urinario causadas por Escherichia coli?": "Es la Nitrofurantoína",
    "¿Cuál es el nombre de un antibiótico carbapenémico utilizado en infecciones graves y resistentes a otros antibióticos?": "El medicamento es Meropenem",
    "Explique la importancia de la linezolida como antibiótico.": "La linezolida se utiliza para tratar infecciones bacterianas resistentes a otros antibióticos, especialmente aquellas causadas por bacterias Gram-positivas, como el Staphylococcus aureus resistente a la meticilina (MRSA).",
    "¿Qué antibiótico se utiliza para tratar infecciones causadas por Helicobacter pylori?": "Claritromicina",
    "Explique la función de la amikacina como antibiótico aminoglucósido.": "La amikacina se utiliza para tratar infecciones graves causadas por bacterias Gram-negativas y es especialmente útil cuando otras opciones de tratamiento son limitadas debido a la resistencia bacteriana.",
    "¿Qué es el fluconazol y cuál es su mecanismo de acción?": "El fluconazol es un antifúngico que pertenece al grupo de los azoles. Actúa inhibiendo la enzima lanosterol 14-α-demetilasa, impidiendo la conversión del lanosterol en ergosterol, componente esencial de la membrana celular fúngica.",
    "¿Cuál es el propósito del clotrimazol y cómo actúa?": "El clotrimazol es un antifúngico imidazólico utilizado para tratar infecciones por hongos. Funciona interfiriendo con la síntesis del ergosterol, un componente crucial de las membranas celulares fúngicas.",
    "¿Qué es el terbinafina y cuál es su mecanismo de acción?": "La terbinafina es un antifúngico que actúa inhibiendo la enzima esqualeno epoxidasa, bloqueando así la síntesis de ergosterol en los hongos. Esto afecta la integridad de la membrana celular fúngica.",
    "¿Cuál es la función del miconazol y cómo ejerce su acción antifúngica?": "El miconazol es un antifúngico azólico que inhibe la enzima lanosterol 14-α-demetilasa, interfiriendo en la síntesis de ergosterol en los hongos y comprometiendo la integridad de sus membranas celulares.",
    "¿Qué es la anfotericina B y cómo funciona como antifúngico?": "La anfotericina B es un antifúngico poliénico que se une a los esteroles en las membranas celulares fúngicas, alterando su permeabilidad y llevando a la fuga de componentes celulares, lo que resulta en la muerte del hongo.",
    "¿Cuál es el propósito del ketoconazol y cuál es su mecanismo de acción?": "El ketoconazol es un antifúngico imidazólico que inhibe la enzima lanosterol 14-α-demetilasa, interfiriendo en la síntesis de ergosterol en los hongos y comprometiendo la estructura de sus membranas celulares.",
    "¿Qué es el itraconazol y cómo ejerce su acción antifúngica?": "El itraconazol es un antifúngico azólico que afecta la síntesis de ergosterol en los hongos al inhibir la enzima lanosterol 14-α-demetilasa, comprometiendo así la integridad de sus membranas celulares.",
    "¿Cuál es la función del voriconazol y cómo actúa como antifúngico?": "El voriconazol es un antifúngico azólico que inhibe la enzima lanosterol 14-α-demetilasa, interfiriendo en la síntesis de ergosterol en los hongos y comprometiendo la estructura de sus membranas celulares.",
    "¿Qué es el caspofungin y cuál es su mecanismo de acción como antifúngico?": "La caspofungina es un antifúngico del grupo de las equinocandinas que inhibe la síntesis de la pared celular de los hongos al dirigirse a la enzima 1,3-beta-D-glucano sintetasa, esencial para la formación de la pared celular.",
    "¿Cuál es la acción del nistatina y cómo se utiliza como antifúngico?": "La nistatina es un antifúngico poliénico que se une a los esteroles en las membranas celulares fúngicas, alterando su permeabilidad y llevando a la fuga de componentes celulares, lo que resulta en la muerte del hongo.",
    "¿Qué es el posaconazol y cuál es su mecanismo de acción como antifúngico?": "El posaconazol es un antifúngico azólico que inhibe la enzima lanosterol 14-α-demetilasa, interfiriendo en la síntesis de ergosterol en los hongos y comprometiendo la estructura de sus membranas celulares.",
    "¿Cuál es la función del ciclopirox y cómo actúa como antifúngico?": "El ciclopirox es un agente antifúngico que inhibe la función de la enzima hemo peroxidasa, interfiriendo en la síntesis de ácidos nucleicos y proteínas en los hongos.",
    "¿Qué es el griseofulvina y cuál es su mecanismo de acción como antifúngico?": "La griseofulvina es un antifúngico que interfiere con la mitosis fúngica al unirse a la tubulina, afectando la formación del huso mitótico y la replicación del ADN.",
    "¿Cuál es la acción del flucitosina y cómo se utiliza como antifúngico?": "La flucitosina es un antifúngico que se convierte en 5-fluorouracilo dentro de las células fúngicas, interfiriendo en la síntesis del ácido ribonucleico (ARN) y el ácido desoxirribonucleico (ADN).",
    "¿Qué es el amorolfina y cuál es su mecanismo de acción como antifúngico?": "La amorolfina es un antifúngico que altera la síntesis de los esteroles en las membranas celulares fúngicas, provocando la pérdida de su integridad y la muerte del hongo.",
    "¿Cuál es la función del econazol y cómo actúa como antifúngico?": "El econazol es un antifúngico azólico que inhibe la enzima lanosterol 14-α-demetilasa, interfiriendo en la síntesis de ergosterol en los hongos y comprometiendo la estructura de sus membranas celulares.",
    "¿Qué es el tioconazol y cuál es su mecanismo de acción como antifúngico?": "El tioconazol es un antifúngico imidazólico que inhibe la enzima lanosterol 14-α-demetilasa, interfiriendo en la síntesis de ergosterol en los hongos y comprometiendo la integridad de sus membranas celulares.",
    "¿Qué es la amoxicilina, y cuál es su mecanismo de acción?": "La amoxicilina es un antibiótico de la familia de las penicilinas. Su mecanismo de acción consiste en interferir con la síntesis de la pared celular bacteriana, lo que lleva a la lisis (ruptura) de las bacterias. Se utiliza para tratar infecciones bacterianas como sinusitis, otitis, infecciones del tracto respiratorio y urinario.",
    "¿Qué es el ciprofloxacino y cómo actúa?": "El ciprofloxacino es una fluoroquinolona que inhibe la enzima ADN girasa y la topoisomerasa IV en las bacterias, impidiendo la replicación del ADN y llevando a la muerte celular. Se utiliza para tratar infecciones del tracto urinario, respiratorio y gastrointestinal, así como infecciones de la piel y tejidos blandos.",
    "¿Cuál es la función de la azitromicina y su mecanismo de acción?": "La azitromicina es un antibiótico macrólido que inhibe la síntesis de proteínas bacterianas al unirse a la subunidad 50S del ribosoma. Se utiliza para tratar infecciones respiratorias, como neumonía y bronquitis, así como infecciones de la piel y tejidos blandos.",
    "¿Qué es la gentamicina y cómo funciona?": "La gentamicina es un antibiótico aminoglucósido que inhibe la síntesis de proteínas bacterianas al unirse a la subunidad 30S del ribosoma. Se utiliza para tratar infecciones graves causadas por bacterias Gram-negativas. Puede aplicarse en infecciones del tracto urinario, respiratorio y de la piel.",
    "¿Cuál es la función de la doxiciclina y su mecanismo de acción?": "La doxiciclina es una tetraciclina que inhibe la síntesis de proteínas bacterianas al unirse a la subunidad 30S del ribosoma. Se utiliza para tratar infecciones causadas por bacterias Gram-positivas y Gram-negativas, así como infecciones respiratorias, del tracto urinario y de la piel.",
    "¿Qué es la dicloxacilina y cómo actúa?": "La dicloxacilina es una penicilina resistente a la penicilinasa que inhibe la síntesis de la pared celular bacteriana. Se utiliza para tratar infecciones causadas por bacterias Gram-positivas, como Staphylococcus aureus. Puede aplicarse en infecciones de piel y tejidos blandos.",
    "¿Cuál es la acción de la eritromicina y su mecanismo de acción?": "La eritromicina es un antibiótico macrólido que inhibe la síntesis de proteínas bacterianas al unirse a la subunidad 50S del ribosoma. Se utiliza para tratar infecciones respiratorias, de la piel, tejidos blandos y en personas alérgicas a la penicilina.",
    "¿Qué es la vancomicina y cómo funciona?": "La vancomicina es un antibiótico glicopéptido que inhibe la síntesis de la pared celular bacteriana. Se utiliza para tratar infecciones graves causadas por bacterias Gram-positivas, especialmente aquellas resistentes a la meticilina como el Staphylococcus aureus resistente a la meticilina (MRSA).",
    "¿Cuál es la función del metronidazol y su mecanismo de acción?": "El metronidazol es un antibiótico que interfiere con el ADN bacteriano al generar especies reactivas de oxígeno. Se utiliza para tratar infecciones causadas por bacterias anaerobias, como las del tracto gastrointestinal y ginecológico.",
    "¿Qué es la amikacina y cómo actúa?": "La amikacina es un aminoglucósido que inhibe la síntesis de proteínas bacterianas al unirse a la subunidad 30S del ribosoma. Se utiliza para tratar infecciones graves causadas por bacterias Gram-negativas. Puede aplicarse en infecciones del tracto urinario, respiratorio y de la piel.",
    "¿Cuál es la acción del levofloxacino y su mecanismo de acción?": "El levofloxacino es una fluoroquinolona que inhibe la enzima ADN girasa y la topoisomerasa IV en las bacterias, impidiendo la replicación del ADN y llevando a la muerte celular. Se utiliza para tratar infecciones del tracto respiratorio, urinario y de la piel.",
    "¿Qué es la claritromicina y cómo funciona?": "La claritromicina es un antibiótico macrólido que inhibe la síntesis de proteínas bacterianas al unirse a la subunidad 50S del ribosoma. Se utiliza para tratar infecciones respiratorias, de la piel y tejidos blandos, así como úlceras causadas por Helicobacter pylori.",
    "¿Cuál es la función del meropenem y su mecanismo de acción?": "El meropenem es un antibiótico carbapenémico que inhibe la síntesis de la pared celular bacteriana. Se utiliza para tratar infecciones graves y resistentes a otros antibióticos causadas por bacterias Gram-positivas y Gram-negativas.",
    "¿Qué es la linezolida y cómo actúa?": "La linezolida es un antibiótico del grupo de las oxazolidinonas que inhibe la síntesis de proteínas bacterianas al interferir con la formación del complejo ribosoma-ARN de transferencia. Se utiliza para tratar infecciones causadas por bacterias Gram-positivas, incluido el MRSA.",
    "¿Cuál es la acción del sulfametoxazol/trimetoprim y su mecanismo de acción?": "El sulfametoxazol/trimetoprim es una combinación de un sulfamídico y una dihidrofolato reductasa. Actúa inhibiendo la síntesis de ácido fólico en las bacterias, afectando su capacidad de reproducción. Se utiliza para tratar infecciones del tracto urinario, respiratorio y gastrointestinales.",
    "¿Qué es la nitrofurantoína y cómo funciona?": "La nitrofurantoína es un antibiótico que afecta la síntesis de ADN bacteriano al interferir con varios procesos celulares. Se utiliza para tratar infecciones del tracto urinario causadas por bacterias Gram-negativas y algunas Gram-positivas.",
    "¿Cuál es la función del imipenem y su mecanismo de acción?": "El imipenem es un antibiótico carbapenémico que inhibe la síntesis de la pared celular bacteriana. Se utiliza para tratar infecciones graves causadas por bacterias Gram-positivas y Gram-negativas. Suele combinarse con cilastatina para prevenir su degradación renal.",
    "¿Qué es la ceftriaxona y cómo actúa?": "La ceftriaxona es una cefalosporina que inhibe la síntesis de la pared celular bacteriana. Se utiliza para tratar infecciones graves causadas por bacterias Gram-positivas y Gram-negativas. Puede aplicarse en infecciones del tracto respiratorio, urinario y del sistema nervioso central.",
    "¿Cuál es la acción del clindamicina y su mecanismo de acción?": "La clindamicina es un antibiótico que inhibe la síntesis de proteínas bacterianas al unirse a la subunidad 50S del ribosoma. Se utiliza para tratar infecciones graves causadas por bacterias Gram-positivas y anaerobias, así como infecciones dentales.",
    "¿Qué es la tetraciclina y cómo funciona?": "La tetraciclina es un antibiótico que inhibe la síntesis de proteínas bacterianas al unirse a la subunidad 30S del ribosoma. Se utiliza para tratar infecciones causadas por bacterias Gram-positivas y Gram-negativas, así como enfermedades transmitidas por garrapatas."
}

questions_list = (
    "¿Qué es Kron Smart Chain?",
    "¿Cuál es el protocolo de consenso utilizado por Kron Smart Chain?",
    "¿Cómo se crean los criptoactivos en la Red Kron?",
    "¿Qué hace destacar a Kron Smart Chain en cuanto a la tokenización de activos del mundo real?",
    "¿Cómo se gestionan las recompensas en Kron Smart Chain?",
    "¿Qué ventajas tiene el modelo UTXO en Kron Smart Chain?",
    "¿Cómo contribuye la base de datos relacional en Kron Smart Chain?",
    "¿Cómo se refuerza la privacidad y seguridad en la red de Kron Smart Chain?",
    "¿Cuál es el enfoque principal de Kron Smart Chain?",
    "¿Cómo contribuye la participación activa de los nodos en la red?",
    "¿Qué son los criptoactivos en la red Kron?",
    "¿Cómo funciona el estándar de token kr10 en Kron Smart Chain?",
    "¿Cuál es la característica destacada de Kron Smart Chain en la tokenización de activos del mundo real?",
    "¿Cómo se pueden pagar recompensas en Kron Smart Chain?",
    "¿Cómo garantiza Kron Smart Chain la privacidad en las transacciones?",
    "¿Cuál es el modelo central en la arquitectura de Kron Smart Chain?",
    "¿Por qué es relevante el uso de la red Tor en Kron Smart Chain?",
    "¿En qué sectores puede Kron Smart Chain encontrar aplicaciones?",
)

preguntas = list(data.keys())
respuestas = list(data.values())
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(preguntas, respuestas)
def handle_message(update: Update, context: CallbackContext):
    pregunta_usuario = update.message.text
    respuesta_modelo = model.predict([pregunta_usuario])[0]
    update.message.reply_text(f"{respuesta_modelo}")
updater = Updater(token='', use_context=True)

dp = updater.dispatcher

dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

updater.start_polling()

updater.idle()
