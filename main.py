from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import io
import pytesseract
import spacy
import fitz


#instancia
app = FastAPI()

#tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# spacy modelo españo
nlp_en = spacy.load("en_core_web_sm")
nlp_es = spacy.load("es_core_news_sm")


#endpoins
@app.get("/")
def read_root():
    return {"Mensaje": "Hello World Api funcionando"}

#enpoint openCV
@app.post("/limpiar_imagen/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()

    # Convertir contenido binario a imagen
    np_array = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ajustar contraste y brillo
    alpha = 1.2  # Contraste (>1 aumenta contraste)
    beta = -15   # Brillo (0 sin cambio)
    gray_contraste = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Nitidez
    arraynitidez = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    imagen_nitida = cv2.filter2D(gray_contraste, -1, arraynitidez)

    # Convertir a PNG en memoria
    _, buffer = cv2.imencode('.png', imagen_nitida)
    bytes_io = buffer.tobytes()

    # Devolver imagen procesada
    return StreamingResponse(io.BytesIO(bytes_io), media_type="image/png")

#endpoint ocr
@app.post("/leer_texto/")
async def leer_texto(file: UploadFile = File(...)):
    content = await file.read()

    # Convertir contenido binario a imagen
    np_array = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar binarización adaptativa para mejorar lectura
    binarizada = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )

    # Extraer texto con Tesseract (puedes cambiar lang="spa" si está en español)
    texto_limpio = pytesseract.image_to_string(binarizada, lang="eng")

    # Devolver texto extraído
    return JSONResponse(content={
        "texto_extraido": texto_limpio or "No se detectó texto",
        "longitud_texto": len(texto_limpio)
    })

#enpoint procesar imagen - extraer texto - y npl
@app.post("/procesar_imagen/")
async def procesar_imagen(file: UploadFile = File(...)):
    content = await file.read()

    np_array = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Ajustar contraste y brillo
    alpha = 1.2  # Contraste (>1 aumenta contraste)
    beta = -15   # Brillo (0 sin cambio)
    gray_contraste = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Mejorar nitidez
    arraynitidez = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    imagen_nitida = cv2.filter2D(gray_contraste, -1, arraynitidez)

    # Binarización adaptativa (mejora el OCR)
    binarizada = cv2.adaptiveThreshold(
        imagen_nitida, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 2
    )

    # Extraer texto con Tesseract OCR
    texto_extraido = pytesseract.image_to_string(binarizada, lang="eng").strip()


    doc = nlp_es(texto_extraido)
    entidades = [{"texto": ent.text, "tipo": ent.label_} for ent in doc.ents]

    #respuesta JSON
    return JSONResponse(content={
        "nombre_archivo": file.filename,
        "texto_extraido": texto_extraido or "No se detectó texto",
        "longitud_texto": len(texto_extraido),
        "entidades": entidades
    })

@app.post("/procesar_pdf/")
async def procesar_pdf_digital(file: UploadFile = File(...)):
    # Leer el PDF cargado
    content = await file.read()
    
    # Guardar temporalmente el PDF
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(content)

    # Abrir el PDF con PyMuPDF
    doc_pdf = fitz.open(temp_path)
    texto_extraido = ""

    # Extraer texto de todas las páginas
    for pagina in doc_pdf:
        texto_extraido += pagina.get_text("text") + "\n"

    doc_pdf.close()

    # Procesar con spaCy
    doc = nlp_es(texto_extraido)
    entidades = [{"texto": ent.text, "tipo": ent.label_} for ent in doc.ents]

    # Eliminar archivo temporal
    import os
    os.remove(temp_path)

    # Retornar resultado
    return JSONResponse(content={
        "nombre_archivo": file.filename,
        "texto_extraido": texto_extraido.strip() or "No se detectó texto",
        "longitud_texto": len(texto_extraido),
        "entidades": entidades
    })


@app.post("/procesar_archivo/")
async def procesar_archivo(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename.lower()

    # PDF
    if filename.endswith(".pdf"):
        temp_path = f"temp_{filename}"
        with open(temp_path, "wb") as f:
            f.write(content)

        doc_pdf = fitz.open(temp_path)
        texto_extraido = ""
        for pagina in doc_pdf:
            texto_extraido += pagina.get_text("text") + "\n"
        doc_pdf.close()

        # Procesar con spaCy
        doc = nlp_es(texto_extraido)
        entidades = [{"texto": ent.text, "tipo": ent.label_} for ent in doc.ents]

        import os
        os.remove(temp_path)

        return JSONResponse(content={
            "tipo_archivo": "PDF digital",
            "nombre_archivo": file.filename,
            "texto_extraido": texto_extraido.strip() or "No se detectó texto",
            "longitud_texto": len(texto_extraido),
            "entidades": entidades
        })

    # imagen
    if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        np_array = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Ajustar contraste y brillo
        alpha = 1.2
        beta = -15
        gray_contraste = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        # Nitidez
        arraynitidez = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        imagen_nitida = cv2.filter2D(gray_contraste, -1, arraynitidez)

        # Binarización adaptativa
        binarizada = cv2.adaptiveThreshold(
            imagen_nitida, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 2
        )

        # OCR con Tesseract
        texto_extraido = pytesseract.image_to_string(binarizada, lang="eng").strip()

        # Procesar con spaCy
        doc = nlp_es(texto_extraido)
        entidades = [{"texto": ent.text, "tipo": ent.label_} for ent in doc.ents]

        return JSONResponse(content={
            "tipo_archivo": "Imagen",
            "nombre_archivo": file.filename,
            "texto_extraido": texto_extraido or "No se detectó texto",
            "longitud_texto": len(texto_extraido),
            "entidades": entidades
        })