from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io

#instancia
app = FastAPI()

#endpoins
@app.get("/")
def read_root():
    return {"Mensaje": "Hello World Api funcionando"}

@app.post("/limpiar_imagen/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()

    # esto onvertirte contenido binario de bytes a una imagen en un arreglo
    np_array = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # aqui guarda la imagen procesada en un archivo temporal en memoria
    _, buffer = cv2.imencode('.png', gray)
    bytes_io = buffer.tobytes()

    #Devuelve una imagen PNG con SreamingResponse en memoria
    return StreamingResponse(io.BytesIO(bytes_io), media_type="image/png")

    """ return {
        "nombre_archivo": file.filename,
        "type": file.content_type, 
        "size": len(content)
        } """