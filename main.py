import uvicorn
from fastapi import FastAPI,Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from stream_utils import Streaming
import threading


app = FastAPI()



app.mount("/static", StaticFiles(directory="static"), name="static")

stream_thread = None


streaming = Streaming()

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

@app.get("/devices")
def devices():
    return streaming.list_available_devices()

@app.get("/start")
def start_stream(
    source: str = Query("0"),
    fps: int = Query(15),
    blur_strength: int = Query(21),
    background: str = Query("none")
):
    streaming.update_streaming_config(in_source=source, out_source=None, fps=fps, blur_strength=blur_strength, background=background)

    global stream_thread

    if streaming.running:
        return JSONResponse(content={"message": "Stream already running"}, status_code=400)

    if fps < 1 or fps > 60:
        return JSONResponse(content={"message": "Invalid FPS value (1-60)"}, status_code=400)

    stream_thread = threading.Thread(
        target=streaming.stream_video, args=()
    )
    stream_thread.start()
    return {"message": f"Streaming started from source: {fps} FPS and blur strength {blur_strength}"}


@app.get("/stop")
def stop_stream():
    return streaming.update_running_status()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)