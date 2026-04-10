"""Vercel FastAPI entrypoint.

If importing the main app fails, expose a minimal error route so deployment logs
and the browser show the actual exception instead of a generic import failure.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

try:
    from web_backend import app as app
except Exception as exc:  # pragma: no cover - only used when startup fails
    app = FastAPI()

    @app.api_route('/{path:path}', methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'])
    async def startup_error(path: str):
        return JSONResponse({
            'status': 'error',
            'message': 'Application failed to start.',
            'details': str(exc)
        }, status_code=500)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=5000)
