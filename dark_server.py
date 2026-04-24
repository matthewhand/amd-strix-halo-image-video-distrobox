from slopfinity.server import app  # noqa: F401

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=9099)
