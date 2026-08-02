import os

from slopfinity.server import app  # noqa: F401

if __name__ == '__main__':
    import uvicorn
    # Loopback by default; operators who need LAN/docker/proxy access
    # set SLOPFINITY_BIND_HOST=0.0.0.0 and SLOPFINITY_TRUSTED_ORIGINS
    # together. See slopfinity/server.py for the CSRF middleware that
    # consumes those env vars.
    host = os.environ.get('SLOPFINITY_BIND_HOST', '127.0.0.1')
    port = int(os.environ.get('SLOPFINITY_BIND_PORT', '9099'))
    uvicorn.run(app, host=host, port=port)
