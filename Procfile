web: uvicorn slopfinity.server:app --host ${SLOPFINITY_BIND_HOST:-127.0.0.1} --port ${SLOPFINITY_BIND_PORT:-9099} --reload --reload-dir slopfinity
css: bin/tailwindcss -i src/tailwind.css -o slopfinity/static/tailwind.css --content "slopfinity/templates/*.html,slopfinity/static/app.js" --watch
