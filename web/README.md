# Echo web app

The Echo web front end: Next.js (App Router, TypeScript), built as a static export that the FastAPI backend
serves from the Hugging Face Space. It replaces `backend/static/index.html`, which stays at `/legacy` until it
is retired.

## Pages

| URL | Page |
| --- | --- |
| `/` | Search, record or hum; a recording's result shows here |
| `/song/?id=<track id>&key=<artist:title>&d=<0.50–0.95>` | A song and its matches (`d` only when not the default 0.70) |
| `/search/?q=<query>` | Up to 20 songs for a query |
| `/artist/?name=<artist>` | That artist's songs among the top 20 search results (there is no artist endpoint) |
| `/visualizations/` | Model visualizations (images from `/static/`) |

Song ids are in the query string because a static export cannot pre-build a page per song.

## Develop

```bash
npm install
npm run dev
```

`next dev` proxies `/api` and `/static` to the live Space; set `ECHO_BACKEND` to use another backend.

## Build

```bash
npm run lint
npm run build
```

The build writes `out/`. The manual Deploy workflow builds it and copies it to `backend/web`, which
`backend/main.py` mounts at `/` (after `/api`, `/static` and `/legacy`).

## Spotify

The redirect URI is the site root (`https://shreyash616-echo-backend.hf.space/`, unchanged from the old page);
after login the app returns to the page the playlist was made from. Uses `POST /me/playlists` and
`POST /playlists/{id}/items` and checks every response. The localStorage keys are the old page's.
