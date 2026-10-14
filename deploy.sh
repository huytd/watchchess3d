#!/bin/sh
# Deploys the screen-sharing page (/) to Vercel as plain static files: no server, no build.
# vercel.json and .vercelignore say what goes up; the /server page and app.py stay out.
#
#   ./deploy.sh            production deployment
#   ./deploy.sh --preview  a preview deployment, at its own URL
#
# Needs Node.js. The first run asks you to log in and to pick or create the Vercel project
# (it remembers the choice in .vercel/). Elsewhere, such as in CI, set VERCEL_TOKEN (from
# vercel.com/account/tokens) along with VERCEL_ORG_ID and VERCEL_PROJECT_ID.
set -e
cd "$(dirname "$0")"
target=--prod
if [ "$1" = "--preview" ]; then target=; shift; fi
exec npx --yes vercel@64 deploy $target ${VERCEL_TOKEN:+--token "$VERCEL_TOKEN"} "$@"
