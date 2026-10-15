#!/bin/bash
# Download the 11 Sandstones binaries (Neumann et al., DRP-317, DOI 10.17612/F4H1-W124)
# from digitalporousmedia.org -- anonymous, no login. Each: 1000^3 uint8, 2.25 um, 0 = pore.
# The download API returns a one-shot tapis postit URL as JSON; the trailing slash matters.
# Resumable: finished files (exactly 1e9 bytes) are skipped.
#
#   nohup bash scripts/fetch_sandstones11.sh > FETCH_SST11.log 2>&1 &
set -u
OUT=${OUT:-DATA/rock/sandstones11}
API=https://digitalporousmedia.org/api/datafiles/tapis/download/projects/drp.project.published.DRP-317
mkdir -p "$OUT"
while IFS='|' read -r dir stem; do
  f="$OUT/${stem}_2d25um_binary.raw"
  if [ -f "$f" ] && [ "$(stat -c %s "$f")" -eq 1000000000 ]; then echo "have $f"; continue; fi
  d=$(printf '%s' "$dir" | sed 's/ /%20/g')
  url=$(curl -sS -A "Mozilla/5.0" --max-time 60 "$API/$d/${stem}_2d25um_binary.raw/${stem}_2d25um_binary.raw/" \
        | python3 -I -c "import json,sys; print(json.load(sys.stdin)['data'])")
  echo "get $dir -> $f  $(date -Is)"
  curl -sS -A "Mozilla/5.0" --retry 3 -o "$f.part" "$url" && mv "$f.part" "$f"
  ls -l "$f"
done <<'LIST'
Bandera Brown|BanderaBrown
Bandera Gray|BanderaGray
Bentheimer|Bentheimer
Berea|Berea
Berea Sister Gray|BSG
Berea Upper Gray|BUG
Buff Berea|BB
CastleGate|CastleGate
Kirby|Kirby
Leopard|Leopard
Parker|Parker
LIST
echo "FETCH DONE $(date -Is)"
