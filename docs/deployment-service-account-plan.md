# Deployment Plan: Non-Sudo Service Account & Tailscale HTTPS Reverse Proxy

> **Status:** Draft plan (no runtime or host changes applied yet).
> **Scope split:** Generic reverse-proxy support lives in public `inductivebias/voxel`; all hostnames, Tailscale Serve URLs, ACLs, credentials, rig configs, and `systemd` unit files live exclusively in **`voxel-private`**.

---

## 1. Goals & Security Boundaries

1. **No `sudo` on the runtime account:** Run `vxl serve` under a dedicated unprivileged Linux service user (e.g. `lm1-service`) with zero `sudo` / `wheel` membership.
2. **Loopback-only application bind:** Bind `vxl serve` strictly to `127.0.0.1:8000` so the microscope API is never directly exposed on LAN interfaces.
3. **HTTPS termination via Tailscale Serve (managed via `voxel-private`):** Expose the loopback server over Tailscale **HTTPS** (`https://<host>.<tailnet>.ts.net`), with all URL serving, Tailscale Serve configuration, and Tailnet access policy managed exclusively through **`voxel-private`** — never committed to public `voxel` and never delegated to the runtime service user.
4. **Zero-`sudo` code deploys:** Allow operators or CI to update the checkout, rebuild the web UI, and restart the Voxel service without interactive `sudo`.

---

## 2. Why HTTPS (Not Plain HTTP) Is Required

Tailscale Serve should terminate **HTTPS** (`tailscale serve --bg --https=443 http://127.0.0.1:8000`), not plain HTTP:

- **WebGPU preview renderer:** Browsers disable `navigator.gpu` on non-localhost `http://` origins.
- **Xbox Gamepad API (PR #7):** Browsers disable `navigator.getGamepads()` on non-localhost `http://` origins.
- **Session cookie protection:** `ControlAccess` marks `voxel_session` as `Secure` when `scheme == "https"`.
- **Automatic TLS:** Tailscale provisions and renews valid Let's Encrypt certificates for `*.ts.net` automatically, with zero extra reverse-proxy software required.

---

## 3. Required Generic Changes in Public `inductivebias/voxel` (~25 LOC PR)

Inspecting `src/vxl/web/app.py`, `src/vxl/web/access.py`, and `src/vxl/web/router.py` shows **three concrete blockers** when placing `vxl serve --host 127.0.0.1` behind *any* local TLS-terminating reverse proxy (Tailscale Serve, Caddy, nginx):

| # | Current Code | What Breaks Behind a Local Reverse Proxy | Generic Fix in `inductivebias/voxel` |
| :---: | --- | --- | --- |
| 1 | `app.py` `_allowed_hosts("127.0.0.1")` returns only `("localhost", "127.0.0.1", "[::1]")` | `TrustedHostMiddleware` rejects proxied `Host: <public-or-tailnet-fqdn>` with **HTTP 400 `Invalid host header`**. | Support comma-separated `VOXEL_ALLOWED_HOSTS` env var (merged into `_allowed_hosts()`). |
| 2 | `app.py` `uvicorn.run(...)` does not set `proxy_headers=True` / `forwarded_allow_ips` | Uvicorn ignores `X-Forwarded-Proto: https`, leaving `scope["scheme"] == "http"`. As a result:<br>• `ControlAccess` expects `Origin: http://...` while browser sends `Origin: https://...` → **HTTP 403**.<br>• `get_discovery()` returns `ws://...` instead of `wss://...` → **mixed-content WebSocket blocked**.<br>• `voxel_session` cookie omits `Secure`. | Add opt-in `VOXEL_PROXY_HEADERS=1` (or `forwarded_allow_ips="127.0.0.1"` when `VOXEL_ALLOWED_HOSTS` is set) so Uvicorn trusts `X-Forwarded-*` headers **only from loopback (`127.0.0.1,::1`)**. |
| 3 | `app.py` only enforces `VOXEL_WEB_PASSWORD` when `host` is non-loopback | Loopback + reverse proxy exposes remote microscope control with **no application password** unless explicitly configured. | Require `VOXEL_WEB_PASSWORD` whenever `VOXEL_ALLOWED_HOSTS` / proxy mode is active (unless an explicit opt-out like `VOXEL_ALLOW_UNAUTHENTICATED_PROXY=1` is set for ACL-only tailnets). |

### Verification Tests for the Public PR (`tests/web/test_access.py`)
- Request with `Host: rig.example.ts.net`, `X-Forwarded-Proto: https`, `Origin: https://rig.example.ts.net` returns `200`, sets `Secure` cookie, and `GET /api/stations/{id}/discovery` emits `wss://rig.example.ts.net/...`.
- Spoofed `Host` not in `VOXEL_ALLOWED_HOSTS` returns `400`.
- Mismatched `Origin` (`http://...` or third-party origin) returns `403`.
- Proxy mode without `VOXEL_WEB_PASSWORD` fails fast on startup.

---

## 4. Service Account & Hardware Provisioning (Owned by `voxel-private`)

All concrete host provisioning scripts, unit files, and URLs live in **`voxel-private`**. Only the one-time provisioning script requires `sudo`.

### 4.1 Create the Unprivileged Service User & Groups

```bash
# Run once as an administrator during host provisioning:
sudo groupadd --system voxel || true
sudo useradd --system --create-home --home-dir /var/lib/lm1-service \
  --shell /usr/sbin/nologin --gid voxel lm1-service

# Grant minimum required hardware-device groups (verified against typical Linux rig nodes):
#   - users:   PVCAM PCIe camera nodes (/dev/pvcamPCIE_*)
#   - dialout: USB/ACM serial devices (lasers, filter wheels: /dev/ttyUSB*, /dev/ttyACM*)
#   - plugdev: USB peripherals / HID
#   - video, render: NVIDIA/DRI GPU nodes (/dev/nvidia*, /dev/dri/*)
sudo usermod -aG users,dialout,plugdev,video,render lm1-service
```

> **Pre-provisioning hardware permission check:** Confirm ownership/ACLs on every hardware node on the target rig before switching over:
> - Cameras: `ls -l /dev/pvcam*` (confirm group/ACL allows `lm1-service` read/write; add a `udev` rule if needed)
> - Serial lasers / filter wheels: `ls -l /dev/ttyUSB* /dev/ttyACM*` (verify `dialout` group)
> - NI-DAQmx: `ls -l /dev/nipalk` (verify `0666` or NI group membership)
> - CUDA GPU: `ls -l /dev/nvidia*` and test `uv run python -c "from numba import cuda; print(cuda.is_available())"` as `lm1-service`
> - Zaber stages: TCP (`55551`) — requires no OS device group, only LAN route to controller subnet.

### 4.2 Directory Layout & Shared Data Permissions

| Path | Owner / Mode | Purpose |
| --- | --- | --- |
| `/opt/voxel` | `lm1-service:voxel` (`2775`) | Dedicated Voxel git checkout, `.venv`, and built `web/station` bundle |
| `/var/lib/lm1-service/.voxel` | `lm1-service:voxel` (`0750`) | Station config, instrument stores, SQLite records DB |
| `/etc/voxel/lm1.env` | `root:voxel` (`0640`) | `VOXEL_WEB_PASSWORD`, `VOXEL_ALLOWED_HOSTS`, `VOXEL_PROXY_HEADERS=1`, `VOXEL_SYSTEM_CONFIG` |
| `<DATA_ROOT>/<rig-data>` | `lm1-service:voxel` (`2775` + default ACL `g:voxel:rwx`) | Acquired `.ome.zarr`, snapshots, SNR reports — writable by `lm1-service`, readable by operators in group `voxel` without `sudo` |
| `/dev/shm` | `tmpfs` (confirm ≥ 250 GB available) | `OMEZarrWriter` shared-memory ring buffer |

### 4.3 `systemd` Unit Template (stored in `voxel-private/systemd/vxl-lm1.service`)

```ini
[Unit]
Description=Voxel Microscope Control Server
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
Type=simple
User=lm1-service
Group=voxel
UMask=0002
WorkingDirectory=/opt/voxel
EnvironmentFile=/etc/voxel/lm1.env
ExecStart=/opt/voxel/.venv/bin/vxl serve --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5
TimeoutStopSec=30

# Required for OME-Zarr shard file handles & pinned CUDA host memory (cudaHostAlloc):
LimitNOFILE=65536
LimitMEMLOCK=infinity

# Defense-in-depth hardening (compatible with GPU/PVCAM/NI-DAQmx/serial):
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/voxel /var/lib/lm1-service /data /tmp /dev/shm

[Install]
WantedBy=multi-user.target
```

> **Note on `ProtectSystem=strict` + NI-DAQmx / PVCAM:** Test startup carefully; if vendor drivers write lock/cache files under `/var/tmp` or `/opt/pvcam`, add those explicit paths to `ReadWritePaths=`.

---

## 5. Tailscale Serve & `voxel-private` URL Governance

Keep **all** URL exposure and Tailscale state isolated from both public `voxel` and `lm1-service`:

1. **Private repo (`voxel-private`):**
   - Stores the rig's exact MagicDNS URL (`https://<host>.<tailnet>.ts.net`), `systemd` unit, `/etc/voxel/lm1.env` template (secrets out of git or encrypted via SOPS/1Password), and `tailscale serve` setup script.
   - Public `inductivebias/voxel` contains zero references to the rig hostname, tailnet ID, or private URLs.
2. **Tailscale account / ACL boundary (`voxel-private`):**
   - Tag the microscope host in Tailscale (e.g. `tag:voxel-rig`) rather than authenticating `tailscaled` under a personal user login — this avoids personal key expiry knocking the rig offline and decouples ownership from individual sudo accounts.
   - Restrict Tailscale ACL `grants` so only authorized operator group members (e.g. `group:voxel-operators`) can reach `tag:voxel-rig:443`.
   - Configure Tailscale Serve **once** during provisioning (stored persistently by `tailscaled` across reboots):
     ```bash
     sudo tailscale serve --bg --https=443 http://127.0.0.1:8000
     tailscale serve status
     ```
   - Do **not** grant `lm1-service` `tailscale --operator` permissions — the runtime account should never be able to modify Tailscale Serve/Funnel routes.
   - Confirm **Tailscale Funnel is disabled** on the node/tailnet policy so microscope control can never be accidentally exposed to the public internet.

---

## 6. New-Machine Bootstrap & Zero-`sudo` Deployment Workflow

The same plan doubles as a repeatable **new-machine / fresh-account runbook**: public `voxel` holds the generic parameterized provisioning and deploy logic (zero hostnames, URLs, or secrets), while **`voxel-private`** supplies only the per-rig `.env`, `config.yaml`, and Tailscale Serve registration.

### 6.1 New-Machine Prerequisites (Admin, One-Time)

1. **Base OS & tooling (Ubuntu/Debian Linux):**
   - `git`, `acl` (`setfacl`), `curl`, `openssl`
   - `uv` installed system-wide (e.g. `/usr/local/bin/uv`) so any service user can run `uv sync`
   - Node.js 24 LTS + `corepack` / `pnpm@12.16.1` (for building `web/station`)
   - Tailscale daemon (`tailscaled`) enrolled with a rig tag (`tag:voxel-rig`), **not** a personal user login
2. **Hardware drivers & SDKs (as applicable to the rig):**
   - NVIDIA driver + CUDA runtime (for GPU pyramid/`nvcomp` writer)
   - Teledyne Photometrics PVCAM kernel driver + SDK (`/opt/pvcam`, verify `/dev/pvcam*` group/ACL)
   - NI-DAQmx Linux driver (`nidaqmx` / `/dev/nipalk`)
   - USB-serial udev rules (fixed `/dev/serial/by-id/...` symlinks for lasers & filter wheels)
   - Verify `/dev/shm` tmpfs size is large enough for the writer ring buffer (`df -h /dev/shm`)

### 6.2 Parameterized Provisioning Blueprint (One-Time `sudo`, Generic)

A generic script (e.g. `scripts/deploy/provision.sh` or copied into `voxel-private`) creates the account, directories, `systemd` unit, and narrow restart rule from environment variables — no machine names hardcoded:

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${VOXEL_SERVICE_USER:=voxel-service}"
: "${VOXEL_GROUP:=voxel}"
: "${VOXEL_INSTALL_DIR:=/opt/voxel}"
: "${VOXEL_STATE_DIR:=/var/lib/${VOXEL_SERVICE_USER}}"
: "${VOXEL_DATA_DIR:=/data/voxel}"
: "${VOXEL_ENV_FILE:=/etc/voxel/${VOXEL_SERVICE_USER}.env}"
: "${VOXEL_SERVICE_NAME:=vxl-${VOXEL_SERVICE_USER}}"
: "${VOXEL_REPO_URL:=https://github.com/inductivebias/voxel.git}"
: "${VOXEL_BRANCH:=mobi}"

# 1. Service group & unprivileged user (no sudo, no login shell)
getent group "$VOXEL_GROUP" >/dev/null || groupadd --system "$VOXEL_GROUP"
id -u "$VOXEL_SERVICE_USER" >/dev/null 2>&1 || \
  useradd --system --create-home --home-dir "$VOXEL_STATE_DIR" \
    --shell /usr/sbin/nologin --gid "$VOXEL_GROUP" "$VOXEL_SERVICE_USER"

for grp in users dialout plugdev video render; do
  getent group "$grp" >/dev/null && usermod -aG "$grp" "$VOXEL_SERVICE_USER"
done

# 2. Directories with setgid + default group ACL so operators in $VOXEL_GROUP can read data
install -d -o "$VOXEL_SERVICE_USER" -g "$VOXEL_GROUP" -m 2775 "$VOXEL_INSTALL_DIR" "$VOXEL_DATA_DIR"
install -d -o "$VOXEL_SERVICE_USER" -g "$VOXEL_GROUP" -m 0750 "$VOXEL_STATE_DIR/.voxel"
setfacl -R -m "g:${VOXEL_GROUP}:rwx" -m "d:g:${VOXEL_GROUP}:rwx" "$VOXEL_DATA_DIR"

# 3. Initial env file (generate strong password once; populate VOXEL_ALLOWED_HOSTS from voxel-private)
install -d -o root -g "$VOXEL_GROUP" -m 0750 "$(dirname "$VOXEL_ENV_FILE")"
if [[ ! -f "$VOXEL_ENV_FILE" ]]; then
  umask 027
  cat >"$VOXEL_ENV_FILE" <<EOF
VOXEL_WEB_PASSWORD=$(openssl rand -base64 24)
VOXEL_PROXY_HEADERS=1
VOXEL_ALLOWED_HOSTS=
EOF
  chown "root:${VOXEL_GROUP}" "$VOXEL_ENV_FILE"
  chmod 0640 "$VOXEL_ENV_FILE"
fi

# 4. Initial repo clone owned by service user
if [[ ! -d "$VOXEL_INSTALL_DIR/.git" ]]; then
  sudo -u "$VOXEL_SERVICE_USER" git clone --branch "$VOXEL_BRANCH" "$VOXEL_REPO_URL" "$VOXEL_INSTALL_DIR"
fi

# 5. Narrow sudoers rule allowing ONLY restart/status of this single unit
cat >"/etc/sudoers.d/${VOXEL_SERVICE_NAME}" <<EOF
%${VOXEL_GROUP} ALL=(root) NOPASSWD: /usr/bin/systemctl restart ${VOXEL_SERVICE_NAME}.service, /usr/bin/systemctl status ${VOXEL_SERVICE_NAME}.service
EOF
chmod 0440 "/etc/sudoers.d/${VOXEL_SERVICE_NAME}"
visudo -cf "/etc/sudoers.d/${VOXEL_SERVICE_NAME}"
```

### 6.3 Zero-`sudo` Deploy & Rollback Blueprint (Repeatable Updates)

Once provisioned, deploying any commit/tag — or rolling back to a previous commit — runs as `VOXEL_SERVICE_USER` with no general `sudo`:

```bash
#!/usr/bin/env bash
set -euo pipefail

REF="${1:?usage: deploy.sh <git-ref-or-sha>}"
: "${VOXEL_INSTALL_DIR:=/opt/voxel}"
: "${VOXEL_SERVICE_NAME:=vxl-lm1-service}"
: "${VOXEL_UV_EXTRAS:=--extra web --extra zaber --extra pvcam-camera}"

cd "$VOXEL_INSTALL_DIR"
PREV_SHA="$(git rev-parse HEAD)"
echo "Deploying ${REF} (previous: ${PREV_SHA})"

git fetch origin --tags
git checkout --detach "$REF"

# 1. Sync Python workspace (exact locked dependencies)
uv sync --locked --all-packages ${VOXEL_UV_EXTRAS}

# 2. Build web UI bundle served by `vxl serve`
pnpm -C web install --frozen-lockfile
pnpm -C web/station build

# 3. Smoke-check CLI & GPU availability before restarting live service
uv run --locked vxl --help >/dev/null
uv run --locked python -c "import vxl, ome_zarr_writer; print('vxl import ok')"

# 4. Restart via single scoped sudoers rule (or `systemctl --user restart` in user-unit mode)
sudo /usr/bin/systemctl restart "${VOXEL_SERVICE_NAME}.service"
sudo /usr/bin/systemctl is-active --quiet "${VOXEL_SERVICE_NAME}.service"
echo "Deployed $(git rev-parse --short HEAD) successfully. Rollback: $0 ${PREV_SHA}"
```

> **Alternative (`systemd --user` mode):** If you prefer zero `sudoers` entries at all, enable lingering once (`sudo loginctl enable-linger "$VOXEL_SERVICE_USER"`) and place the unit in `~${VOXEL_SERVICE_USER}/.config/systemd/user/vxl.service`. System units with the single scoped `systemctl restart` rule are preferred on hardware rigs because they support `ProtectSystem=strict` and `LimitMEMLOCK=infinity` cleanly.

### 6.4 First-Time Rig Config Wiring (from `voxel-private`)

After provisioning a new machine:
1. Copy or symlink the rig's private `system.yaml`, `station.yaml`, and `<rig>.voxel/config.yaml` from **`voxel-private`** into `/var/lib/<service-user>/.voxel/`.
2. Set `VOXEL_ALLOWED_HOSTS=<host>.<tailnet>.ts.net` in `/etc/voxel/<service-user>.env`.
3. Run `deploy.sh origin/mobi` to build the venv and web UI and start the service.
4. From the **`voxel-private`** admin workflow, enable `sudo tailscale serve --bg --https=443 http://127.0.0.1:8000` and verify ACL grants.

---

## 7. Phased Rollout & Verification Checklist

### Phase 1 — Public `voxel` PR (generic reverse-proxy support)
- [ ] Add `VOXEL_ALLOWED_HOSTS` and loopback `proxy_headers` support in `src/vxl/web/app.py`.
- [ ] Add unit tests in `tests/web/` covering forwarded HTTPS headers, `wss://` discovery URLs, host rejection, origin rejection, and password enforcement.
- [ ] Pass `CI` (`ruff`, `basedpyright`, `pytest`, `web`).

### Phase 2 — `voxel-private` repository setup
- [ ] Create private repo (or directory in private rig infra) containing `systemd/vxl-lm1.service`, `scripts/provision-lm1.sh`, `scripts/deploy-lm1.sh`, `tailscale/serve.sh`, and ACL policy snippet.

### Phase 3 — Host provisioning & dry-run verification on target rig
- [ ] Provision `lm1-service` user, groups, `/opt/voxel`, `/var/lib/lm1-service`, `/etc/voxel/lm1.env`, and `<DATA_ROOT>` ACLs.
- [ ] Verify hardware access as `lm1-service` (zero `sudo`):
  - [ ] CUDA available (`numba.cuda.is_available() == True`)
  - [ ] PVCAM camera enumeration succeeds (`vxl-pvcam-diagnose`)
  - [ ] Zaber probe succeeds (`vxl-zaber-diagnose`)
  - [ ] NI-DAQmx task creation succeeds
  - [ ] Serial lasers / filter wheels open cleanly
  - [ ] Write & read test file in `<DATA_ROOT>` with expected `voxel` group permissions
- [ ] Start `vxl-lm1.service` bound to `127.0.0.1:8000`; verify `ss -tlnp` shows loopback-only listener.
- [ ] Enable `tailscale serve --bg --https=443 http://127.0.0.1:8000` under `voxel-private` governance and verify Funnel is off.
- [ ] End-to-end browser check at `https://<host>.<tailnet>.ts.net`:
  - [ ] Basic auth prompt succeeds (`voxel` + `VOXEL_WEB_PASSWORD`) and sets `Secure; HttpOnly; SameSite=Strict` cookie.
  - [ ] WebSocket state, preview, and log streams connect over `wss://` with zero console/mixed-content errors.
  - [ ] WebGPU live preview renders cleanly.
  - [ ] Xbox controller Gamepad API detects USB controller over HTTPS (once PR `#7` is merged).
  - [ ] Single-frame snapshot & short OME-Zarr stack write cleanly to `<DATA_ROOT>` and are readable by operators without `sudo`.
- [ ] Decommission old ad-hoc `vxl serve` sessions running under personal `sudo` accounts.
