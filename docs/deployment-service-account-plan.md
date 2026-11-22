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

## 6. Easy Zero-`sudo` Deployment Workflow (stored in `voxel-private`)

Two clean patterns for deploying a new `mobi` commit without general `sudo`:

- **Option 1 — Scoped restart helper (recommended with system unit):**
  - Operators in group `voxel` (or a deploy key) run `/opt/voxel/deploy.sh <git-ref>` as `lm1-service`, which:
    1. Fetches and checks out `<git-ref>` in `/opt/voxel`
    2. Runs `uv sync --locked --all-packages --extra web --extra zaber` (plus rig hardware extras)
    3. Rebuilds the web SPA (`pnpm -C web install --frozen-lockfile && pnpm -C web/station build`)
    4. Calls a single narrow `sudoers` rule (`%voxel ALL=(root) NOPASSWD: /usr/bin/systemctl restart vxl-lm1.service, /usr/bin/systemctl status vxl-lm1.service`) — no shell, no arbitrary `sudo`.
- **Option 2 — `systemd --user` service (`loginctl enable-linger lm1-service`):**
  - Zero `sudoers` entries at all; `lm1-service` manages its own unit via `systemctl --user restart vxl.service`. Slightly less kernel sandboxing (`ProtectSystem=strict` unavailable in user units), so Option 1 is preferable for a hardware rig.

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
