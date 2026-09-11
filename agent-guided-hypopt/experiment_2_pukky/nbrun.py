#!/usr/bin/env python3
"""nbrun - execute ONE cell of a notebook in place, keeping a live kernel alive
between calls so later cells see earlier cells' variables (like pressing
Shift+Enter cell by cell in Jupyter).

Usage
-----
  python3 nbrun.py <notebook.ipynb> <cell_index>   # run cell (0-based), print its
                                                   # stdout/stderr, save in place
  python3 nbrun.py <notebook.ipynb> --stop         # shut down that notebook's kernel
  python3 nbrun.py <notebook.ipynb> --status

Why: `papermill x.ipynb x.ipynb` (or "Run All") executes the whole notebook in one
pass and cannot stop mid-way to let you fill a WRITE cell after an earlier cell
produced results. AG-HYPOPT trials must run cell-by-cell; this runner executes
exactly one cell per invocation against a persistent kernel (started independent
of this process so it survives between calls). Kernel state is tracked per
notebook under /tmp/nbrun/.
"""
import sys, os, json, hashlib, time, shutil, subprocess

STORE = "/tmp/nbrun"

def _usage():
    print(__doc__)
    sys.exit(2)

def _key(nb):
    return hashlib.sha1(os.path.abspath(nb).encode()).hexdigest()[:16]

def _state_path(nb):
    os.makedirs(STORE, exist_ok=True)
    return os.path.join(STORE, _key(nb) + ".json")

def _client(nb):
    from jupyter_client import BlockingKernelClient, KernelManager
    sp = _state_path(nb)
    if os.path.exists(sp):
        try:
            st = json.load(open(sp))
            if st.get("conn") and os.path.exists(st["conn"]):
                kc = BlockingKernelClient(connection_file=st["conn"])
                kc.load_connection_file()
                kc.start_channels()
                kc.wait_for_ready(timeout=15)
                return kc
        except Exception:
            pass
    # start a fresh, independent kernel for this notebook
    km = KernelManager(kernel_name="python3")
    cwd = os.path.dirname(os.path.abspath(nb)) or "."
    devnull = subprocess.DEVNULL
    try:
        km.start_kernel(cwd=cwd, independent=True, stdout=devnull, stderr=devnull)
    except TypeError:
        km.start_kernel(cwd=cwd, independent=True)
    stable = os.path.join(STORE, _key(nb) + ".conn.json")
    shutil.copyfile(km.connection_file, stable)     # survives this process' cleanup
    json.dump({"conn": stable}, open(sp, "w"))
    kc = km.client()
    kc.start_channels()
    kc.wait_for_ready(timeout=180)
    return kc

def _run(nb, idx):
    data = json.load(open(nb))
    cell = data["cells"][idx]
    if cell["cell_type"] != "code":
        print(f"cell {idx} is {cell['cell_type']}, not code", file=sys.stderr)
        sys.exit(3)
    src = "".join(cell["source"])
    kc = _client(nb)
    mid = kc.execute(src)
    outs = []
    while True:
        msg = kc.get_iopub_msg(timeout=14400)
        if msg["parent_header"].get("msg_id") != mid:
            continue
        t = msg["msg_type"]; c = msg["content"]
        if t == "stream":
            outs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
            (sys.stdout if c["name"] == "stdout" else sys.stderr).write(c["text"])
            (sys.stdout if c["name"] == "stdout" else sys.stderr).flush()
        elif t == "execute_result":
            outs.append({"output_type": "execute_result", "data": c["data"],
                         "metadata": c.get("metadata", {}), "execution_count": c.get("execution_count")})
        elif t == "display_data":
            outs.append({"output_type": "display_data", "data": c["data"], "metadata": c.get("metadata", {})})
        elif t == "error":
            outs.append({"output_type": "error", "ename": c["ename"], "evalue": c["evalue"],
                         "traceback": c["traceback"]})
            sys.stderr.write("\n".join(c["traceback"]) + "\n")
        elif t == "status" and c["execution_state"] == "idle":
            break
    cell["outputs"] = outs
    json.dump(data, open(nb, "w"), indent=1)
    kc.stop_channels()
    print(f"\n[nbrun] cell {idx} done -> {nb}")

def _stop(nb):
    from jupyter_client import BlockingKernelClient
    sp = _state_path(nb)
    if os.path.exists(sp):
        st = json.load(open(sp))
        try:
            kc = BlockingKernelClient(connection_file=st["conn"])
            kc.load_connection_file(); kc.start_channels()
            kc.shutdown(); time.sleep(1)
        except Exception:
            pass
        for f in (st.get("conn"), sp):
            try: os.remove(f)
            except Exception: pass
    print("[nbrun] stopped")

def _status(nb):
    from jupyter_client import BlockingKernelClient
    sp = _state_path(nb)
    if not os.path.exists(sp):
        print("[nbrun] no kernel"); return
    st = json.load(open(sp))
    try:
        kc = BlockingKernelClient(connection_file=st["conn"]); kc.load_connection_file()
        kc.start_channels(); kc.wait_for_ready(timeout=10)
        print("[nbrun] kernel alive"); kc.stop_channels()
    except Exception as e:
        print("[nbrun] kernel dead:", e)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        _usage()
    nb = sys.argv[1]
    if len(sys.argv) == 3 and sys.argv[2] == "--stop":
        _stop(nb)
    elif len(sys.argv) == 3 and sys.argv[2] == "--status":
        _status(nb)
    elif len(sys.argv) == 3:
        _run(nb, int(sys.argv[2]))
    else:
        _usage()
