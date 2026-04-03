"""
kgz CLI — execute code on Kaggle kernels from your terminal.

Usage:
    kgz run URL "print('hello')"
    kgz exec URL -f train.py
    kgz status URL
    kgz interrupt URL
    kgz wait URL
    kgz upload URL local_file [remote_path]
    kgz download URL remote_file [local_path]
    kgz ls URL [path]
    kgz info URL
    kgz snapshot URL
    kgz resources URL
    kgz sync URL local_dir [remote_dir]
    kgz notebook URL cells.txt -o output.ipynb
    kgz sessions
"""

import sys
import json
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="kgz",
        description="Execute code on remote Kaggle Jupyter kernels",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # kgz run URL "code"
    p = sub.add_parser("run", help="Execute code on the kernel")
    p.add_argument("url"); p.add_argument("code")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--json", action="store_true", help="Output as JSON")

    # kgz exec URL -f script.py
    p = sub.add_parser("exec", help="Execute a local Python file")
    p.add_argument("url"); p.add_argument("-f", "--file", required=True)
    p.add_argument("--timeout", type=int, default=600)

    # kgz status URL
    p = sub.add_parser("status", help="Check kernel state"); p.add_argument("url")

    # kgz interrupt URL
    p = sub.add_parser("interrupt", help="Interrupt execution"); p.add_argument("url")

    # kgz wait URL
    p = sub.add_parser("wait", help="Wait until idle")
    p.add_argument("url"); p.add_argument("--timeout", type=int, default=3600)

    # kgz restart URL
    p = sub.add_parser("restart", help="Restart kernel"); p.add_argument("url")

    # kgz upload URL local [remote]
    p = sub.add_parser("upload", help="Upload file")
    p.add_argument("url"); p.add_argument("local"); p.add_argument("remote", nargs="?")

    # kgz download URL remote [local]
    p = sub.add_parser("download", help="Download file")
    p.add_argument("url"); p.add_argument("remote"); p.add_argument("local", nargs="?")

    # kgz ls URL [path]
    p = sub.add_parser("ls", help="List remote files")
    p.add_argument("url"); p.add_argument("path", nargs="?", default="")

    # kgz info URL
    p = sub.add_parser("info", help="Show kernel info"); p.add_argument("url")

    # kgz snapshot URL
    p = sub.add_parser("snapshot", help="Inspect remote variables"); p.add_argument("url")

    # kgz resources URL
    p = sub.add_parser("resources", help="GPU/TPU/CPU usage"); p.add_argument("url")

    # kgz sync URL local_dir [remote_dir]
    p = sub.add_parser("sync", help="Watch & sync local dir to kernel")
    p.add_argument("url"); p.add_argument("local_dir")
    p.add_argument("remote_dir", nargs="?", default="")

    # kgz notebook URL -f cells.txt -o output.ipynb
    p = sub.add_parser("notebook", help="Run cells from file as notebook")
    p.add_argument("url"); p.add_argument("-f", "--file", required=True)
    p.add_argument("-o", "--output", default="output.ipynb")

    # kgz sessions
    sub.add_parser("sessions", help="List saved sessions")

    args = parser.parse_args()

    from kgz.kernel import Kernel
    from kgz import file_ops

    if args.command == "run":
        k = Kernel(args.url)
        result = k.execute(args.code, timeout=args.timeout, stream=not args.no_stream and not args.json)
        if args.json:
            print(json.dumps({"success": result.success, "stdout": result.stdout,
                              "return_value": result.return_value, "error": result.error_name,
                              "elapsed": result.elapsed_seconds}, indent=2))

    elif args.command == "exec":
        k = Kernel(args.url)
        with open(args.file) as f:
            code = f.read()
        k.execute(code, timeout=args.timeout)

    elif args.command == "status":
        print(Kernel(args.url).status())

    elif args.command == "interrupt":
        Kernel(args.url).interrupt(); print("Interrupted")

    elif args.command == "wait":
        k = Kernel(args.url)
        print("Waiting..."); k.wait(timeout=args.timeout); print("Idle!")

    elif args.command == "restart":
        Kernel(args.url).restart(); print("Restarted")

    elif args.command == "upload":
        p = file_ops.upload_file(args.url, args.local, args.remote)
        print(f"Uploaded to {p}")

    elif args.command == "download":
        p = file_ops.download_file(args.url, args.remote, args.local)
        print(f"Downloaded to {p}")

    elif args.command == "ls":
        for f in file_ops.list_files(args.url, args.path):
            kind = "d" if f["type"] == "directory" else "-"
            print(f"{kind} {f.get('size', 0):>10}  {f['name']}")

    elif args.command == "info":
        k = Kernel(args.url)
        import urllib.request
        api = json.loads(urllib.request.urlopen(f"{k._http_url}/api").read())
        print(f"Jupyter:  v{api.get('version', '?')}")
        print(f"Kernel:   {k.kernel_id}")
        print(f"Status:   {k.status()}")
        print(f"Session:  {k.name}")

    elif args.command == "snapshot":
        snap = Kernel(args.url).snapshot()
        for name, info in snap.items():
            shape = info.get("shape", "")
            print(f"  {name:20s} {info['type']:15s} {shape}")

    elif args.command == "resources":
        res = Kernel(args.url).resources()
        print(json.dumps(res, indent=2))

    elif args.command == "sync":
        from kgz.sync import FileSync
        sync = FileSync(args.url, args.local_dir, args.remote_dir)
        print(f"Syncing {args.local_dir} → {args.remote_dir or '/'}")
        initial = sync.push()
        print(f"Initial upload: {len(initial)} files")
        print("Watching for changes (Ctrl-C to stop)...")
        try:
            sync.start()
            while True:
                import time; time.sleep(1)
        except KeyboardInterrupt:
            sync.stop()
            print("\nSync stopped.")

    elif args.command == "notebook":
        k = Kernel(args.url)
        with open(args.file) as f:
            cells = [c.strip() for c in f.read().split("# %%") if c.strip()]
        results = k.execute_notebook(cells)
        k.to_notebook(args.output)
        passed = sum(1 for r in results if r.success)
        print(f"\n{passed}/{len(results)} cells passed. Saved to {args.output}")

    elif args.command == "sessions":
        import time as _time
        sessions = Kernel.list_sessions()
        if not sessions:
            print("No saved sessions. Use k.save_session() to save one.")
        for s in sessions:
            age = int(_time.time() - s.get("saved_at", 0))
            print(f"  {s['name']:20s} {age//3600}h ago  history: {s.get('history_len', 0)}")


if __name__ == "__main__":
    import time
    main()
