"""Bubblewrap command execution. There is deliberately no host fallback."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .types import Profile


class SandboxError(RuntimeError):
    pass


class CommandCancelled(InterruptedError):
    def __init__(self, result):
        super().__init__("command cancelled; partial output retained, execution unverified")
        self.result = result


class SandboxRunner:
    def __init__(self, profile: Profile, dependencies: list[str] | None = None, cancelled=lambda: False):
        self.profile = profile
        self.dependencies = dependencies or []
        self.cancelled = cancelled
        self.protected_files = {}

    def command(self, root: Path, argv: list[str], cwd: str = ".") -> list[str]:
        if not argv or any(not isinstance(arg, str) or "\0" in arg for arg in argv):
            raise ValueError("command must be a nonempty argument array")
        relative = Path(cwd)
        if relative.is_absolute() or ".." in relative.parts or not (root/relative).resolve().is_relative_to(root.resolve()):
            raise ValueError("command cwd escapes workspace")
        if not (root/relative).is_dir():
            raise ValueError("command cwd does not exist")
        if not self.profile.allow_install and (
            Path(argv[0]).name.startswith("pip") or ("-m" in argv and "pip" in argv)
            or (Path(argv[0]).name in {"npm", "yarn", "pnpm"} and any(a in {"install", "add"} for a in argv[1:]))):
            raise PermissionError("dependency installation requires an explicitly enabled profile")
        command = ["bwrap", "--die-with-parent", "--new-session", "--unshare-all"]
        if self.profile.allow_network:
            command.append("--share-net")
        for path in ("/usr", "/bin", "/sbin", "/lib", "/lib64"):
            if Path(path).exists():
                command.extend(["--ro-bind", path, path])
        for path in ("/etc/ld.so.cache", "/etc/localtime"):
            if Path(path).exists():
                command.extend(["--ro-bind", path, path])
        if self.profile.allow_network:
            for path in ("/etc/resolv.conf", "/etc/ssl/certs"):
                if Path(path).exists():
                    command.extend(["--ro-bind", path, path])
        command.extend(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                        "--dir", "/home/sandbox", "--bind", str(root), "/workspace"])
        for dependency in sorted(set([sys.prefix, *self.dependencies])):
            p = Path(dependency).resolve()
            if (not p.exists() or Path.home().is_relative_to(p) or p in map(Path,("/tmp","/proc","/dev","/home/sandbox"))
                    or p.is_relative_to("/workspace")):
                raise ValueError("dependency mount must be a specific existing toolchain/dependency path")
            if not str(p).startswith("/usr/"):
                command.extend(["--ro-bind", str(p), str(p)])
        for target, source in self.protected_files.items():
            command.extend(["--ro-bind", str(source), target])
        git_path = root / ".git"
        if git_path.exists():
            command.extend(["--ro-bind", str(git_path), "/workspace/.git"])
        command.extend(["--chdir", str(Path("/workspace")/relative), "--clearenv",
                        "--setenv", "PATH", f"{sys.prefix}/bin:/usr/bin:/bin",
                        "--setenv", "HOME", "/home/sandbox", "--setenv", "TMPDIR", "/tmp",
                        "--setenv", "LANG", "C.UTF-8", "--setenv", "PYTHONDONTWRITEBYTECODE", "1"])
        wrapper = (
            "import os,resource,sys; "
            "resource.setrlimit(resource.RLIMIT_CORE,(0,0)); "
            "resource.setrlimit(resource.RLIMIT_AS,(4294967296,4294967296)); "
            "resource.setrlimit(resource.RLIMIT_FSIZE,(134217728,134217728)); "
            "resource.setrlimit(resource.RLIMIT_NOFILE,(256,256)); "
            "os.execvp(sys.argv[1],sys.argv[1:])"
        )
        return [*command, sys.executable, "-I", "-c", wrapper, *argv]

    def run(self, root: Path, argv: list[str], *, cwd: str = ".", timeout: float | None = None) -> dict:
        command = self.command(root, argv, cwd)
        started = time.monotonic()
        timed_out = False
        cancelled = False
        with tempfile.TemporaryFile() as output:
            try:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                           env={"PATH": "/usr/bin:/bin"}, start_new_session=True)
            except OSError as exc:
                raise SandboxError(f"sandbox could not start: {exc}") from exc
            try:
                while process.poll() is None:
                    if self.cancelled():
                        cancelled = True
                        break
                    if time.monotonic()-started >= min(timeout or self.profile.command_timeout_seconds,
                                                       self.profile.command_timeout_seconds):
                        timed_out = True
                        break
                    time.sleep(0.05)
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            output.seek(0, os.SEEK_END)
            length = output.tell()
            output.seek(0)
            text = output.read(1024 * 1024).decode(errors="replace")
            if length > 1024*1024:
                output.seek(max(0, length-65536))
                text += "\n[output truncated]\n" + output.read().decode(errors="replace")
            if text.startswith("bwrap:") and process.returncode != 0:
                raise SandboxError(text[:2000])
            result = {"argv": argv, "cwd": cwd, "exit_code": process.returncode,
                    "timed_out": timed_out, "output": text, "output_truncated": length>1024*1024,
                    "elapsed_seconds": round(time.monotonic()-started, 3)}
            if cancelled:
                raise CommandCancelled({**result, "cancelled": True})
            return result
