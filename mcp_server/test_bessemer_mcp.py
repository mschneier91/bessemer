#!/usr/bin/env python3
"""mcp_server_test: drive bessemer_mcp.py over stdio as an MCP client would.

usage: test_bessemer_mcp.py <run_case> <work dir> [mpi launcher...]

Checks what a user relies on: the handshake and tool list; the setup check
finds the build; deck keys come from the build; the examples are listed; a new
case from an example is written under the runs directory and checks out; a
misspelled key gets an error naming it and a missing boundary an error naming
the boundary; a run at one rank finishes with status ok, and its DOF counts are
the ones the check reported.
"""

import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


class Client:
    def __init__(self, env):
        self.p = subprocess.Popen([sys.executable, os.path.join(HERE, "bessemer_mcp.py")],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  env=env, universal_newlines=True)
        self.n = 0

    def request(self, method, params=None):
        self.n += 1
        msg = {"jsonrpc": "2.0", "id": self.n, "method": method}
        if params is not None:
            msg["params"] = params
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()
        reply = json.loads(self.p.stdout.readline())
        assert reply.get("id") == self.n, reply
        assert "error" not in reply, reply
        return reply["result"]

    def notify(self, method):
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.p.stdin.flush()

    def call(self, tool, **args):
        r = self.request("tools/call", {"name": tool, "arguments": args})
        if r["isError"]:
            return {"_error": r["content"][0]["text"]}
        return r["structuredContent"]

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=30)


def check(cond, what):
    if not cond:
        print("FAIL:", what)
        sys.exit(1)
    print("ok:", what)


def main():
    run_case, work = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    launcher = " ".join(sys.argv[3:]) or "mpirun"
    runs = os.path.join(work, "runs")
    shutil.rmtree(runs, ignore_errors=True)
    env = dict(os.environ, BESSEMER_RUN_CASE=run_case, BESSEMER_ENV="none",
               BESSEMER_MPIEXEC=launcher, BESSEMER_RUNS=runs)
    c = Client(env)

    init = c.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                    "clientInfo": {"name": "test", "version": "0"}})
    c.notify("notifications/initialized")
    check(init["protocolVersion"] == "2025-06-18" and init["instructions"],
          "initialize negotiates the version and gives instructions")
    names = {t["name"] for t in c.request("tools/list")["tools"]}
    check({"check_setup", "deck_keys", "new_case", "check_deck", "run_deck",
           "run_status", "read_results"} <= names, "the tools are listed")
    check(any(r["name"] == "using" for r in c.request("resources/list")["resources"]),
          "the docs are resources")

    setup = c.call("check_setup")
    check(setup.get("ready") and setup["run_case"] == run_case and setup["deck_keys"] > 50,
          "check_setup finds the build and its deck keys")

    keys = c.call("deck_keys", section="time")
    paths = {k["path"] for k in keys["keys"]}
    check("time.dt" in paths and "time.t_final" in paths
          and all(p == "time" or p.startswith("time.") for p in paths), "deck_keys searches a section")
    check(any("cfl" in k["path"] for k in c.call("deck_keys", query="cfl")["keys"]),
          "deck_keys searches descriptions and paths")

    ex = c.call("list_examples", query="cavity")
    check(any(e["path"] == os.path.join("cases", "cavity.yaml") for e in ex["examples"]),
          "list_examples finds the cavity deck")

    case = c.call("new_case", name="cav", template="cases/cavity.yaml",
                  set={"physics.Re": 50, "time.t_final": 0.05,
                       "discretization.order_u": 3})
    deck = os.path.join(runs, "cav", "cav.yaml")
    check(os.path.exists(deck), "new_case writes runs/<name>/<name>.yaml")
    text = open(deck).read()
    check("Re: 50" in text and "# Lid-driven cavity" in text,
          "new_case applies the changes and keeps the template's comments")
    rep = case["check"]
    check(rep["status"] == "ok" and rep["mesh"]["elements"] == 36
          and rep["orders"]["velocity"] == 3 and rep["dofs"]["velocity"] > 0
          and rep["dofs"]["pressure"] > 0, "the new case checks out, with its DOFs")
    check("estimate" in rep, "check_deck estimates the memory")

    bad = os.path.join(runs, "bad.yaml")
    with open(bad, "w") as f:
        f.write(text.replace("t_final:", "t_finale:"))
    rep = c.call("check_deck", path=bad)
    check(rep["status"] == "error" and "t_finale" in rep["error"],
          "a misspelled key gets an error naming it: " + rep.get("error", "")[:120])

    with open(bad, "w") as f:
        f.write(text.replace("[xmin, xmax, ymin]", "[xmin, xmax]"))
    rep = c.call("check_deck", path=bad)
    check(rep["status"] == "error" and "ymin" in rep["error"],
          "an uncovered boundary gets an error naming it: " + rep.get("error", "")[:120])

    run = c.call("run_deck", path=deck, ranks=1)
    check(run.get("started"), "run_deck starts the run")
    deadline = time.time() + 300
    while time.time() < deadline:
        st = c.call("run_status", id=run["id"])
        if st["state"] != "running":
            break
        time.sleep(1)
    check(st["state"] == "finished", "the run finishes: %s\n%s"
          % (st["state"], st.get("log_tail", "")))
    check(any(r["id"] == run["id"] for r in c.call("run_status")["runs"]),
          "run_status lists the run")
    res = c.call("read_results", id=run["id"])
    s = res["summary"]
    check(s["status"] == "ok", "read_results reads status ok")
    check(s["mesh"]["velocity_dofs"] == case["check"]["dofs"]["velocity"]
          and s["mesh"]["pressure_dofs"] == case["check"]["dofs"]["pressure"],
          "the run's DOFs are the ones check_deck reported")
    check(res["notes"], "read_results says what one run does not establish")
    c.close()
    print("mcp_server_test: all passed")


if __name__ == "__main__":
    main()
