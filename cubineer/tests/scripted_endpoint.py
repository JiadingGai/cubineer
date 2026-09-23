"""Responses transport fixture; GPU outcomes are independently supplied to workers."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import uuid
import time
import shlex
import sys


class Endpoint:
    def __init__(self):
        self.requests = []
        self.errors = []
        self.failure_status = None
        self.candidate_failure = None
        self.candidate_delay = None
        self.candidate_delay_seconds = 0
        self.candidate_calls = 0
        self.candidate_lock = threading.Lock()
        self.delay = 0
        self.memory_failure = False
        self.memory_calls = 0
        self.headers = []
        self.workspace_probe = False
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                try:
                    request = json.loads(
                        self.rfile.read(int(self.headers["Content-Length"]))
                    )
                    endpoint.requests.append(request)
                    endpoint.headers.append(dict(self.headers))
                    schema = request["text"]["format"]["schema"]["properties"]
                    candidate_failure = False
                    request_delay = endpoint.delay
                    if "code" in schema:
                        with endpoint.candidate_lock:
                            endpoint.candidate_calls += 1
                            candidate_call = endpoint.candidate_calls
                            candidate_failure = (
                                candidate_call == endpoint.candidate_failure
                            )
                            if candidate_call == endpoint.candidate_delay:
                                request_delay = endpoint.candidate_delay_seconds
                    time.sleep(request_delay)
                    if endpoint.failure_status or candidate_failure:
                        body = json.dumps(
                            {
                                "error": {
                                    "message": "scripted capability failure",
                                    "type": "invalid_request_error",
                                }
                            }
                        ).encode()
                        self.send_response(endpoint.failure_status or 500)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    output = endpoint.output(request)
                    identity = "resp-" + uuid.uuid4().hex
                    events = [
                        {"type": "response.created", "response": {"id": identity}},
                        {"type": "response.output_item.done", "item": output},
                        {
                            "type": "response.completed",
                            "response": {
                                "id": identity,
                                "usage": {
                                    "input_tokens": 100,
                                    "output_tokens": 25,
                                    "total_tokens": 125,
                                },
                            },
                        },
                    ]
                    body = "".join(
                        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                        for event in events
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as error:
                    endpoint.errors.append(repr(error))
                    self.send_error(500, str(error))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def output(self, request):
        schema = request["text"]["format"]["schema"]["properties"]
        names = {tool.get("name") for tool in request.get("tools", [])}
        assert not names.intersection(
            {
                "spawn_agent",
                "send_message",
                "send_input",
                "multi_agent_v1",
                "multi_agent_v2",
            }
        )
        outputs = [
            item
            for item in request["input"]
            if item.get("type") == "function_call_output"
        ]
        if "final_bottleneck" in schema:
            if not outputs:
                return {
                    "type": "function_call",
                    "call_id": "reference-profile",
                    "name": "profile_reference_with_ncu",
                    "arguments": "{}",
                }
            result = {
                "invalid_reference_code": False,
                "math_operations": ["elementwise"],
                "dominant_operation": "addition",
                "problem_dimensions_str": "N=1024",
                "total_flops": 1024,
                "total_bytes": 12288,
                "compute_intensity": 1 / 12,
                "theoretical_bottleneck": "memory_bound",
                "actual_bottleneck": "memory_bound",
                "classification_match": True,
                "final_bottleneck": "memory_bound",
                "sm_throughput": 12,
                "dram_throughput": 80,
                "reference_latency_ms": 2,
                "occupancy": 50,
                "kernel_name": "vector_add",
                "achievable_speedup": 2,
                "recommended_techniques": ["memory_coalescing"],
                "analysis_notes": "Synthetic profiler evidence.",
            }
        elif "code" in schema:
            if self.workspace_probe and not outputs:
                script = 'from pathlib import Path; Path("local-proof.txt").write_text("inside"); Path("../escape-proof.txt").write_text("outside")'
                return {
                    "type": "function_call",
                    "call_id": "workspace-probe",
                    "name": "exec_command",
                    "arguments": json.dumps(
                        {
                            "cmd": shlex.join([sys.executable, "-c", script]),
                            "login": False,
                        }
                    ),
                }
            result = {
                "code": "# Scripted candidate; never GPU-validated.\nclass ModelNew:\n    pass\n",
                "approach": "fixture candidate",
                "confidence": 0.5,
            }
        elif "findings" in schema:
            self.memory_calls += 1
            result = (
                {
                    "findings": [
                        {
                            "category": "learnings",
                            "text": "unknown",
                            "evidence": "unseen",
                        }
                    ]
                }
                if self.memory_failure == self.memory_calls
                else {"findings": []}
            )
        else:
            if "memory_coalescing" in names and not outputs:
                return {
                    "type": "function_call",
                    "call_id": "analyzer",
                    "name": "memory_coalescing",
                    "arguments": "{}",
                }
            result = {
                "bottleneck_type": "memory",
                "severity": "moderate",
                "key_observations": ["Synthetic uncoalesced loads"],
                "optimization_recommendations": ["Use contiguous accesses"],
                "estimated_improvement": 2,
                "priority_score": 0.8,
                "summary": "Synthetic memory bottleneck",
            }
        return {
            "type": "message",
            "id": uuid.uuid4().hex,
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": json.dumps(result)}],
        }
