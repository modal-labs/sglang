#!/usr/bin/env python3
import argparse
import subprocess
import sys
import tempfile
import yaml
from typing import List, Dict, Any

def create_mprocs_config(decode_servers: List[Dict[str, Any]], prefill_servers: List[Dict[str, Any]], lb_config: Dict[str, Any] = None) -> str:
    """Create mprocs configuration file"""
    import yaml
    
    procs = {}
    
    # Add decode servers
    for i, server in enumerate(decode_servers):
        gpu_id = server['gpu_id']
        port = server['port']
        cuda_graph_max_bs = server['cuda_graph_max_bs']
        
        cmd = f"python3 -m sglang.launch_server --model Qwen/Qwen3-8B --speculative-algorithm EAGLE3 --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 --speculative-num-steps 6 --speculative-eagle-topk 10 --speculative-num-draft-tokens 32 --mem-fraction 0.9 --cuda-graph-max-bs {cuda_graph_max_bs} --dtype bfloat16 --disaggregation-mode decode --disaggregation-transfer-backend nixl --port {port} --attention-backend fa3"
        
        procs[f"decode-{i}"] = {
            "shell": cmd,
            "env": {
                "CUDA_VISIBLE_DEVICES": str(gpu_id)
            }
        }
    
    # Add prefill servers
    for i, server in enumerate(prefill_servers):
        gpu_id = server['gpu_id']
        port = server['port']
        bootstrap_port = server['bootstrap_port']
        cuda_graph_max_bs = server['cuda_graph_max_bs']
        
        cmd = f"python3 -m sglang.launch_server --model Qwen/Qwen3-8B --speculative-algorithm EAGLE3 --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 --speculative-num-steps 6 --speculative-eagle-topk 10 --speculative-num-draft-tokens 32 --mem-fraction 0.9 --cuda-graph-max-bs {cuda_graph_max_bs} --dtype bfloat16 --disaggregation-mode prefill --disaggregation-transfer-backend nixl --disable-overlap-schedule --port {port} --disaggregation-bootstrap-port {bootstrap_port}"
        
        procs[f"prefill-{i}"] = {
            "shell": cmd,
            "env": {
                "CUDA_VISIBLE_DEVICES": str(gpu_id)
            }
        }
    
    # Add load balancer
    if lb_config:
        prefill_urls = lb_config['prefill_urls']
        decode_urls = lb_config['decode_urls']
        bootstrap_ports = lb_config['bootstrap_ports']
        lb_host = lb_config['lb_host']
        lb_port = lb_config['lb_port']
        
        cmd_parts = ["python -m sglang.srt.disaggregation.mini_lb"]
        cmd_parts.append("--prefill")
        cmd_parts.extend(prefill_urls)
        cmd_parts.append("--decode")
        cmd_parts.extend(decode_urls)
        cmd_parts.extend(["--host", lb_host, "--port", str(lb_port)])
        cmd_parts.append("--prefill-bootstrap-ports")
        cmd_parts.extend([str(port) for port in bootstrap_ports])
        
        cmd = " ".join(cmd_parts)
        
        procs["load-balancer"] = {
            "shell": cmd
        }
    
    config = {"procs": procs}
    return yaml.dump(config, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(description="Launch SGLang prefill and decode servers using mprocs")
    parser.add_argument("--num-prefill", type=int, default=1, help="Number of prefill servers to launch")
    parser.add_argument("--num-decode", type=int, default=1, help="Number of decode servers to launch")
    parser.add_argument("--start-gpu", type=int, default=0, help="Starting GPU ID")
    parser.add_argument("--start-port", type=int, default=30000, help="Starting port number")
    parser.add_argument("--start-bootstrap-port", type=int, default=20000, help="Starting bootstrap port number")
    parser.add_argument("--cuda-graph-max-bs", type=int, default=16, help="CUDA graph max batch size for both prefill and decode servers")
    parser.add_argument("--lb-host", type=str, default="0.0.0.0", help="Load balancer host")
    parser.add_argument("--lb-port", type=int, default=8000, help="Load balancer port")
    
    args = parser.parse_args()
    
    current_gpu = args.start_gpu
    current_port = args.start_port
    current_bootstrap_port = args.start_bootstrap_port
    
    decode_servers = []
    prefill_servers = []
    decode_urls = []
    prefill_urls = []
    bootstrap_ports = []
    
    # Configure decode servers
    for i in range(args.num_decode):
        decode_servers.append({
            'gpu_id': current_gpu,
            'port': current_port,
            'cuda_graph_max_bs': args.cuda_graph_max_bs
        })
        decode_urls.append(f"http://127.0.0.1:{current_port}")
        current_gpu += 1
        current_port += 1
    
    # Configure prefill servers
    for i in range(args.num_prefill):
        prefill_servers.append({
            'gpu_id': current_gpu,
            'port': current_port,
            'bootstrap_port': current_bootstrap_port,
            'cuda_graph_max_bs': args.cuda_graph_max_bs
        })
        prefill_urls.append(f"http://127.0.0.1:{current_port}")
        bootstrap_ports.append(current_bootstrap_port)
        current_gpu += 1
        current_port += 1
        current_bootstrap_port += 1
    
    # Configure load balancer
    lb_config = None
    if args.num_decode > 0 or args.num_prefill > 0:
        lb_config = {
            'prefill_urls': prefill_urls,
            'decode_urls': decode_urls,
            'bootstrap_ports': bootstrap_ports,
            'lb_host': args.lb_host,
            'lb_port': args.lb_port
        }
    
    # Create mprocs config file
    config_content = create_mprocs_config(decode_servers, prefill_servers, lb_config)
    
    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        config_file = f.name
    
    print(f"Generated mprocs config file: {config_file}")
    print(f"Launching {args.num_decode} decode servers and {args.num_prefill} prefill servers")
    if lb_config:
        print(f"Load balancer will be available at http://{args.lb_host}:{args.lb_port}")
    
    try:
        # Launch mprocs with the config file
        subprocess.run(["mprocs", "--config", config_file], check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except FileNotFoundError:
        print("Error: mprocs not found. Please install mprocs first:")
        print("cargo install mprocs")
        sys.exit(1)
    finally:
        # Clean up config file
        try:
            os.unlink(config_file)
        except:
            pass

if __name__ == "__main__":
    main()