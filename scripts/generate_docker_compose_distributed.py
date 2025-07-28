#!/usr/bin/env python3
"""
Generate docker-compose.distributed.yml for 2 workers + coordinator
Simple inheritance version - cleaner and easier to understand
"""

def generate_docker_compose_distributed(num_workers=2, output_file="docker-compose.distributed.yml"):
    """
    Generate docker-compose file for distributed training using simple inheritance
    
    Args:
        num_workers: Number of worker containers (default 2)
        output_file: Output file name
    
    Returns:
        str: Path to generated file
    """
    
    compose_content = f"""# Generated docker-compose for distributed RL training
# Architecture: 1 Coordinator + {num_workers} Workers
# Uses simple inheritance: DistributedWorker(BaseWorker)

services:
  coordinator:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rl-coordinator
    environment:
      - PYTHONPATH=/workspace/project
      - CUDA_VISIBLE_DEVICES=""
      - NUM_WORKERS={num_workers}
    volumes:
      - ./distributed_shared:/workspace/distributed_shared
      - ./logs:/workspace/logs
      - .:/workspace/project
    working_dir: /workspace/project
    command: >
      python scripts/coordinator_entry.py
      --shared_dir /workspace/distributed_shared
      --num_workers {num_workers}
      --check_interval 30
    networks:
      - rl-training
    restart: unless-stopped

"""

    # Generate worker services
    for i in range(num_workers):
        worker_service = f"""  worker-{i}:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: rl-worker-{i}
    environment:
      - PYTHONPATH=/workspace/project
      - CUDA_VISIBLE_DEVICES=0
      - ENV=${{ENV:-CartPole-v1}}
      - WORKER_ID={i}
    volumes:
      - ./distributed_shared:/workspace/distributed_shared
      - ./logs:/workspace/logs
      - .:/workspace/project
    working_dir: /workspace/project
    command: >
      python scripts/worker_entry.py
      --worker_id {i}
      --shared_dir /workspace/distributed_shared
      --env ${{ENV:-CartPole-v1}}
      --max_episodes 1000
    networks:
      - rl-training
    depends_on:
      - coordinator
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]

"""
        compose_content += worker_service

    # Add networks
    compose_content += """networks:
  rl-training:
    driver: bridge

volumes:
  distributed_shared:
    driver: local
"""

    # Write to file
    with open(output_file, 'w') as f:
        f.write(compose_content)
    
    print(f"Generated {output_file} with {num_workers} workers")
    print(f"Architecture: 1 Coordinator + {num_workers} DistributedWorkers")
    print(f"Uses simple inheritance: DistributedWorker(BaseWorker)")
    return output_file


def generate_docker_compose_simple(output_file="docker-compose.simple.yml"):
    """
    Generate simple docker-compose for testing (2 workers + coordinator)
    """
    return generate_docker_compose_distributed(num_workers=2, output_file=output_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate docker-compose for distributed training')
    parser.add_argument('--workers', type=int, default=2, help='Number of workers (default: 2)')
    parser.add_argument('--output', type=str, default='docker-compose.distributed.yml', help='Output file')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Default environment')
    
    args = parser.parse_args()
    
    print(f"🎯 Generating docker-compose configuration:")
    print(f"   Workers: {args.workers}")
    print(f"   Environment: {args.env}")
    print(f"   Output: {args.output}")
    print(f"   Architecture: Simple inheritance (DistributedWorker extends BaseWorker)")
    
    result = generate_docker_compose_distributed(args.workers, args.output)
    
    print(f"\n✅ Generated {result}")
    print(f"\n💡 Usage:")
    print(f"   ENV={args.env} docker-compose -f {args.output} up -d")
    print(f"   python scripts/distributed_cli.py start --workers {args.workers}")
    print(f"   make dist-start  # If using the new Makefile")