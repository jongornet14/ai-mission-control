#!/bin/bash

# Worker Health Check Script for Distributed RL Training
# Comprehensive monitoring and verification

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}   Distributed Worker Health Check${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_container_status() {
    echo -e "${PURPLE}📊 CONTAINER STATUS CHECK${NC}"
    echo "================================"
    
    # Get all worker containers
    WORKERS=$(docker ps --filter "name=ai-mission-control-worker" --format "{{.Names}}" 2>/dev/null || echo "")
    WORKER_COUNT=$(echo "$WORKERS" | grep -c "worker" 2>/dev/null || echo "0")
    
    if [ "$WORKER_COUNT" -eq 0 ]; then
        print_error "No worker containers found running"
        echo "💡 Try: make distributed-up"
        return 1
    fi
    
    print_success "Found $WORKER_COUNT active worker containers"
    
    # Check each worker status
    echo ""
    echo "Worker Container Details:"
    docker ps --filter "name=ai-mission-control-worker" --format "table {{.Names}}\t{{.Status}}\t{{.RunningFor}}\t{{.Ports}}"
    
    # Check for restart loops
    echo ""
    RESTARTING=$(docker ps --filter "name=ai-mission-control-worker" --format "{{.Status}}" | grep -c "Restarting" 2>/dev/null || echo "0")
    if [ "$RESTARTING" -gt 0 ]; then
        print_error "$RESTARTING workers are in restart loop"
        echo "💡 Check logs: make distributed-logs-workers"
        return 1
    else
        print_success "No workers in restart loop"
    fi
    
    return 0
}

check_worker_processes() {
    echo ""
    echo -e "${PURPLE}🔍 WORKER PROCESS CHECK${NC}"
    echo "================================"
    
    WORKERS=$(docker ps --filter "name=ai-mission-control-worker" --format "{{.Names}}" 2>/dev/null)
    
    if [ -z "$WORKERS" ]; then
        print_error "No workers to check"
        return 1
    fi
    
    for worker in $WORKERS; do
        echo ""
        echo "--- $worker ---"
        
        # Check if Python process is running
        PYTHON_PROC=$(docker exec "$worker" pgrep -f "distributed_worker.py" 2>/dev/null || echo "")
        if [ -n "$PYTHON_PROC" ]; then
            print_success "Python training process active (PID: $PYTHON_PROC)"
        else
            print_error "No distributed_worker.py process found"
            continue
        fi
        
        # Check CPU usage
        CPU_USAGE=$(docker exec "$worker" top -bn1 | grep "distributed_worker" | head -1 | awk '{print $9}' 2>/dev/null || echo "0")
        if [ -n "$CPU_USAGE" ] && [ "$CPU_USAGE" != "0" ]; then
            print_success "CPU usage: ${CPU_USAGE}%"
        else
            print_warning "Low/no CPU usage detected"
        fi
        
        # Check memory usage
        MEM_USAGE=$(docker exec "$worker" ps aux | grep "distributed_worker" | grep -v grep | awk '{print $4}' | head -1 2>/dev/null || echo "0")
        if [ -n "$MEM_USAGE" ] && [ "$MEM_USAGE" != "0" ]; then
            print_success "Memory usage: ${MEM_USAGE}%"
        else
            print_warning "Low memory usage detected"
        fi
    done
    
    return 0
}

check_gpu_usage() {
    echo ""
    echo -e "${PURPLE}🖥️  GPU USAGE CHECK${NC}"
    echo "================================"
    
    FIRST_WORKER=$(docker ps --filter "name=ai-mission-control-worker" --format "{{.Names}}" | head -1)
    
    if [ -z "$FIRST_WORKER" ]; then
        print_error "No workers available for GPU check"
        return 1
    fi
    
    # Check GPU availability
    GPU_INFO=$(docker exec "$FIRST_WORKER" nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || echo "")
    
    if [ -n "$GPU_INFO" ]; then
        print_success "GPU accessible from workers"
        echo "$GPU_INFO"
        
        # Check if GPU memory is being used
        GPU_MEM_USED=$(echo "$GPU_INFO" | tail -1 | cut -d',' -f3 | tr -d ' MiB')
        if [ "$GPU_MEM_USED" -gt 100 2>/dev/null ]; then
            print_success "GPU memory in use: ${GPU_MEM_USED}MiB"
        else
            print_warning "Low GPU memory usage: ${GPU_MEM_USED}MiB"
        fi
    else
        print_error "Cannot access GPU from workers"
        return 1
    fi
    
    return 0
}

check_log_generation() {
    echo ""
    echo -e "${PURPLE}📝 LOG GENERATION CHECK${NC}"
    echo "================================"
    
    # Check if shared directory exists
    if [ ! -d "distributed_shared" ]; then
        print_error "distributed_shared directory not found"
        return 1
    fi
    
    print_success "distributed_shared directory exists"
    
    # Check for worker logs
    if [ -d "distributed_shared/worker_logs" ]; then
        WORKER_LOG_DIRS=$(find distributed_shared/worker_logs -name "worker_*" -type d 2>/dev/null | wc -l)
        if [ "$WORKER_LOG_DIRS" -gt 0 ]; then
            print_success "Found $WORKER_LOG_DIRS worker log directories"
            
            # Check for TensorBoard logs
            TB_LOGS=$(find distributed_shared/worker_logs -name "tensorboard" -type d 2>/dev/null | wc -l)
            if [ "$TB_LOGS" -gt 0 ]; then
                print_success "TensorBoard logs found in $TB_LOGS workers"
            else
                print_warning "No TensorBoard logs found yet"
            fi
            
            # Check for recent log activity (last 5 minutes)
            RECENT_LOGS=$(find distributed_shared/worker_logs -name "*.log" -newermt "5 minutes ago" 2>/dev/null | wc -l)
            if [ "$RECENT_LOGS" -gt 0 ]; then
                print_success "Recent log activity detected ($RECENT_LOGS files)"
            else
                print_warning "No recent log activity (last 5 minutes)"
            fi
        else
            print_warning "No worker log directories found yet"
        fi
    else
        print_warning "worker_logs directory not found yet"
    fi
    
    # Check for model checkpoints
    if [ -d "distributed_shared/models" ]; then
        MODEL_COUNT=$(find distributed_shared/models -name "*.pt" 2>/dev/null | wc -l)
        if [ "$MODEL_COUNT" -gt 0 ]; then
            print_success "Found $MODEL_COUNT model checkpoints"
        else
            print_warning "No model checkpoints found yet"
        fi
    fi
    
    return 0
}

check_training_progress() {
    echo ""
    echo -e "${PURPLE}📈 TRAINING PROGRESS CHECK${NC}"
    echo "================================"
    
    # Check coordinator logs for training progress
    COORD_LOGS=$(docker logs rl-coordinator --tail=20 2>/dev/null | grep -E "(BEST WORKER|Episode.*Reward|✅)" | tail -5)
    
    if [ -n "$COORD_LOGS" ]; then
        print_success "Recent coordinator activity:"
        echo "$COORD_LOGS"
    else
        print_warning "No recent coordinator training activity found"
    fi
    
    # Check worker performance metrics
    if [ -d "distributed_shared/metrics" ]; then
        METRICS_FILES=$(find distributed_shared/metrics -name "worker_*_performance.json" 2>/dev/null | wc -l)
        if [ "$METRICS_FILES" -gt 0 ]; then
            print_success "Found performance metrics for $METRICS_FILES workers"
            
            # Show latest performance data
            echo ""
            echo "Latest Worker Performance:"
            for metrics_file in distributed_shared/metrics/worker_*_performance.json; do
                if [ -f "$metrics_file" ]; then
                    WORKER_ID=$(basename "$metrics_file" | sed 's/worker_\([0-9]*\)_performance.json/\1/')
                    AVG_REWARD=$(python3 -c "import json; data=json.load(open('$metrics_file')); print(f'{data[\"avg_reward\"]:.2f}')" 2>/dev/null || echo "N/A")
                    EPISODES=$(python3 -c "import json; data=json.load(open('$metrics_file')); print(data[\"total_episodes\"])" 2>/dev/null || echo "N/A")
                    echo "  Worker $WORKER_ID: $EPISODES episodes, avg reward: $AVG_REWARD"
                fi
            done
        else
            print_warning "No worker performance metrics found yet"
        fi
    fi
    
    return 0
}

check_worker_communication() {
    echo ""
    echo -e "${PURPLE}🔄 WORKER COMMUNICATION CHECK${NC}"
    echo "================================"
    
    # Check for coordinator signals
    SIGNAL_COUNT=$(find distributed_shared -name "update_worker_*.signal" 2>/dev/null | wc -l)
    if [ "$SIGNAL_COUNT" -gt 0 ]; then
        print_success "Found $SIGNAL_COUNT coordination signals"
    else
        print_info "No active coordination signals (normal if recently synced)"
    fi
    
    # Check for best model
    if [ -f "distributed_shared/best_model/current_best.pt" ]; then
        MODEL_AGE=$(stat -c %Y "distributed_shared/best_model/current_best.pt" 2>/dev/null)
        CURRENT_TIME=$(date +%s)
        AGE_MINUTES=$(( (CURRENT_TIME - MODEL_AGE) / 60 ))
        
        if [ "$AGE_MINUTES" -lt 30 ]; then
            print_success "Best model is recent (${AGE_MINUTES} minutes old)"
        else
            print_warning "Best model is old (${AGE_MINUTES} minutes old)"
        fi
    else
        print_warning "No best model found yet"
    fi
    
    return 0
}

run_quick_worker_test() {
    echo ""
    echo -e "${PURPLE}🧪 QUICK WORKER FUNCTIONALITY TEST${NC}"
    echo "================================"
    
    print_info "Testing worker script execution..."
    
    # Test worker script can run
    TEST_OUTPUT=$(docker run --rm --gpus all \
        -v "$(pwd)":/workspace/project \
        -v "$(pwd)"/distributed_shared:/workspace/distributed_shared \
        ai-mission-control:latest \
        timeout 30 python /workspace/project/scripts/distributed_worker.py \
            --worker_id 99 \
            --shared_dir /workspace/distributed_shared \
            --env CartPole-v1 \
            --max_episodes 1 2>&1 || echo "TEST_FAILED")
    
    if echo "$TEST_OUTPUT" | grep -q "TEST_FAILED"; then
        print_error "Worker script test failed"
        echo "Error output:"
        echo "$TEST_OUTPUT" | tail -10
        return 1
    elif echo "$TEST_OUTPUT" | grep -q -E "(🚀|Episode.*Reward|✅)"; then
        print_success "Worker script test passed"
        return 0
    else
        print_warning "Worker script test inconclusive"
        echo "Last few lines of output:"
        echo "$TEST_OUTPUT" | tail -5
        return 1
    fi
}

generate_health_summary() {
    echo ""
    echo -e "${BLUE}📊 HEALTH SUMMARY${NC}"
    echo "================================"
    
    TOTAL_CHECKS=6
    PASSED_CHECKS=0
    
    # Re-run quick checks for summary
    if docker ps --filter "name=ai-mission-control-worker" --format "{{.Names}}" | grep -q worker; then
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        print_success "Container Status: HEALTHY"
    else
        print_error "Container Status: FAILED"
    fi
    
    if [ -d "distributed_shared/worker_logs" ] && [ "$(find distributed_shared/worker_logs -name "worker_*" -type d | wc -l)" -gt 0 ]; then
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        print_success "Log Generation: WORKING"
    else
        print_error "Log Generation: FAILED"
    fi
    
    if [ -d "distributed_shared/models" ] && [ "$(find distributed_shared/models -name "*.pt" | wc -l)" -gt 0 ]; then
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        print_success "Model Checkpoints: FOUND"
    else
        print_warning "Model Checkpoints: NOT YET"
    fi
    
    if [ -d "distributed_shared/metrics" ] && [ "$(find distributed_shared/metrics -name "*.json" | wc -l)" -gt 0 ]; then
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        print_success "Performance Metrics: AVAILABLE"
    else
        print_warning "Performance Metrics: NOT YET"
    fi
    
    echo ""
    echo -e "${BLUE}Overall Health: $PASSED_CHECKS/$TOTAL_CHECKS checks passed${NC}"
    
    if [ "$PASSED_CHECKS" -ge 4 ]; then
        print_success "Workers appear to be functioning correctly! 🎉"
        echo ""
        echo "💡 Next steps:"
        echo "  - Monitor progress: make distributed-logs"
        echo "  - View TensorBoard: make distributed-tensorboard"
        echo "  - Check GPU usage: make distributed-watch-gpu"
    elif [ "$PASSED_CHECKS" -ge 2 ]; then
        print_warning "Workers are partially functional - may need more time"
        echo ""
        echo "💡 Suggested actions:"
        echo "  - Wait 5-10 minutes for training to stabilize"
        echo "  - Check logs: make distributed-logs-workers"
        echo "  - Monitor: watch -n 10 'make distributed-worker-count'"
    else
        print_error "Workers appear to have issues"
        echo ""
        echo "💡 Troubleshooting:"
        echo "  - Check logs: make distributed-logs-workers"
        echo "  - Restart: make distributed-down && make distributed-up"
        echo "  - Manual test: Run quick worker functionality test above"
    fi
}

main() {
    print_header
    
    check_container_status && echo ""
    check_worker_processes && echo ""
    check_gpu_usage && echo ""
    check_log_generation && echo ""
    check_training_progress && echo ""
    check_worker_communication && echo ""
    
    # Optional quick test (uncomment if needed)
    # run_quick_worker_test && echo ""
    
    generate_health_summary
}

# Run main function
main "$@"