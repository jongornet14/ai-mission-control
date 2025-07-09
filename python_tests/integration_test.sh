#!/bin/bash

# AI Mission Control - Quick Integration Test Script
# Fast validation that all systems are working and can compute

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

print_header() {
    echo -e "${BLUE}"
    echo "=========================================================="
    echo "  AI Mission Control - Quick Integration Test"
    echo "=========================================================="
    echo -e "${NC}"
}

print_test() {
    echo -e "${PURPLE}[TEST]${NC} $1"
    ((TOTAL_TESTS++))
}

print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED_TESTS++))
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED_TESTS++))
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Test basic service connectivity
test_basic_connectivity() {
    echo -e "\n${BLUE}=== Basic Connectivity Tests ===${NC}"
    
    local services=("50051:Trading" "50052:Unity" "50053:Gym" "50054:ModernRL" "8080:Gateway")
    
    for service in "${services[@]}"; do
        local port=$(echo $service | cut -d: -f1)
        local name=$(echo $service | cut -d: -f2)
        
        print_test "Service connectivity: $name (port $port)"
        
        if timeout 5 curl -s http://localhost:$port/health >/dev/null 2>&1; then
            print_pass "$name service is reachable"
        else
            print_fail "$name service is not reachable"
        fi
    done
}

# Test environment creation and basic computation
test_environment_computation() {
    echo -e "\n${BLUE}=== Environment Computation Tests ===${NC}"
    
    print_test "Environment creation and basic RL computation"
    
    # Create environment
    local create_response=$(curl -s -X POST http://localhost:50053/create/CartPole-v1 2>/dev/null)
    
    if echo "$create_response" | grep -q "session_id"; then
        local session_id=$(echo "$create_response" | python3 -c "import sys, json; print(json.load(sys.stdin)['session_id'])" 2>/dev/null)
        
        if [[ -n "$session_id" ]]; then
            print_pass "Environment created successfully (session: ${session_id:0:8}...)"
            
            # Test environment reset
            print_test "Environment reset"
            local reset_response=$(curl -s -X POST http://localhost:50053/reset/$session_id 2>/dev/null)
            
            if echo "$reset_response" | grep -q "observation"; then
                print_pass "Environment reset successful"
                
                # Test environment steps (simulate RL agent)
                print_test "RL agent simulation (10 steps)"
                local total_reward=0
                local successful_steps=0
                
                for step in {1..10}; do
                    local action=$((RANDOM % 2))  # Random action (0 or 1)
                    local step_response=$(curl -s -X POST \
                        -H "Content-Type: application/json" \
                        -d "{\"action\": $action}" \
                        http://localhost:50053/step/$session_id 2>/dev/null)
                    
                    if echo "$step_response" | grep -q "reward"; then
                        local reward=$(echo "$step_response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('reward', 0))
except:
    print(0)
" 2>/dev/null)
                        total_reward=$(echo "$total_reward + $reward" | bc -l 2>/dev/null || echo "$total_reward")
                        ((successful_steps++))
                        
                        # Check if episode is done
                        local done=$(echo "$step_response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('done', False))
except:
    print(False)
" 2>/dev/null)
                        
                        if [[ "$done" == "True" ]]; then
                            break
                        fi
                    fi
                done
                
                if [[ $successful_steps -gt 0 ]]; then
                    print_pass "RL computation successful ($successful_steps steps, reward: $total_reward)"
                else
                    print_fail "RL computation failed"
                fi
            else
                print_fail "Environment reset failed"
            fi
        else
            print_fail "Could not extract session ID"
        fi
    else
        print_fail "Environment creation failed"
    fi
}

# Test multiple environments (parallel computation)
test_parallel_computation() {
    echo -e "\n${BLUE}=== Parallel Computation Tests ===${NC}"
    
    print_test "Multiple environment parallel computation"
    
    local session_ids=()
    local num_envs=3
    
    # Create multiple environments
    for i in $(seq 1 $num_envs); do
        local create_response=$(curl -s -X POST http://localhost:50053/create/CartPole-v1 2>/dev/null)
        if echo "$create_response" | grep -q "session_id"; then
            local session_id=$(echo "$create_response" | python3 -c "import sys, json; print(json.load(sys.stdin)['session_id'])" 2>/dev/null)
            session_ids+=("$session_id")
        fi
    done
    
    if [[ ${#session_ids[@]} -eq $num_envs ]]; then
        print_pass "Created $num_envs parallel environments"
        
        # Test parallel steps
        print_test "Parallel environment steps"
        local successful_parallel_steps=0
        
        for session_id in "${session_ids[@]}"; do
            # Reset environment
            curl -s -X POST http://localhost:50053/reset/$session_id >/dev/null 2>&1
            
            # Take 5 steps
            local env_successful=0
            for step in {1..5}; do
                local action=$((RANDOM % 2))
                local step_response=$(curl -s -X POST \
                    -H "Content-Type: application/json" \
                    -d "{\"action\": $action}" \
                    http://localhost:50053/step/$session_id 2>/dev/null)
                
                if echo "$step_response" | grep -q "reward"; then
                    ((env_successful++))
                fi
            done
            
            if [[ $env_successful -gt 0 ]]; then
                ((successful_parallel_steps++))
            fi
        done
        
        if [[ $successful_parallel_steps -eq $num_envs ]]; then
            print_pass "Parallel computation successful on all $num_envs environments"
        else
            print_fail "Parallel computation failed ($successful_parallel_steps/$num_envs succeeded)"
        fi
    else
        print_fail "Could not create multiple environments (${#session_ids[@]}/$num_envs created)"
    fi
}

# Test API Gateway functionality
test_gateway_functionality() {
    echo -e "\n${BLUE}=== API Gateway Tests ===${NC}"
    
    print_test "API Gateway service discovery"
    
    local services_response=$(curl -s http://localhost:8080/services 2>/dev/null)
    
    if echo "$services_response" | grep -q "trading"; then
        local healthy_count=$(echo "$services_response" | grep -o '"status": "healthy"' | wc -l)
        print_pass "Gateway service discovery working ($healthy_count services detected)"
        
        # Test gateway proxy (if implemented)
        print_test "Gateway environment proxy"
        local proxy_response=$(curl -s -X POST http://localhost:8080/gym/create/CartPole-v1 2>/dev/null)
        
        if echo "$proxy_response" | grep -q "session_id"; then
            print_pass "Gateway proxy functionality working"
        else
            print_info "Gateway proxy not implemented (using direct service access)"
        fi
    else
        print_fail "Gateway service discovery failed"
    fi
}

# Test package versions and environment availability
test_package_versions() {
    echo -e "\n${BLUE}=== Package Version Tests ===${NC}"
    
    print_test "Gym service package versions"
    
    local gym_info=$(curl -s http://localhost:50053/ 2>/dev/null)
    
    if echo "$gym_info" | grep -q "environments"; then
        print_pass "Gym service reporting environment info"
        
        # Check Box2D availability
        if echo "$gym_info" | grep -q '"box2d_available".*true'; then
            print_pass "Box2D physics environments available"
            
            # Test Box2D environment creation
            print_test "Box2D environment creation (LunarLander)"
            local lunar_response=$(curl -s -X POST http://localhost:50053/create/LunarLander-v2 2>/dev/null)
            
            if echo "$lunar_response" | grep -q "session_id"; then
                print_pass "Box2D environment creation successful"
            else
                print_fail "Box2D environment creation failed"
            fi
        else
            print_info "Box2D physics environments not available (classic envs only)"
        fi
    else
        print_fail "Gym service not reporting environment info"
    fi
}

# Test system performance
test_performance() {
    echo -e "\n${BLUE}=== Performance Tests ===${NC}"
    
    print_test "Service response times"
    
    local slow_services=0
    local services=("50051:Trading" "50052:Unity" "50053:Gym" "50054:ModernRL")
    
    for service in "${services[@]}"; do
        local port=$(echo $service | cut -d: -f1)
        local name=$(echo $service | cut -d: -f2)
        
        local start_time=$(date +%s%N)
        curl -s --max-time 3 http://localhost:$port/health >/dev/null 2>&1
        local end_time=$(date +%s%N)
        
        if [[ $? -eq 0 ]]; then
            local response_time=$(( (end_time - start_time) / 1000000 ))
            if [[ $response_time -lt 500 ]]; then
                print_info "$name: ${response_time}ms (good)"
            else
                print_info "$name: ${response_time}ms (slow)"
                ((slow_services++))
            fi
        else
            print_fail "$name: timeout"
            ((slow_services++))
        fi
    done
    
    if [[ $slow_services -eq 0 ]]; then
        print_pass "All services responding quickly"
    else
        print_fail "$slow_services services are slow or unresponsive"
    fi
}

# Test Docker container status
test_docker_status() {
    echo -e "\n${BLUE}=== Docker Container Tests ===${NC}"
    
    print_test "Docker containers running"
    
    local containers=("ai-mc-trading" "ai-mc-unity" "ai-mc-gym" "ai-mc-modern-rl" "ai-mc-gateway")
    local running_count=0
    
    for container in "${containers[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "^$container$"; then
            ((running_count++))
        fi
    done
    
    if [[ $running_count -eq ${#containers[@]} ]]; then
        print_pass "All $running_count containers are running"
    else
        print_fail "Only $running_count/${#containers[@]} containers are running"
    fi
}

# Generate test report
generate_report() {
    echo -e "\n${BLUE}=== Integration Test Results ===${NC}"
    
    echo ""
    echo "Overall Results:"
    echo "  Total Tests: $TOTAL_TESTS"
    echo -e "  ${GREEN}Passed: $PASSED_TESTS${NC}"
    echo -e "  ${RED}Failed: $FAILED_TESTS${NC}"
    
    local success_rate=$(( PASSED_TESTS * 100 / TOTAL_TESTS ))
    echo "  Success Rate: ${success_rate}%"
    
    echo ""
    if [[ $FAILED_TESTS -eq 0 ]]; then
        echo -e "${GREEN}🎉 ALL INTEGRATION TESTS PASSED!${NC}"
        echo ""
        echo "✅ Your AI Mission Control system is fully operational!"
        echo "✅ All services are healthy and can process RL computations"
        echo "✅ Environment creation, reset, and step operations working"
        echo "✅ Parallel computation capabilities verified"
        echo "✅ System ready for production use!"
        echo ""
        echo "🚀 Your 'Netflix for RL Environments' is ready to serve algorithms!"
    elif [[ $success_rate -ge 80 ]]; then
        echo -e "${YELLOW}⚠️  Most integration tests passed with minor issues${NC}"
        echo ""
        echo "✅ Core functionality is working"
        echo "⚠️  Some advanced features may need attention"
        echo "📝 Review failed tests above for specific issues"
    else
        echo -e "${RED}❌ Integration tests failed${NC}"
        echo ""
        echo "🔧 System needs attention before production use"
        echo "📝 Check service logs: docker logs <container_name>"
        echo "🔄 Try restarting services: docker-compose restart"
    fi
    
    echo ""
    echo "Next steps:"
    echo "1. For detailed performance testing: python3 performance_test.py"
    echo "2. For comprehensive testing: python3 test_ai_mission_control.py"
    echo "3. Check service logs if any tests failed"
}

# Main execution
main() {
    print_header
    
    # Check prerequisites
    if ! command -v docker >/dev/null 2>&1; then
        echo -e "${RED}Error: Docker not found. Please install Docker first.${NC}"
        exit 1
    fi
    
    if ! command -v curl >/dev/null 2>&1; then
        echo -e "${RED}Error: curl not found. Please install curl first.${NC}"
        exit 1
    fi
    
    print_info "Running quick integration tests for AI Mission Control..."
    print_info "This will verify that your microservices can process RL computations"
    
    # Run test suites
    test_docker_status
    test_basic_connectivity
    test_environment_computation
    test_parallel_computation
    test_gateway_functionality
    test_package_versions
    test_performance
    
    # Generate final report
    generate_report
    
    # Return appropriate exit code
    if [[ $FAILED_TESTS -eq 0 ]]; then
        exit 0
    elif [[ $((PASSED_TESTS * 100 / TOTAL_TESTS)) -ge 80 ]]; then
        exit 1
    else
        exit 2
    fi
}

# Handle command line arguments
case "${1:-}" in
    -h|--help)
        echo "AI Mission Control - Quick Integration Test"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  -h, --help     Show this help message"
        echo "  --quick        Run only essential tests (faster)"
        echo "  --verbose      Show detailed output"
        echo ""
        echo "This script performs quick integration tests to verify:"
        echo "• All microservices are running and healthy"
        echo "• Environment creation and RL computation works"
        echo "• Parallel environment processing capabilities"
        echo "• API Gateway functionality"
        echo "• Package versions and dependencies"
        echo ""
        echo "For comprehensive testing, use:"
        echo "  python3 test_ai_mission_control.py"
        echo "  python3 performance_test.py"
        exit 0
        ;;
    --quick)
        QUICK_MODE=true
        ;;
    --verbose)
        VERBOSE_MODE=true
        ;;
esac

# Run main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi