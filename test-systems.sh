#!/bin/bash

# AI Mission Control - Comprehensive Test Suite
# Tests all services, environments, and functionality

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

print_header() {
    echo -e "${BLUE}"
    echo "=========================================================="
    echo "  AI Mission Control - Comprehensive Test Suite"
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

print_section() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Test if service is running and healthy
test_service_health() {
    local service_name=$1
    local port=$2
    local timeout=${3:-5}
    
    print_test "Service health: $service_name (port $port)"
    
    # Use a more reliable curl command with timeout
    local response=$(timeout $timeout curl -s http://localhost:$port/health 2>/dev/null || echo "TIMEOUT")
    
    if [[ "$response" != "TIMEOUT" ]] && echo "$response" | grep -q "healthy"; then
        print_pass "$service_name is healthy"
        return 0
    else
        print_fail "$service_name health check failed"
        return 1
    fi
}

# Test service API endpoints
test_service_api() {
    local service_name=$1
    local port=$2
    
    print_test "API endpoints: $service_name"
    
    # Test root endpoint
    local response=$(curl -s http://localhost:$port/ 2>/dev/null)
    if [[ $? -eq 0 ]] && echo "$response" | grep -q "message"; then
        print_pass "$service_name API root endpoint working"
    else
        print_fail "$service_name API root endpoint failed"
        return 1
    fi
    
    # Test if response is valid JSON
    if echo "$response" | python3 -m json.tool >/dev/null 2>&1; then
        print_pass "$service_name returns valid JSON"
    else
        print_fail "$service_name returns invalid JSON"
        return 1
    fi
    
    return 0
}

# Test Docker containers
test_docker_containers() {
    print_section "Docker Container Tests"
    
    local containers=("ai-mc-trading" "ai-mc-unity" "ai-mc-gym" "ai-mc-modern-rl" "ai-mc-gateway")
    
    for container in "${containers[@]}"; do
        print_test "Container status: $container"
        
        # Use a simpler, more reliable check
        if docker ps | grep -q "$container"; then
            print_pass "$container is running"
        else
            print_fail "$container is not running"
        fi
    done
}

# Test all services
test_all_services() {
    print_section "Service Health Tests"
    
    # Test each service
    test_service_health "Trading Service" 50051
    test_service_health "Unity Service" 50052  
    test_service_health "Gym Service" 50053
    test_service_health "Modern RL Service" 50054
    test_service_health "API Gateway" 8080
}

# Test service APIs
test_all_apis() {
    print_section "Service API Tests"
    
    test_service_api "Trading Service" 50051
    test_service_api "Unity Service" 50052
    test_service_api "Gym Service" 50053
    test_service_api "Modern RL Service" 50054
    test_service_api "API Gateway" 8080
}

# Test specific gym functionality
test_gym_functionality() {
    print_section "Gym Service Functionality Tests"
    
    print_test "Gym service environment listing"
    local gym_response=$(curl -s http://localhost:50053/ 2>/dev/null)
    
    if echo "$gym_response" | grep -q "environments"; then
        print_pass "Gym service lists available environments"
        
        # Check Box2D availability
        if echo "$gym_response" | grep -q '"box2d_available".*true'; then
            print_pass "Box2D physics environments are available"
            print_info "Physics environments: LunarLander-v2, BipedalWalker-v3"
        elif echo "$gym_response" | grep -q '"box2d_available".*false'; then
            print_fail "Box2D physics environments are NOT available"
            print_info "Only classic environments: CartPole-v1, MountainCar-v0"
        else
            print_fail "Cannot determine Box2D availability"
        fi
    else
        print_fail "Gym service environment listing failed"
    fi
    
    # Test environment creation
    print_test "Gym environment creation (CartPole-v1)"
    local create_response=$(curl -s -X POST http://localhost:50053/create/CartPole-v1 2>/dev/null)
    
    if echo "$create_response" | grep -q "session_id"; then
        print_pass "CartPole-v1 environment creation successful"
    else
        print_fail "CartPole-v1 environment creation failed"
    fi
}

# Test API Gateway functionality
test_gateway_functionality() {
    print_section "API Gateway Functionality Tests"
    
    print_test "API Gateway service discovery"
    local services_response=$(curl -s http://localhost:8080/services 2>/dev/null)
    
    if [[ $? -eq 0 ]] && echo "$services_response" | grep -q "trading"; then
        print_pass "API Gateway service discovery working"
        
        # Count healthy services
        local healthy_count=$(echo "$services_response" | grep -o '"status": "healthy"' | wc -l)
        print_info "Healthy services detected: $healthy_count/4"
        
        if [[ $healthy_count -eq 4 ]]; then
            print_pass "All environment services are healthy via gateway"
        else
            print_fail "Not all services are healthy via gateway"
        fi
    else
        print_fail "API Gateway service discovery failed"
    fi
}

# Test package versions in containers
test_package_versions() {
    print_section "Package Version Tests"
    
    # Test key packages in gym service
    print_test "Gym service package versions"
    
    local gym_packages=$(docker exec ai-mc-gym python -c "
import gym, numpy, pygame
print(f'gym={gym.__version__}')
print(f'numpy={numpy.__version__}') 
print(f'pygame={pygame.version.ver}')
try:
    import Box2D
    print('box2d=available')
except ImportError:
    print('box2d=not_available')
" 2>/dev/null)
    
    if echo "$gym_packages" | grep -q "gym=0.26.2"; then
        print_pass "Gym version 0.26.2 confirmed"
    else
        print_fail "Gym version mismatch"
    fi
    
    if echo "$gym_packages" | grep -q "numpy=1.26.4"; then
        print_pass "NumPy version 1.26.4 confirmed"
    else
        print_fail "NumPy version mismatch"
    fi
    
    if echo "$gym_packages" | grep -q "box2d=available"; then
        print_pass "Box2D package available in container"
    else
        print_fail "Box2D package not available in container"
    fi
}

# Test network connectivity between services
test_service_connectivity() {
    print_section "Service Connectivity Tests"
    
    print_test "Inter-service network connectivity"
    
    # Test if API gateway can reach other services
    local gateway_test=$(docker exec ai-mc-gateway python -c "
import asyncio
import httpx

async def test_connectivity():
    services = {
        'trading': 'http://trading-service:50051/health',
        'unity': 'http://unity-service:50052/health', 
        'gym': 'http://gym-service:50053/health',
        'modern_rl': 'http://modern-rl-service:50054/health'
    }
    
    async with httpx.AsyncClient() as client:
        for name, url in services.items():
            try:
                response = await client.get(url, timeout=5.0)
                if response.status_code == 200:
                    print(f'{name}=reachable')
                else:
                    print(f'{name}=unreachable')
            except:
                print(f'{name}=unreachable')

asyncio.run(test_connectivity())
" 2>/dev/null)
    
    local reachable_count=$(echo "$gateway_test" | grep -c "reachable")
    if [[ $reachable_count -eq 4 ]]; then
        print_pass "All services reachable from API gateway"
    else
        print_fail "Not all services reachable from API gateway ($reachable_count/4)"
    fi
}

# Test performance and response times
test_performance() {
    print_section "Performance Tests"
    
    local services=("50051" "50052" "50053" "50054" "8080")
    local service_names=("Trading" "Unity" "Gym" "Modern RL" "Gateway")
    
    for i in "${!services[@]}"; do
        local port="${services[$i]}"
        local name="${service_names[$i]}"
        
        print_test "$name service response time"
        
        local start_time=$(date +%s%N)
        curl -s --max-time 5 http://localhost:$port/health >/dev/null 2>&1
        local end_time=$(date +%s%N)
        
        if [[ $? -eq 0 ]]; then
            local response_time=$(( (end_time - start_time) / 1000000 )) # Convert to milliseconds
            if [[ $response_time -lt 1000 ]]; then
                print_pass "$name responds in ${response_time}ms (good)"
            else
                print_fail "$name responds in ${response_time}ms (slow)"
            fi
        else
            print_fail "$name service timeout"
        fi
    done
}

# Create a simple Python test for the universal client
test_universal_client() {
    print_section "Universal Client Tests (Future)"
    
    print_info "Universal client library tests would go here"
    print_info "Testing ai_mission_control.create() with different environments"
    print_info "This requires implementing the actual gRPC client library"
    
    # For now, just test that we can reach the services
    print_test "Service endpoint accessibility for universal client"
    
    local all_reachable=true
    for port in 50051 50052 50053 50054; do
        if ! curl -s --max-time 3 http://localhost:$port/health >/dev/null 2>&1; then
            all_reachable=false
            break
        fi
    done
    
    if $all_reachable; then
        print_pass "All service endpoints accessible for universal client"
    else
        print_fail "Some service endpoints not accessible"
    fi
}

# Generate test report
generate_report() {
    print_section "Test Results Summary"
    
    echo ""
    echo -e "${BLUE}Overall Results:${NC}"
    echo "  Total Tests: $TOTAL_TESTS"
    echo -e "  ${GREEN}Passed: $PASSED_TESTS${NC}"
    echo -e "  ${RED}Failed: $FAILED_TESTS${NC}"
    
    local success_rate=$(( PASSED_TESTS * 100 / TOTAL_TESTS ))
    echo "  Success Rate: ${success_rate}%"
    
    echo ""
    if [[ $FAILED_TESTS -eq 0 ]]; then
        echo -e "${GREEN}🎉 ALL TESTS PASSED! Your AI Mission Control system is fully operational!${NC}"
        echo ""
        echo "✅ All services are healthy and responsive"
        echo "✅ Package versions are correct"  
        echo "✅ Service connectivity is working"
        echo "✅ API endpoints are functional"
        echo ""
        echo "Your 'Netflix for RL Environments' is ready to use!"
    else
        echo -e "${YELLOW}⚠️  Some tests failed. Check the details above.${NC}"
        echo ""
        echo "Common issues to check:"
        echo "1. All containers running: docker ps"
        echo "2. Service logs: make logs"
        echo "3. Port conflicts: netstat -tulpn | grep :8080"
    fi
}

# Main test execution
main() {
    case "${1:-}" in
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  -h, --help        Show this help message"
            echo "  --all             Run all tests (default)"
            echo "  --quick           Run only basic health checks"
            echo "  --full            Run all tests (same as --all)"
            echo "  --gym-only        Test only gym service"
            echo "  --performance     Run only performance tests"
            exit 0
            ;;
        --all|--full)
            FULL_MODE=true
            ;;
        --quick)
            QUICK_MODE=true
            ;;
        --gym-only)
            GYM_ONLY=true
            ;;
        --performance)
            PERFORMANCE_ONLY=true
            ;;
    esac
    
    print_header
    
    # Check if services are running
    if ! docker ps | grep -q "ai-mc-"; then
        echo -e "${RED}Error: AI Mission Control services are not running!${NC}"
        echo "Start them with: make up"
        exit 1
    fi
    
    print_info "Testing AI Mission Control system..."
    echo ""
    
    if [[ "${PERFORMANCE_ONLY:-}" == "true" ]]; then
        test_performance
    elif [[ "${GYM_ONLY:-}" == "true" ]]; then
        test_gym_functionality
        test_package_versions
    elif [[ "${QUICK_MODE:-}" == "true" ]]; then
        test_docker_containers
        test_all_services
    else
        # Full test suite
        test_docker_containers
        test_all_services
        test_all_apis
        test_gym_functionality
        test_gateway_functionality
        test_package_versions
        test_service_connectivity
        test_performance
        test_universal_client
    fi
    
    generate_report
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi